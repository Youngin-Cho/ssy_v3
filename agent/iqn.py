import random
import torch
import torch.optim as optim

from collections import deque
from torch.optim.lr_scheduler import StepLR
from agent.network_iqn import QuantileNetwork
from torch_geometric.data import Batch


class Agent:
    def __init__(self, meta_data,  # 그래프 구조에 대한 정보
                 state_size,  # 노드 타입 별 특성 벡터의 크기
                 num_nodes,  # 노드 타입 별 그래프 내 노드의 개수
                 embed_dim,  # node embedding 크기
                 num_heads,  # HGT layer에서의 attention head의 수
                 num_HGT_layers,  # HGT layer의 개수
                 num_q_layers,  # state-action feature layer의 개수
                 n_cos,  # quantile fraction의 cosine 임베딩 차원
                 num_quantiles,  # 학습/행동 선택에 사용하는 quantile 샘플 수
                 kappa,  # quantile huber loss의 threshold
                 lr,  # 학습률
                 lr_decay,  # 학습률에 대한 감소비율
                 lr_step,  # 학습률 감소를 위한 스텝 수
                 gamma,  # 감가율
                 buffer_size,  # replay buffer 크기
                 batch_size,  # 학습 배치 크기
                 min_buffer_size,  # 학습을 시작하기 위한 최소 buffer 크기
                 target_update,  # target network 동기화 주기 (학습 스텝 단위)
                 epsilon_start,  # epsilon 초기값
                 epsilon_end,  # epsilon 최솟값
                 epsilon_decay,  # epsilon 감소율 (스텝 단위)
                 use_gnn=True,
                 device="cpu"):

        self.gamma = gamma
        self.num_quantiles = num_quantiles
        self.kappa = kappa
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.target_update = target_update
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.device = device

        self.network = QuantileNetwork(meta_data, state_size, num_nodes, embed_dim, num_heads,
                                       num_HGT_layers, num_q_layers, n_cos=n_cos, use_gnn=use_gnn).to(device)
        self.target_network = QuantileNetwork(meta_data, state_size, num_nodes, embed_dim, num_heads,
                                              num_HGT_layers, num_q_layers, n_cos=n_cos, use_gnn=use_gnn).to(device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.scheduler = StepLR(optimizer=self.optimizer, step_size=lr_step, gamma=lr_decay)

        self.data = deque(maxlen=buffer_size)
        self.train_step = 0

    def put_data(self, transition, crane_id):
        self.data.append(transition)

    def _epsilon(self):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            pow(self.epsilon_decay, self.train_step)

    def get_action(self, s, mask, crane_id):
        self.network.eval()
        with torch.no_grad():
            action, q_value = self.network.act(s, mask, crane_id, self.num_quantiles, epsilon=self._epsilon())
        return action, q_value

    def train(self, crane_id):
        if len(self.data) < max(self.batch_size, self.min_buffer_size):
            return 0.0

        self.network.train()
        batch = random.sample(self.data, self.batch_size)
        s_lst, a_lst, r_lst, s_prime_lst, mask_lst, next_mask_lst, done_lst = [], [], [], [], [], [], []

        for s, a, r, s_prime, mask, next_mask, done in batch:
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            mask_lst.append(mask.unsqueeze(0))
            next_mask_lst.append(next_mask.unsqueeze(0))
            done_lst.append([0 if done else 1])

        s = Batch.from_data_list(s_lst).to(self.device)
        a = torch.tensor(a_lst).to(self.device)
        r = torch.tensor(r_lst, dtype=torch.float).to(self.device)
        s_prime = Batch.from_data_list(s_prime_lst).to(self.device)
        mask = torch.concat(mask_lst).to(self.device)
        next_mask = torch.concat(next_mask_lst).to(self.device)
        done = torch.tensor(done_lst, dtype=torch.float).to(self.device)

        quantiles, current_taus = self.network.evaluate(s, self.num_quantiles)  # (batch, action_dim, N)
        a_index = a.unsqueeze(-1).expand(-1, -1, self.num_quantiles)  # (batch, 1, N)
        current_quantiles = quantiles.gather(1, a_index).squeeze(1)  # (batch, N)

        with torch.no_grad():
            next_quantiles, _ = self.target_network.evaluate(s_prime, self.num_quantiles)  # (batch, action_dim, N)
            next_mask_flat = next_mask.transpose(1, 2).flatten(1)  # (batch, action_dim)
            next_q_mean = next_quantiles.mean(dim=-1)
            next_q_mean = next_q_mean.masked_fill(~next_mask_flat, float('-inf'))
            best_action = next_q_mean.argmax(dim=1, keepdim=True)  # (batch, 1)
            best_action_index = best_action.unsqueeze(-1).expand(-1, -1, self.num_quantiles)
            target_quantiles = next_quantiles.gather(1, best_action_index).squeeze(1)  # (batch, N)
            target = r + self.gamma * target_quantiles * done  # (batch, N)

        # quantile huber loss
        td_error = target.unsqueeze(1) - current_quantiles.unsqueeze(2)  # (batch, N, N')
        abs_td_error = td_error.abs()
        huber_loss = torch.where(abs_td_error <= self.kappa,
                                 0.5 * td_error.pow(2),
                                 self.kappa * (abs_td_error - 0.5 * self.kappa))
        quantile_weight = (current_taus.unsqueeze(-1) - (td_error.detach() < 0).float()).abs()
        loss = (quantile_weight * huber_loss / self.kappa).mean(dim=2).sum(dim=1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_step += 1
        if self.train_step % self.target_update == 0:
            self.target_network.load_state_dict(self.network.state_dict())

        return loss.item()

    def save_network(self, e, file_dir):
        torch.save({"episode": e,
                    "model_state_dict": self.network.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict()},
                   file_dir + "episode%d.pt" % e)
