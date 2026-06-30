import random
import torch
import torch.optim as optim
import torch.nn.functional as F

from collections import deque
from torch.optim.lr_scheduler import StepLR
from agent.network_dqn import QNetwork
from torch_geometric.data import Batch


class Agent:
    def __init__(self, meta_data,  # 그래프 구조에 대한 정보
                 state_size,  # 노드 타입 별 특성 벡터의 크기
                 num_nodes,  # 노드 타입 별 그래프 내 노드의 개수
                 embed_dim,  # node embedding 크기
                 num_heads,  # HGT layer에서의 attention head의 수
                 num_HGT_layers,  # HGT layer의 개수
                 num_q_layers,  # Q-head layer의 개수
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
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.target_update = target_update
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.device = device

        self.network = QNetwork(meta_data, state_size, num_nodes, embed_dim, num_heads,
                                num_HGT_layers, num_q_layers, use_gnn=use_gnn).to(device)
        self.target_network = QNetwork(meta_data, state_size, num_nodes, embed_dim, num_heads,
                                       num_HGT_layers, num_q_layers, use_gnn=use_gnn).to(device)
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
            action, q_value = self.network.act(s, mask, crane_id, epsilon=self._epsilon())
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

        q_taken, _ = self.network.evaluate(s, a, mask, crane_id)
        with torch.no_grad():
            _, q_max_next = self.target_network.evaluate(s_prime, a, next_mask, crane_id)
            target = r + self.gamma * q_max_next * done

        loss = F.smooth_l1_loss(q_taken, target)

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
