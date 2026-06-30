import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.optim.lr_scheduler import StepLR
from agent.network_a2c import ActorCritic
from torch_geometric.data import Batch


class Agent:
    def __init__(self, meta_data,  # 그래프 구조에 대한 정보
                 state_size,  # 노드 타입 별 특성 벡터의 크기
                 num_nodes,  # 노드 타입 별 그래프 내 노드의 개수
                 embed_dim,  # node embedding 크기
                 num_heads,  # HGT layer에서의 attention head의 수
                 num_HGT_layers,  # HGT layer의 개수
                 num_actor_layers,  # actor layer의 개수
                 num_critic_layers,  # critic layer의 개수
                 lr,  # 학습률
                 lr_decay,  # 학습률에 대한 감소비율
                 lr_step,  # 학습률 감소를 위한 스텝 수
                 gamma,  # 감가율
                 lmbda,  # gae 파라미터
                 V_coeff,  # 가치함수 학습에 대한 가중치
                 E_coeff,  # 엔트로피에 대한 가중치
                 use_gnn=True,
                 device="cpu"):

        self.gamma = gamma
        self.lmbda = lmbda
        self.V_coeff = V_coeff
        self.E_coeff = E_coeff
        self.device = device

        self.network = ActorCritic(meta_data, state_size, num_nodes, embed_dim, num_heads,
                                   num_HGT_layers, num_actor_layers, num_critic_layers, use_gnn=use_gnn).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.scheduler = StepLR(optimizer=self.optimizer, step_size=lr_step, gamma=lr_decay)

        self.data = []

    def put_data(self, transition, crane_id):
        self.data.append(transition)

    def make_batch(self, crane_id):
        s_lst, a_lst, r_lst, s_prime_lst, v_lst, mask_lst, done_lst \
            = [], [], [], [], [], [], []

        data = self.data[:]

        for i, transition in enumerate(data):
            s, a, r, s_prime, a_logprob, v, mask, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            if i > 0:
                v_lst.append([v])
            mask_lst.append(mask.unsqueeze(0))
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        if done_lst[-1] == 0:
            v_lst.append([0.0])
        else:
            with torch.no_grad():
                _, _, v = self.network.act(s_prime_lst[-1], mask, crane_id)
            v_lst.append([v])

        s, a, r, s_prime, v, mask, done \
            = (Batch.from_data_list(s_lst).to(self.device),
               torch.tensor(a_lst).to(self.device),
               torch.tensor(r_lst, dtype=torch.float).to(self.device),
               Batch.from_data_list(s_prime_lst).to(self.device),
               torch.tensor(v_lst, dtype=torch.float).to(self.device),
               torch.concat(mask_lst).to(self.device),
               torch.tensor(done_lst, dtype=torch.float).to(self.device))

        self.data = []

        return s, a, r, s_prime, v, mask, done

    def get_action(self, s, mask, crane_id):
        self.network.eval()
        with torch.no_grad():
            a = self.network.act(s, mask, crane_id, greedy=False)
        return a

    def train(self, crane_id):
        self.network.train()
        s, a, r, s_prime, v, mask, done = self.make_batch(crane_id)

        td_target = r + self.gamma * v * done
        delta = td_target - v

        advantage_lst = []
        advantage = 0.0
        for delta_t in delta.flip(dims=(0,)):
            advantage = self.gamma * self.lmbda * advantage + delta_t
            advantage_lst.append(advantage)
        advantage_lst.reverse()
        advantage = torch.concat(advantage_lst).unsqueeze(-1).to(self.device)

        a_logprob, new_v, dist_entropy = self.network.evaluate(s, a, mask, crane_id)

        policy_loss = - a_logprob * advantage.detach()
        value_loss = self.V_coeff * F.smooth_l1_loss(new_v, td_target)
        entropy_loss = - self.E_coeff * dist_entropy
        loss = policy_loss + value_loss + entropy_loss

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        return loss.mean().item()

    def save_network(self, e, file_dir):
        torch.save({"episode": e,
                    "model_state_dict": self.network.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict()},
                   file_dir + "episode%d.pt" % e)
