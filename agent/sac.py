import random
import torch
import torch.optim as optim
import torch.nn.functional as F

from collections import deque
from torch.optim.lr_scheduler import StepLR
from agent.network_sac import PolicyNetwork, QNetwork
from torch_geometric.data import Batch


class Agent:
    def __init__(self, meta_data,  # 그래프 구조에 대한 정보
                 state_size,  # 노드 타입 별 특성 벡터의 크기
                 num_nodes,  # 노드 타입 별 그래프 내 노드의 개수
                 embed_dim,  # node embedding 크기
                 num_heads,  # HGT layer에서의 attention head의 수
                 num_HGT_layers,  # HGT layer의 개수
                 num_actor_layers,  # actor layer의 개수
                 num_critic_layers,  # critic(Q) layer의 개수
                 lr,  # 학습률
                 lr_decay,  # 학습률에 대한 감소비율
                 lr_step,  # 학습률 감소를 위한 스텝 수
                 gamma,  # 감가율
                 buffer_size,  # replay buffer 크기
                 batch_size,  # 학습 배치 크기
                 min_buffer_size,  # 학습을 시작하기 위한 최소 buffer 크기
                 tau,  # target network에 대한 soft update 비율
                 alpha_init,  # 엔트로피 온도 파라미터의 초기값
                 target_entropy_ratio,  # 목표 엔트로피를 정할 때 사용하는 비율 (가능한 행동 수의 log값 대비)
                 use_gnn=True,
                 device="cpu"):

        self.gamma = gamma
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.tau = tau
        self.target_entropy_ratio = target_entropy_ratio
        self.device = device

        self.actor = PolicyNetwork(meta_data, state_size, num_nodes, embed_dim, num_heads,
                                   num_HGT_layers, num_actor_layers, use_gnn=use_gnn).to(device)
        self.critic1 = QNetwork(meta_data, state_size, num_nodes, embed_dim, num_heads,
                                num_HGT_layers, num_critic_layers, use_gnn=use_gnn).to(device)
        self.critic2 = QNetwork(meta_data, state_size, num_nodes, embed_dim, num_heads,
                                num_HGT_layers, num_critic_layers, use_gnn=use_gnn).to(device)
        self.critic1_target = QNetwork(meta_data, state_size, num_nodes, embed_dim, num_heads,
                                       num_HGT_layers, num_critic_layers, use_gnn=use_gnn).to(device)
        self.critic2_target = QNetwork(meta_data, state_size, num_nodes, embed_dim, num_heads,
                                       num_HGT_layers, num_critic_layers, use_gnn=use_gnn).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic1_target.eval()
        self.critic2_target.eval()

        self.log_alpha = torch.tensor(float(torch.log(torch.tensor(alpha_init))),
                                      requires_grad=True, device=device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        self.actor_scheduler = StepLR(optimizer=self.actor_optimizer, step_size=lr_step, gamma=lr_decay)
        self.critic_scheduler = StepLR(optimizer=self.critic_optimizer, step_size=lr_step, gamma=lr_decay)
        self.scheduler = self.actor_scheduler  # 로깅 시 참조하는 대표 학습률 스케줄러

        self.data = deque(maxlen=buffer_size)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def put_data(self, transition, crane_id):
        self.data.append(transition)

    def get_action(self, s, mask, crane_id):
        self.actor.eval()
        with torch.no_grad():
            action, log_prob = self.actor.act(s, mask, crane_id, greedy=False)
        return action, log_prob

    def step_schedulers(self):
        self.actor_scheduler.step()
        self.critic_scheduler.step()

    def train(self, crane_id):
        if len(self.data) < max(self.batch_size, self.min_buffer_size):
            return 0.0

        self.actor.train()
        self.critic1.train()
        self.critic2.train()

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

        # critic(Q1, Q2) 업데이트
        with torch.no_grad():
            next_probs, next_log_probs = self.actor.evaluate(s_prime, next_mask, crane_id)
            next_q1 = self.critic1_target.evaluate(s_prime)
            next_q2 = self.critic2_target.evaluate(s_prime)
            next_q_min = torch.min(next_q1, next_q2)
            next_v = (next_probs * (next_q_min - self.alpha * next_log_probs)).sum(dim=1, keepdim=True)
            target = r + self.gamma * next_v * done

        q1 = self.critic1.evaluate(s)
        q2 = self.critic2.evaluate(s)
        q1_taken = q1.gather(1, a)
        q2_taken = q2.gather(1, a)

        critic_loss = F.smooth_l1_loss(q1_taken, target) + F.smooth_l1_loss(q2_taken, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # actor(policy) 업데이트
        probs, log_probs = self.actor.evaluate(s, mask, crane_id)
        with torch.no_grad():
            q1_eval = self.critic1.evaluate(s)
            q2_eval = self.critic2.evaluate(s)
            q_min = torch.min(q1_eval, q2_eval)

        actor_loss = (probs * (self.alpha.detach() * log_probs - q_min)).sum(dim=1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 엔트로피 온도(alpha) 업데이트
        entropy = -(probs.detach() * log_probs.detach()).sum(dim=1, keepdim=True)
        num_valid_actions = mask.sum(dim=(1, 2)).clamp(min=1).float().unsqueeze(-1)
        target_entropy = self.target_entropy_ratio * torch.log(num_valid_actions)
        alpha_loss = -(self.log_alpha * (target_entropy - entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # target network soft update
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        return critic_loss.item() + actor_loss.item()

    def save_network(self, e, file_dir):
        torch.save({"episode": e,
                    "actor_state_dict": self.actor.state_dict(),
                    "critic1_state_dict": self.critic1.state_dict(),
                    "critic2_state_dict": self.critic2.state_dict(),
                    "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                    "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                    "log_alpha": self.log_alpha.detach().cpu()},
                   file_dir + "episode%d.pt" % e)
