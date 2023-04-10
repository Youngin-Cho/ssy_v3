import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import Batch
from collections import deque
from agent.network import Network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')


class PrioritizedReplay(object):
    """
    Proportional Prioritization
    """

    def __init__(self, capacity, batch_size, gamma=0.99, n_step=1, alpha=0.6, beta_start=0.4, beta_frames=10000,
                 parallel_env=4):
        self.capacity = capacity
        self.batch_size = batch_size
        self.gamma = gamma
        self.n_step = n_step
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.parallel_env = parallel_env

        self.frame = 1  # for beta calculation
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.n_step_buffer = [deque(maxlen=self.n_step) for i in range(parallel_env)]
        self.iter_ = 0

    def calc_multistep_return(self, n_step_buffer):
        Return = 0
        for i in range(self.n_step):
            Return += self.gamma ** i * n_step_buffer[i][2]

        return n_step_buffer[0][0], n_step_buffer[0][1], Return, n_step_buffer[-1][3], n_step_buffer[-1][4]

    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def add(self, state, action, reward, next_state, done):
        if self.iter_ == self.parallel_env:
            self.iter_ = 0

        # n_step calc
        self.n_step_buffer[self.iter_].append((state, action, reward, next_state, done))
        if len(self.n_step_buffer[self.iter_]) == self.n_step:
            state, action, reward, next_state, done = self.calc_multistep_return(self.n_step_buffer[self.iter_])

        max_prio = self.priorities.max() if self.buffer else 1.0  # gives max priority if buffer is not empty else 1

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
        self.iter_ += 1

    def sample(self):
        N = len(self.buffer)
        if N == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        # calc P = p^a/sum(p^a)
        probs = prios ** self.alpha
        P = probs / probs.sum()

        # gets the indices depending on the probability p
        indices = np.random.choice(N, self.batch_size, p=P)
        samples = [self.buffer[idx] for idx in indices]

        beta = self.beta_by_frame(self.frame)
        self.frame += 1

        # Compute importance-sampling weight
        weights = (N * P[indices]) ** (-beta)
        # normalize weights
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        states, actions, rewards, next_states, dones = zip(*samples)
        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)


class Agent():
    def __init__(self, state_size, action_size, meta_data, look_ahead,
                 n_step, batch_size, capacity, lr, tau, gamma, N, worker):
        self.state_size = state_size
        self.action_size = action_size
        self.tau = tau
        self.N = N
        self.entropy_tau = 0.03
        self.lo = -1
        self.alpha = 0.9
        self.gamma = gamma

        self.batch_size = batch_size * worker
        self.Q_updates = 0
        self.n_step = n_step
        self.worker = worker
        self.update_every = worker
        self.last_action = None

        # IQN-Network
        self.qnetwork_local = Network(state_size, action_size, meta_data, look_ahead, N).to(device)
        self.qnetwork_target = Network(state_size, action_size, meta_data, look_ahead, N).to(device)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        # Replay memory
        self.memory = PrioritizedReplay(capacity, self.batch_size, gamma=self.gamma, n_step=n_step, parallel_env=worker)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        loss = None
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) >= self.batch_size:
                experiences = self.memory.sample()
                loss = self.learn(experiences)

        return loss

    def get_action(self, state, possible_actions, eps=0.0, noisy=True):
        state = Batch.from_data_list(state).to(device)

        if random.random() >= eps:  # select greedy action if random number is higher than epsilon or noisy network is used!
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local.get_qvalues(state, noisy=noisy)
            action_values = action_values.cpu().data.numpy()
            mask = np.ones_like(action_values)
            for i in range(len(possible_actions)):
                mask[i, possible_actions[i]] = 0.0
            const = 1.5 * (np.max(action_values) - np.min(action_values))
            action_values = action_values - const * mask
            # temp = action_values - np.inf * mask
            # action_values = np.where(np.isnan(temp), action_values, temp)
            action = np.argmax(action_values, axis=1)
        else:
            action = [random.choices(candidate)[0] for candidate in possible_actions]

        return action

    def learn(self, experiences):
        self.optimizer.zero_grad()

        states, actions, rewards, next_states, dones, idx, weights = experiences
        states = Batch.from_data_list(states).to(device)
        next_states = Batch.from_data_list(next_states).to(device)
        actions = torch.LongTensor(actions).to(device).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).to(device).unsqueeze(1)
        dones = torch.FloatTensor(dones).to(device).unsqueeze(1)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(device)

        Q_targets_next, _ = self.qnetwork_target(next_states, self.N)
        Q_targets_next = Q_targets_next.detach()  # (batch, num_tau, actions)
        q_t_n = Q_targets_next.mean(dim=1)
        # calculate log-pi
        logsum = torch.logsumexp((Q_targets_next - Q_targets_next.max(2)[0].unsqueeze(-1)) / self.entropy_tau, 2).unsqueeze(-1)
        assert logsum.shape == (self.batch_size, self.N, 1), "log pi next has wrong shape"
        tau_log_pi_next = Q_targets_next - Q_targets_next.max(2)[0].unsqueeze(-1) - self.entropy_tau * logsum

        pi_target = F.softmax(q_t_n / self.entropy_tau, dim=1).unsqueeze(1)

        Q_target = (self.gamma ** self.n_step * (pi_target * (Q_targets_next - tau_log_pi_next) * (1 - dones.unsqueeze(-1))).sum(2)).unsqueeze(1)
        assert Q_target.shape == (self.batch_size, 1, self.N)

        q_k_target = self.qnetwork_target.get_qvalues(states, noisy=True).detach()
        v_k_target = q_k_target.max(1)[0].unsqueeze(-1)
        tau_log_pik = q_k_target - v_k_target - self.entropy_tau * torch.logsumexp((q_k_target - v_k_target) / self.entropy_tau, 1).unsqueeze(-1)

        assert tau_log_pik.shape == (self.batch_size, self.action_size), "shape instead is {}".format(tau_log_pik.shape)
        munchausen_addon = tau_log_pik.gather(1, actions)

        # calc munchausen reward:
        munchausen_reward = (rewards + self.alpha * torch.clamp(munchausen_addon, min=self.lo, max=0)).unsqueeze(-1)
        assert munchausen_reward.shape == (self.batch_size, 1, 1)
        # Compute Q targets for current states
        Q_targets = munchausen_reward + Q_target
        # Get expected Q values from local model
        q_k, taus = self.qnetwork_local(states, self.N, noisy=True)
        Q_expected = q_k.gather(2, actions.unsqueeze(-1).expand(self.batch_size, self.N, 1))
        assert Q_expected.shape == (self.batch_size, self.N, 1)

        # Quantile Huber loss
        td_error = Q_targets - Q_expected
        assert td_error.shape == (self.batch_size, self.N, self.N), "wrong td error shape"
        huber_l = calculate_huber_loss(td_error, 1.0)
        quantil_l = abs(taus - (td_error.detach() < 0).float()) * huber_l / 1.0

        loss = quantil_l.sum(dim=1).mean(dim=1, keepdim=True) * weights  # , keepdim=True if per weights get multipl
        loss = loss.mean()

        # Minimize the loss
        loss.backward()
        clip_grad_norm_(self.qnetwork_local.parameters(), 1)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        # update priorities
        td_error = td_error.sum(dim=1).mean(dim=1, keepdim=True)  # not sure about this -> test
        self.memory.update_priorities(idx, abs(td_error.data.cpu().numpy()))
        return loss.detach().cpu().numpy()

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save(self, episode, file_dir):
        torch.save({"episode": episode,
                    "model_state_dict": self.qnetwork_target.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict()},
                   file_dir + "frame-%d.pt" % episode)

def calculate_huber_loss(td_errors, k=1.0):
    loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
    return loss