import random
import numpy as np


def shortest_distance(state, possible_actions, crane_id=0):
    crane_state = state.x_dict["crane"].numpy()[crane_id]
    pile_state = state.x_dict["pile"].numpy()
    crane_state = np.concatenate([crane_state[:22], crane_state[23:26], crane_state[27:43],
                                  crane_state[44:66], crane_state[67:70], crane_state[71:87]], axis=0)
    pile_state = np.concatenate([pile_state[:,:22], pile_state[:,23:26], pile_state[:,27:43],
                                 pile_state[:,44:66], pile_state[:,67:70], pile_state[:,71:87]], axis=1)
    target_pile = np.argmin(pile_state, axis=1) + 1

    mask1 = [i for i in range(len(crane_state)) if i not in target_pile]
    mask2 = [i for i in target_pile if i not in possible_actions]
    crane_state[mask1] = np.inf
    crane_state[mask2] = np.inf

    if np.all(crane_state == np.inf):
        idx = 0
    else:
        idx = random.choice(np.where(crane_state == np.min(crane_state))[0])

    return idx


def longest_distance(state, available_actions):
    state = np.concatenate([state[:, :21, -1], state[:, 22:25, -1], state[:, 26:43, -1]], axis=1)
    state = state.flatten()
    mask = [i for i in range(len(state)) if i not in available_actions]
    state[mask] = -np.inf
    idx = random.choice(np.where(state == np.max(state))[0])
    return idx
def random_selection(state, available_actions):
    idx = random.choice(available_actions)
    return idx