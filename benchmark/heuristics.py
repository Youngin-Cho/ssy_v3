import random
import numpy as np
from utilities import get_coord, get_moving_time


def minimize_avoiding_time(state, possible_actions, crane_id=0):
    x_velocity = 0.5
    safety_margin = 5
    pile_list = [row_id + str(col_id).rjust(2, '0') for row_id in ("A", "B") for col_id in range(0, 41)]
    x_coords = np.zeros(len(pile_list))

    if crane_id == 0:
        opposite_crane_id = 1
    else:
        opposite_crane_id = 0

    from_xcoord_crane = int(state.x_dict["crane"].numpy()[crane_id][0] * 44)
    from_xcoord_opposite_crane = int(state.x_dict["crane"].numpy()[opposite_crane_id][0] * 44)
    to_xcoord_opposite_crane = int(state.x_dict["crane"].numpy()[opposite_crane_id][1] * 44)

    if to_xcoord_opposite_crane == 0:
        actions_wo_interference = possible_actions
    else:
        actions_wo_interference = []
        for a in possible_actions:
            i = a - 1
            to_xcoord_crane = int(x_coords[i] * 44)

            moving_time_to_target_crane = get_moving_time(from_xcoord_crane, to_xcoord_crane)
            moving_time_to_target_opposite_crane = get_moving_time(from_xcoord_opposite_crane, to_xcoord_opposite_crane)

            moving_time_to_target = min(moving_time_to_target_crane, moving_time_to_target_opposite_crane)

            xcoord_crane = from_xcoord_crane + moving_time_to_target * x_velocity \
                           * np.sign(to_xcoord_crane - from_xcoord_crane)
            xcoord_opposite_crane = from_xcoord_opposite_crane + moving_time_to_target * x_velocity \
                                    * np.sign(to_xcoord_opposite_crane - from_xcoord_opposite_crane)

            if (crane_id == 0 and xcoord_crane > xcoord_opposite_crane - safety_margin) \
                    or (crane_id == 1 and xcoord_crane < xcoord_opposite_crane + safety_margin):
                pass
            else:
                actions_wo_interference.append(a)

    if 0 in possible_actions:
        idx = 0
    else:
        if len(actions_wo_interference) != 0:
            idx = random.choice(actions_wo_interference)
        else:
            idx = random.choice(possible_actions)

    return idx


def shortest_distance(state, possible_actions, crane_id=0):
    pile_list = [row_id + str(col_id).rjust(2, '0') for row_id in ("A", "B") for col_id in range(0, 41)]
    x_coords = np.zeros(len(pile_list))
    mask = []
    for i, pile_name in enumerate(pile_list):
        x_coords[i] = get_coord(pile_name)[0] / 44
        if not i + 1 in possible_actions:
            mask.append(i)

    x_crane = state.x_dict["crane"].numpy()[crane_id][0]
    distance = np.abs(x_coords - x_crane)
    distance[mask] = np.inf

    if np.all(distance == np.inf):
        idx = 0
    else:
        idx = random.choice(np.where(distance == np.min(distance))[0]) + 1

    return idx

def shortest_distance_pre(state, possible_actions, crane_id=0):
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


def longest_distance(state, possible_actions, crane_id=0):
    crane_state = state.x_dict["crane"].numpy()[crane_id]
    pile_state = state.x_dict["pile"].numpy()
    crane_state = np.concatenate([crane_state[:22], crane_state[23:26], crane_state[27:43],
                                  crane_state[44:66], crane_state[67:70], crane_state[71:87]], axis=0)
    pile_state = np.concatenate([pile_state[:, :22], pile_state[:, 23:26], pile_state[:, 27:43],
                                 pile_state[:, 44:66], pile_state[:, 67:70], pile_state[:, 71:87]], axis=1)
    target_pile = np.argmin(pile_state, axis=1) + 1

    mask1 = [i for i in range(len(crane_state)) if i not in target_pile]
    mask2 = [i for i in target_pile if i not in possible_actions]
    crane_state[mask1] = - np.inf
    crane_state[mask2] = - np.inf

    if np.all(crane_state == - np.inf):
        idx = 0
    else:
        idx = random.choice(np.where(crane_state == np.max(crane_state))[0])

    return idx


def random_selection(possible_actions):
    idx = random.choice(possible_actions)
    return idx