import random
import numpy as np
from version1.utilities import get_coord, get_moving_time


def minimize_avoiding_time(state, possible_actions, crane_id=1):
    x_velocity = 0.5
    safety_margin = 5
    pile_list = [row_id + str(col_id).rjust(2, '0') for row_id in ("A", "B") for col_id in range(0, 41)]
    x_coords = np.zeros(len(pile_list))

    if crane_id == 1:
        opposite_crane_id = 2
    else:
        opposite_crane_id = 1

    from_xcoord_crane = int(state.x_dict["crane"].numpy()[crane_id - 1][0] * 44)
    from_xcoord_opposite_crane = int(state.x_dict["crane"].numpy()[opposite_crane_id - 1][0] * 44)
    to_xcoord_opposite_crane = int(state.x_dict["crane"].numpy()[opposite_crane_id - 1][1] * 44)

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


def shortest_distance(state, possible_actions, crane_id=1):
    pile_list = [row_id + str(col_id).rjust(2, '0') for row_id in ("A", "B") for col_id in range(0, 41)]
    x_coords = np.zeros(len(pile_list))
    mask = []
    for i, pile_name in enumerate(pile_list):
        x_coords[i] = get_coord(pile_name)[0] / 44
        if not i + 1 in possible_actions:
            mask.append(i)

    x_crane = state.x_dict["crane"].numpy()[crane_id - 1][0]
    distance = np.abs(x_coords - x_crane)
    distance[mask] = np.inf

    if np.all(distance == np.inf):
        idx = 0
    else:
        idx = random.choice(np.where(distance == np.min(distance))[0]) + 1

    return idx


def random_selection(possible_actions):
    idx = random.choice(possible_actions)
    return idx