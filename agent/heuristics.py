import random


def minimize_avoiding_time(state, mask):
    num_cranes = mask.size(0)
    if len(state["piles_ma"]) > 0:
        action = random.choice(state["piles_ma"])
    else:
        action = random.choice(state["piles_all"])
    return action[0] * num_cranes + action[1]


def shortest_distance(state, mask):
    num_cranes = mask.size(0)
    action = random.choice(state["piles_sd"])
    return action[0] * num_cranes + action[1]

def mixed_heruistic(state, mask):
    num_cranes = mask.size(0)
    if len(state["piles_mx"]) > 0:
        action = random.choice(state["piles_mx"])
    else:
        action = random.choice(state["piles_sd"])
    return action[0] * num_cranes + action[1]

def random_selection(state, mask):
    num_cranes = mask.size(0)
    action = random.choice(state["piles_all"])
    return action[0] * num_cranes + action[1]