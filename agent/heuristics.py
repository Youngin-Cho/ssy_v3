import random


def SETT(state, mask):
    num_cranes = mask.size(0)
    action = random.choice(state["piles_SETT"])
    return action[0] * num_cranes + action[1]


def NCR(state, mask):
    num_cranes = mask.size(0)
    if len(state["piles_NCR"]) > 0:
        action = random.choice(state["piles_NCR"])
    else:
        action = random.choice(state["piles_all"])
    return action[0] * num_cranes + action[1]


# def mixed_heruistic(state, mask):
#     num_cranes = mask.size(0)
#     if len(state["piles_mx"]) > 0:
#         action = random.choice(state["piles_mx"])
#     else:
#         action = random.choice(state["piles_sd"])
#     return action[0] * num_cranes + action[1]


def TDD(state, mask):
    num_cranes = mask.size(0)
    if len(state["piles_TDD"]) > 0:
        action = random.choice(state["piles_TDD"])
    else:
        action = random.choice(state["piles_all"])
    return action[0] * num_cranes + action[1]


def TDT(state, mask):
    num_cranes = mask.size(0)
    if len(state["piles_TDT"]) > 0:
        action = random.choice(state["piles_TDT"])
    else:
        action = random.choice(state["piles_all"])
    return action[0] * num_cranes + action[1]


def RAND(state, mask):
    num_cranes = mask.size(0)
    action = random.choice(state["piles_all"])
    return action[0] * num_cranes + action[1]