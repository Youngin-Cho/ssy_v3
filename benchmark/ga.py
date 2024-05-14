import os
import string
import random
import pygad
import numpy as np
import pandas as pd

from cfg_train import get_cfg
from environment.simulation import Management


num_generations = 10000
num_parents_mating = 10
sol_per_pop = 20


def fitness_fun(ga_instanse, last_gen_fitness):
    print("on_fitness")
    model = Management(df_storage, df_reshuffle, df_retrieval,
                       max_x=max_x, max_y=max_y, row_range=cfg.row_range, bay_range=cfg.bay_range,
                       input_points=input_points, output_points=output_points,
                       working_crane_ids=working_crane_ids, safety_margin=safety_margin,
                       multi_num=multi_num, multi_w=multi_w, multi_dis=multi_dis,
                       record_events=record_events)


def crossover_func(parents, offspring_size, ga_instance):
    field_size = (offspring_size[0], int(offspring_size[1] / 2))

    offspring = np.empty(offspring_size, dtype=ga_instance.gene_type[0])
    offspring_part1 = np.empty(field_size, dtype=ga_instance.gene_type[0])
    offspring_part2 = np.empty(field_size, dtype=ga_instance.gene_type[0])

    # uniform crossover for genes in the first field
    genes_sources = np.random.randint(low=0, high=2, size=field_size)

    for k in range(offspring_size[0]):
        probs = np.random.random(size=parents.shape[0])
        indices = list(set(np.where(probs <= 0.2)[0]))

        if len(indices) == 0:
            offspring[k, :] = parents[k % parents.shape[0], :]
            continue
        elif len(indices) == 1:
            parent1_idx = indices[0]
            parent2_idx = parent1_idx
        else:
            indices = random.sample(indices, 2)
            parent1_idx = indices[0]
            parent2_idx = indices[1]

        for gene_idx in range(field_size[1]):
            if (genes_sources[k, gene_idx] == 0):
                offspring_part1[k, gene_idx] = parents[parent1_idx, gene_idx]
            elif (genes_sources[k, gene_idx] == 1):
                offspring_part1[k, gene_idx] = parents[parent2_idx, gene_idx]

    # partially mapped crossover for genes in the second field
    parents = parents[:, field_size:]
    crossover_points_1 = np.random.randint(low=0, high=np.ceil(parents.shape[1] / 2 + 1), size=offspring_size[0])
    crossover_points_2 = crossover_points_1 + int(parents.shape[1] / 2)

    for k in range(offspring_size[0]):
        probs = np.random.random(size=parents.shape[0])
        indices = list(set(np.where(probs <= 0.2)[0]))

        # If no parent satisfied the probability, no crossover is applied and a parent is selected.
        if len(indices) == 0:
            offspring[k, :] = parents[k % parents.shape[0], :]
            continue
        elif len(indices) == 1:
            parent1_idx = indices[0]
            parent2_idx = parent1_idx
        else:
            indices = random.sample(indices, 2)
            parent1_idx = indices[0]
            parent2_idx = indices[1]

        offspring_part2[k, 0:crossover_points_1[k]] = parents[parent1_idx, 0:crossover_points_1[k]]
        offspring_part2[k, crossover_points_2[k]:] = parents[parent1_idx, crossover_points_2[k]:]
        offspring[k, crossover_points_1[k]:crossover_points_2[k]] = parents[parent2_idx, crossover_points_1[k]:crossover_points_2[k]]

        p1, p2 = [0] * field_size[1], [0] * field_size[1]
        for g in range(field_size[1]):
            p1[parents[parent1_idx, g]] = g
            p2[parents[parent2_idx, g]] = g

        for g in range(crossover_points_1[k], crossover_points_2[k]):
            temp1 = parents[parent1_idx, g]
            temp2 = parents[parent2_idx, g]
            # Swap the matched value
            parents[parent1_idx, g], parents[parent1_idx, p1[temp2]] = temp2, temp1
            parents[parent2_idx, g], ind2[p2[temp1]] = temp1, temp2
            # Position bookkeeping
            p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
            p2[temp1], p2[temp2] = p2[temp2], p2[temp1]


    return offspring



    offspring = ...
    ...
    return np.array(offspring)


def mutation_func(offspring, ga_instance):
    ...
    return offspring


if __name__ == "__main__":
    cfg = get_cfg()
    random_seed = cfg.random_seed
    record_events = bool(cfg.record_events)

    n_bays_in_area1 = cfg.n_bays_in_area1
    n_bays_in_area2 = cfg.n_bays_in_area2
    n_bays_in_area3 = cfg.n_bays_in_area3
    n_bays_in_area4 = cfg.n_bays_in_area4
    n_bays_in_area5 = cfg.n_bays_in_area5
    n_bays_in_area6 = cfg.n_bays_in_area6

    max_x = n_bays_in_area1 + n_bays_in_area2 + n_bays_in_area3 + n_bays_in_area4 + n_bays_in_area5 + n_bays_in_area6 + 4
    max_y = cfg.n_rows
    row_range = (string.ascii_uppercase[0], string.ascii_uppercase[cfg.n_rows - 1])
    bay_range = (
    1, n_bays_in_area1 + n_bays_in_area2 + n_bays_in_area3 + n_bays_in_area4 + n_bays_in_area5 + n_bays_in_area6)
    input_points = (1,)
    output_points = (1 + n_bays_in_area1 + n_bays_in_area2 + 1,
                     1 + n_bays_in_area1 + n_bays_in_area2 + n_bays_in_area3 + 2,
                     1 + n_bays_in_area1 + n_bays_in_area2 + n_bays_in_area3 + n_bays_in_area4 + n_bays_in_area5 + n_bays_in_area6 + 3)
    working_crane_ids = tuple()
    if bool(cfg.is_crane1_working):
        working_crane_ids = working_crane_ids + ("Crane-1",)
    if bool(cfg.is_crane2_working):
        working_crane_ids = working_crane_ids + ("Crane-2",)
    safety_margin = cfg.safety_margin

    multi_num = cfg.multi_num
    multi_w = cfg.multi_w
    multi_dis = cfg.multi_dis

    data_dir = "../input/data/test/case0/instance-1.xlsx"
    test_paths = os.listdir(data_dir)
    index = ["P%d" % i for i in range(1, len(test_paths) + 1)] + ["avg"]
    columns = ["GA"]
    df_makespan = pd.DataFrame(index=index, columns=columns)
    df_empty_travel_time = pd.DataFrame(index=index, columns=columns)
    df_avoiding_time = pd.DataFrame(index=index, columns=columns)
    df_computing_time = pd.DataFrame(index=index, columns=columns)

    for name in columns:
        progress = 0
        list_makespan = []
        list_empty_travel_time = []
        list_avoiding_time = []
        list_computing_time = []

        for prob, path in zip(index, test_paths):
            random.seed(random_seed)

            df_storage = pd.read_excel(data_dir + path, sheet_name="storage", engine="openpyxl")
            df_reshuffle = pd.read_excel(data_dir + path, sheet_name="reshuffle", engine="openpyxl")
            df_retrieval = pd.read_excel(data_dir + path, sheet_name="retrieval", engine="openpyxl")

            makespan = 0.0
            empty_travel_time = 0.0
            avoiding_time = 0.0
            computing_time = 0.0

            num_plates = len(df_storage) + len(df_reshuffle)
            gen_space = [range(2) for _ in range(num_plates)] + [range(num_plates) for _ in range(num_plates)]
            first_field = np.vstack([np.random.randint(2, size=num_plates) for _ in range(sol_per_pop)])
            second_field = np.vstack([np.random.permutation(num_plates) for _ in range(sol_per_pop)])
            initial_population = np.hstack([first_field, second_field])

            ga_instance = pygad.GA(num_generations=num_generations,
                                   num_parents_mating=num_parents_mating,
                                   fitness_func=fitness_fun,
                                   initial_population=initial_population,
                                   gene_type=int,
                                   sol_per_pop=sol_per_pop,
                                   num_genes=num_plates * 2,
                                   parent_selection_type="tournament",
                                   K_tournament=2,
                                   mutation_percent_genes=0.01,
                                   mutation_type="random",
                                   mutation_by_replacement=True,
                                   random_mutation_min_val=0.0,
                                   random_mutation_max_val=1.0)