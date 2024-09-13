import string

from agent.gp_tree import *
from cfg_gp import get_cfg


if __name__ == '__main__':
    cfg = get_cfg()

    pop_size = cfg.pop_size
    min_depth = cfg.min_depth
    max_depth = cfg.max_depth
    generations = cfg.generations
    tournament_size = cfg.tournament_size
    xo_rate = cfg.xo_rate
    prob_mutation = cfg.prob_mutation

    data_dir = cfg.data_dir

    look_ahead = cfg.look_ahead

    functions = [add, sub, mul]
    terminals = ['f%d' % (i + 1) for i in range(2 * cfg.look_ahead + 6)]

    population = init_population(pop_size, min_depth, max_depth, xo_rate, prob_mutation, functions, terminals)
    best_of_run = None
    best_of_run_f = 0
    best_of_run_gen = 0
    fitnesses = [fitness(population[i], data_dir, cfg) for i in range(pop_size)]

    # go evolution!
    for gen in range(generations):
        print("generation %d" % gen)
        nextgen_population = []
        for i in range(pop_size):
            parent1 = selection(population, fitnesses, tournament_size)
            parent2 = selection(population, fitnesses, tournament_size)
            parent1.crossover(parent2)
            parent1.mutation()
            nextgen_population.append(parent1)
        population = nextgen_population
        fitnesses = [fitness(population[i], data_dir, cfg) for i in range(pop_size)]
        if max(fitnesses) > best_of_run_f:
            best_of_run_f = max(fitnesses)
            best_of_run_gen = gen
            best_of_run = deepcopy(population[fitnesses.index(max(fitnesses))])
            print("________________________")
            print("gen:", gen, ", best_of_run_f:", round(max(fitnesses), 3), ", best_of_run:")
            best_of_run.print_tree()

    print("\n\n_________________________________________________\nEND OF RUN\nbest_of_run attained at gen " + str(best_of_run_gen) + \
          " and has f=" + str(round(best_of_run_f, 3)))
    best_of_run.print_tree()