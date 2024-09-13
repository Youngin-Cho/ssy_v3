# tiny genetic programming by © moshe sipper, www.moshesipper.com
import os
import string
import numpy as np
import pandas as pd

from random import random, randint
from statistics import mean
from copy import deepcopy

from environment.env import SteelStockYard


def add(x, y): return x + y


def sub(x, y): return x - y


def mul(x, y): return x * y


class GPTree:
    def __init__(self, min_depth, xo_rate, prob_mutation, functions, terminals, data=None, left=None, right=None):
        self.min_depth = min_depth
        self.xo_rate = xo_rate
        self.prob_mutation = prob_mutation
        self.functions = functions
        self.terminals = terminals
        self.data = data
        self.left = left
        self.right = right

    def node_label(self):  # string label
        if (self.data in self.functions):
            return self.data.__name__
        else:
            return str(self.data)

    def print_tree(self, prefix=""):  # textual printout
        print("%s%s" % (prefix, self.node_label()))
        if self.left:  self.left.print_tree(prefix + "   ")
        if self.right: self.right.print_tree(prefix + "   ")

    def compute_tree(self, state):
        if self.data in self.functions:
            return self.data(self.left.compute_tree(state), self.right.compute_tree(state))
        else:
            idx = int(self.data[-1])
            return state[:, idx - 1]

    def random_tree(self, grow, max_depth, depth=0):  # create random tree using either grow or full method
        if depth < self.min_depth or (depth < max_depth and not grow):
            self.data = self.functions[randint(0, len(self.functions) - 1)]
        elif depth >= max_depth:
            self.data = self.terminals[randint(0, len(self.terminals) - 1)]
        else:  # intermediate depth, grow
            if random() > 0.5:
                self.data = self.terminals[randint(0, len(self.terminals) - 1)]
            else:
                self.data = self.terminals[randint(0, len(self.terminals) - 1)]
        if self.data in self.functions:
            self.left = GPTree(self.min_depth, self.xo_rate, self.prob_mutation, self.functions, self.terminals)
            self.left.random_tree(grow, max_depth, depth=depth + 1)
            self.right = GPTree(self.min_depth, self.xo_rate, self.prob_mutation, self.functions, self.terminals)
            self.right.random_tree(grow, max_depth, depth=depth + 1)

    def mutation(self):
        if random() < self.prob_mutation:  # mutate at this node
            self.random_tree(grow=True, max_depth=2)
        elif self.left:
            self.left.mutation()
        elif self.right:
            self.right.mutation()

    def size(self):  # tree size in nodes
        if self.data in self.terminals: return 1
        l = self.left.size() if self.left else 0
        r = self.right.size() if self.right else 0
        return 1 + l + r

    def build_subtree(self):  # count is list in order to pass "by reference"
        t = GPTree()
        t.data = self.data
        if self.left:  t.left = self.left.build_subtree()
        if self.right: t.right = self.right.build_subtree()
        return t

    def scan_tree(self, count, second):  # note: count is list, so it's passed "by reference"
        count[0] -= 1
        if count[0] <= 1:
            if not second:  # return subtree rooted here
                return self.build_subtree()
            else:  # glue subtree here
                self.data = second.data
                self.left = second.left
                self.right = second.right
        else:
            ret = None
            if self.left and count[0] > 1: ret = self.left.scan_tree(count, second)
            if self.right and count[0] > 1: ret = self.right.scan_tree(count, second)
            return ret

    def crossover(self, other):  # xo 2 trees at random nodes
        if random() < self.xo_rate:
            second = other.scan_tree([randint(1, other.size())], None)  # 2nd random subtree
            self.scan_tree([randint(1, self.size())], second)  # 2nd subtree "glued" inside 1st tree


# end class GPTree

def init_population(pop_size, min_depth, max_depth, xo_rate, prob_mutation, functions, terminals):  # ramped half-and-half
    pop = []
    for md in range(10, max_depth + 1):
        for i in range(int(pop_size / 12)):
            t = GPTree(min_depth, xo_rate, prob_mutation, functions, terminals)
            t.random_tree(grow=True, max_depth=md)  # grow
            pop.append(t)
        for i in range(int(pop_size / 12)):
            t = GPTree(min_depth, xo_rate, prob_mutation, functions, terminals)
            t.random_tree(grow=False, max_depth=md)  # full
            pop.append(t)
    return pop


def simulation(data_path, cfg):
    n_bays_in_area1 = cfg.n_bays_in_area1
    n_bays_in_area2 = cfg.n_bays_in_area2
    n_bays_in_area3 = cfg.n_bays_in_area3
    n_bays_in_area4 = cfg.n_bays_in_area4
    n_bays_in_area5 = cfg.n_bays_in_area5
    n_bays_in_area6 = cfg.n_bays_in_area6

    max_x = n_bays_in_area1 + n_bays_in_area2 + n_bays_in_area3 + n_bays_in_area4 + n_bays_in_area5 + n_bays_in_area6 + 4
    max_y = cfg.n_rows
    row_range = (string.ascii_uppercase[0], string.ascii_uppercase[cfg.n_rows - 1])
    bay_range = (1, n_bays_in_area1 + n_bays_in_area2 + n_bays_in_area3 + n_bays_in_area4 + n_bays_in_area5 + n_bays_in_area6)
    input_points = (1,)
    output_points = (1 + n_bays_in_area1 + n_bays_in_area2 + 1,
                     1 + n_bays_in_area1 + n_bays_in_area2 + n_bays_in_area3 + 2,
                     1 + n_bays_in_area1 + n_bays_in_area2 + n_bays_in_area3 + n_bays_in_area4 + n_bays_in_area5 + n_bays_in_area6 + 3)
    working_crane_ids = tuple()
    if cfg.is_crane1_working:
        working_crane_ids = working_crane_ids + ("Crane-1",)
    if cfg.is_crane2_working:
        working_crane_ids = working_crane_ids + ("Crane-2",)
    safety_margin = cfg.safety_margin

    multi_num = cfg.multi_num
    multi_w = cfg.multi_w
    multi_dis = cfg.multi_dis

    env = SteelStockYard(data_path, look_ahead=cfg.look_ahead,
                         max_x=max_x, max_y=max_y, row_range=row_range, bay_range=bay_range,
                         input_points=input_points, output_points=output_points,
                         working_crane_ids=working_crane_ids, safety_margin=safety_margin,
                         multi_num=multi_num, multi_w=multi_w, multi_dis=multi_dis,
                         algorithm="GP", record_events=False)

    return env


def fitness(individual, data_dir, cfg):
    data_paths = os.listdir(data_dir)
    lst_makespan = []
    for path in data_paths:
        env = simulation(data_dir + path, cfg)

        state, mask, crane_id = env.reset()
        done = False

        while not done:
            priority_score = individual.compute_tree(state)
            priority_score[~mask[crane_id]] = -float('inf')
            action = np.argmax(priority_score) * mask.shape[0] + crane_id

            next_state, reward, done, mask, next_crane_id = env.step(action)
            state = next_state
            crane_id = next_crane_id

            if done:
                makespan = env.model.env.now
                lst_makespan.append(makespan)
                break

    return sum(lst_makespan) / len(lst_makespan)


def selection(population, fitnesses, tournament_size):  # select one individual using tournament selection
    tournament = [randint(0, len(population) - 1) for i in range(tournament_size)]  # select tournament contenders
    tournament_fitnesses = [fitnesses[tournament[i]] for i in range(tournament_size)]
    return deepcopy(population[tournament[tournament_fitnesses.index(max(tournament_fitnesses))]])
