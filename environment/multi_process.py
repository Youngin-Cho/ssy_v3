import numpy as np
from multiprocessing import Process, Pipe

from environment.env import *


def worker(remote, parent_remote, parameters_wrapper):
    parent_remote.close()
    parameters = parameters_wrapper.x
    env = SteelStockYard(look_ahead=parameters["look_ahead"],
                         bays=parameters["bays"],
                         num_of_cranes=parameters["num_of_cranes"],
                         num_of_storage_to_piles=parameters["num_of_storage_to_piles"],
                         num_of_reshuffle_from_piles=parameters["num_of_reshuffle_from_piles"],
                         num_of_reshuffle_to_piles=parameters["num_of_reshuffle_to_piles"],
                         num_of_retrieval_from_piles=parameters["num_of_retrieval_from_piles"])
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == "get_possible_actions":
            possible_actions = env.get_possible_actions()
            remote.send(possible_actions)
        elif cmd == 'close':
            remote.close()
            break
        else:
            raise NotImplementedError


class CloudpickleWrapper(object):
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class ParallelEnv:
    def __init__(self, nenvs, look_ahead=2, bays=("A", "B"), num_of_cranes=1,
                 num_of_storage_to_piles=10, num_of_reshuffle_from_piles=10,
                 num_of_reshuffle_to_piles=20, num_of_retrieval_from_piles=4):
        self.nenvs = nenvs
        self.parameters = {"look_ahead": look_ahead,
                           "bays": bays,
                           "num_of_cranes": num_of_cranes,
                           "num_of_storage_to_piles": num_of_storage_to_piles,
                           "num_of_reshuffle_from_piles": num_of_reshuffle_from_piles,
                           "num_of_reshuffle_to_piles": num_of_reshuffle_to_piles,
                           "num_of_retrieval_from_piles": num_of_retrieval_from_piles}

        self.action_size = 2 * 40 + 2
        self.state_size = {"crane": len(bays) * 44, "pile": len(bays) * 44, "plate": len(bays) * 44}
        self.meta_data = (["crane", "pile", "plate"],
                          [("plate", "stacking", "plate"),
                           ("plate", "locating", "pile"),
                           ("pile", "moving", "crane")])

        self.waiting = False
        self.closed = False

        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(self.parameters)))
                   for (work_remote, remote) in zip(self.work_remotes, self.remotes)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones = zip(*results)
        return obs, rews, dones

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return [remote.recv() for remote in self.remotes]

    def get_possible_actions(self):
        for remote in self.remotes:
            remote.send(("get_possible_actions", None))
        return [remote.recv() for remote in self.remotes]

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
            self.closed = True

    def __len__(self):
        return self.nenvs