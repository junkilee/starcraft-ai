from pysc2.env import sc2_env
from multiprocessing import Pipe, Process
from pysc2.env.environment import StepType
import numpy as np
import time


class SCEnvironmentWrapper:
    def __init__(self, env_kwargs):
        self.env = sc2_env.SC2Env(**env_kwargs)
        self.render = env_kwargs['visualize']
        self.done = False
        self.timestep = None
        self.num_parallel_instances = 1

    def step(self, action_list):
        """
        :param action:
            The action, represented as a generator of pysc2 action objects, to take in the current
            state of the environment.
        :return:
            env_state: The state resulting after the action has been taken.
            total_reward: The accumulated reward from the environment
            done: Whether the action resulted in the environment reaching a terminal state.
        """
        if self.done:
            return None, np.nan, int(self.done)


        total_reward = 0
        for action in action_list:
            self.timestep = self.env.step([action])[0]
            # if self.render:
            #     time.sleep(0.15)

            total_reward += self.timestep.reward
            self.done = int(self.timestep.step_type == StepType.LAST)

            if self.done:
                break
                
        return self.timestep, total_reward, int(self.done)


    def reset(self):
        self.timestep = self.env.reset()[0]
        self.done = False
        return self.timestep, 0, int(self.done)


    def close(self):
        self.env.__exit__(None, None, None)


def run_process(env_factory, pipe):
    environment = env_factory()

    while True:
        endpoint, data = pipe.recv()

        if endpoint == 'step':
            pipe.send(environment.step(data))
        elif endpoint == 'reset':
            pipe.send(environment.reset())
        elif endpoint == 'close':
            environment.close()
            pipe.close()
        else:
            raise Exception("Unsupported endpoint")


class MultipleEnvironment:
    def __init__(self, env_factory, num_parallel_instances=1):
        self.pipes = []
        self.processes = []
        self.num_parallel_instances = num_parallel_instances
        for process_id in range(num_parallel_instances):
            parent_conn, child_conn = Pipe()
            self.pipes.append(parent_conn)
            p = Process(target=run_process, args=(env_factory, child_conn,))
            self.processes.append(p)
            p.start()

    def step(self, actions):
        for pipe, action in zip(self.pipes, actions):
            pipe.send(('step', action))
        return self.get_results()

    def reset(self):
        for pipe in self.pipes:
            pipe.send(('reset', None))
        return self.get_results()

    def get_results(self):
        env_states, rewards, dones = zip(*[pipe.recv() for pipe in self.pipes])
        return env_states, np.stack(rewards), np.stack(dones)

    def close(self):
        for pipe in self.pipes:
            pipe.send(('close', None))
        for process in self.processes:
            process.join()
