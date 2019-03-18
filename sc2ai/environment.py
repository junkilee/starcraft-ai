from pysc2.env import sc2_env
from multiprocessing import Pipe, Process
from pysc2.env.environment import StepType
import numpy as np
import time


class SCEnvironmentWrapper:
    def __init__(self, agent_interface, env_kwargs):
        self.env = sc2_env.SC2Env(**env_kwargs)
        self.render = env_kwargs['visualize']
        self.agent_interface = agent_interface
        self.done = False
        self.timestep = None

    def step(self, action):
        """
        :param action:
            The action, represented as a generator of pysc2 action objects, to take in the current
            state of the environment.
        :return:
            state: The state resulting after the action has been taken.
            done: Whether the action resulted in the environment reaching a terminal state.
        """
        if self.done:
            dummy_state, dummy_mask = self.agent_interface.dummy_state()
            return dummy_state, dummy_mask, np.nan, int(self.done)

        actions = self.agent_interface.convert_action(action)
        actions.__next__()
        action = actions.send(self.timestep)

        total_reward = 0
        while True:
            self.timestep = self.env.step([action])[0]
            if self.render:
                time.sleep(0.15)

            total_reward += self.timestep.reward
            self.done = int(self.timestep.step_type == StepType.LAST)
            state, action_mask = self.agent_interface.convert_state(self.timestep)

            action = actions.send(self.timestep)
            if self.done or action is None:
                return state, action_mask, total_reward, int(self.done)

    def reset(self):
        timestep = self.env.reset()[0]
        state, action_mask = self.agent_interface.convert_state(timestep)
        self.timestep = timestep
        self.done = False
        return state, action_mask,  0, int(self.done)

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
    def __init__(self, env_factory, num_instance=1):
        self.pipes = []
        self.processes = []
        self.num_instances = num_instance
        for process_id in range(num_instance):
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
        states, masks, rewards, dones = zip(*[pipe.recv() for pipe in self.pipes])
        return states, np.stack(masks), np.stack(rewards), np.stack(dones)

    def close(self):
        for pipe in self.pipes:
            pipe.send(('close', None))
        for process in self.processes:
            process.join()
