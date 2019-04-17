import traceback
import os

from sc2ai.environment import MultipleEnvironment, SCEnvironmentWrapper
from sc2ai.tflearner.tflearner import ActorCriticLearner, Rollout
from sc2ai.tflearner.tf_agent import InterfaceAgent, ConvAgent, LSTMAgent
from sc2ai.env_interface import *


# ------------------------ FACTORIES ------------------------

def build_agent_interface(map_name):
    if map_name in {'DefeatRoaches', 'StalkersVsRoaches'}:
        return EmbeddingInterfaceWrapper(RoachesEnvironmentInterface())
    elif map_name == 'MoveToBeacon':
        return BeaconEnvironmentInterface()
    elif map_name == 'BuildMarines':
        return EmbeddingInterfaceWrapper(TrainMarines())
    else:
        raise Exception('Unsupported Map')


# ------------------------ RUNNER ------------------------


class AgentRunner:
    def __init__(self, sc2map, runner_params, model_params, env_kwargs):
        self.sc2map = sc2map
        self.runner_params = runner_params
        self.model_params = model_params
        self.env_kwargs = env_kwargs

        self.rewards_path = os.path.join(self.runner_params['save_dir'], 'rewards.txt')
        self.save_every = self.model_params['save_every']

        # Clear rewards on new run with the same name
        if not self.model_params['load_model']:
            open(self.rewards_path, 'w').close()

        self.initialize()

    def initialize(self, reset=False):
        # Initializes interface, env, agent and learner. On reset we load the model
        self.agent_interface = build_agent_interface(self.sc2map)

        # Pass into MultipleEnvironment a factory to create SCEnvironments
        self.env = MultipleEnvironment(lambda: SCEnvironmentWrapper(self.agent_interface, self.env_kwargs),
                                       num_parallel_instances=self.runner_params['num_parallel_instances'])
        self.agent = ConvAgent(self.agent_interface)

        # On resets, always load the model
        load_model = True if reset else self.model_params['load_model']
        self.learner = ActorCriticLearner(self.env, self.agent,
                                          save_dir=self.runner_params['save_dir'],
                                          load_model=load_model,
                                          gamma=self.model_params['gamma'],
                                          td_lambda=self.model_params['td_lambda'],
                                          learning_rate=self.model_params['learning_rate'])

    # ------------------------ TRAINING ------------------------

    def train_agent(self, num_training_episodes, reset_env_every=1000):
        # Trains the agent for num_training_episodes
        self.episode_count = 0
        save_every = self.model_params['save_every']  # for convenience

        print("Runner: beinning training for", num_training_episodes, "episodes")
        while self.episode_count < num_training_episodes:

            # Resets the environment every once in a while (to deal with memory leak)
            if (self.episode_count % reset_env_every == 0) and (self.episode_count > 0):
                print("Runner: Decided to reset")
                self._reset()

            # Save every once in a while
            if self.episode_count % save_every == 0:
                self.learner.save_model()

            # Train this episode
            self.train_episode()

        print("Runner: training fininshed, trained", self.episode_count, "episodes")
        self._close_env()

    def train_episode(self):
        try:
            rollouts = self.forward_pass()  # Run the model
            self.learner.update_model(rollouts)  # Update the model
            self._log_rewards(rollouts)  # Log these rollouts

        except Exception as e:
            print("Runner: Encountered error in train_episode:", e)
            traceback.print_exc()
            self._reset()

    def forward_pass(self):
        """
        Repeatedly generates actions from the agent and steps in the environment until all environments have reached a
        terminal state. Returns each trajectory in the form of rollouts.
        """
        num_games = self.env.num_parallel_instances

        agent_states, agent_masks, _, dones = self.env.reset()
        rollouts = [Rollout() for _ in range(num_games)]
        memory = self.agent.get_initial_memory(num_games)

        while not all(dones):
            agent_actions, next_memory = self.agent.step(agent_states, agent_masks, memory)
            env_action_lists = self.agent_interface.convert_actions(agent_actions)

            # Feed actions to environment
            next_agent_states, next_masks, rewards, dones = self.env.step(env_action_lists)

            # Record info in rollouts
            for i in range(num_games):
                rollouts[i].add_step(state=agent_states[i],
                                     memory=memory[i],
                                     mask=agent_masks[i],
                                     action=agent_actions[i],
                                     reward=rewards[i],
                                     done=dones[i])
            agent_states, agent_masks = next_agent_states, next_masks
            memory = next_masks

        # Add terminal state in rollbacks
        for i in range(num_games):
            rollouts[i].add_step(state=agent_states[i], memory=memory)
        return rollouts

    # ------------------------ UTILS ------------------------

    def _log_rewards(self, rollouts):
        with open(self.rewards_path, 'a+') as f:
            for r in rollouts:
                f.write('%d\n' % r.total_reward())

    def _reset(self):
        # Shutdown env and reinitialize everything
        print("Runner: Resetting")
        self._close_env()
        self.initialize(reset=True)

    def _close_env(self):
        # Tell env to shutdown
        print("Runner: Shutting down environment")
        self.env.close()
