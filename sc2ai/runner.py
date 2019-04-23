import traceback
import os

from sc2ai.environment import MultipleEnvironment, SCEnvironmentWrapper
from sc2ai.tflearner.tflearner import ActorCriticLearner, Rollout
from sc2ai.tflearner.tf_agent import InterfaceAgent, ConvAgent, LSTMAgent
from sc2ai.env_interface import *

from pysc2.lib.features import PlayerRelative

import skvideo.io

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
            self.episode_count += 1

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

    def forward_pass(self, collect_metadata=True):
        """
        Repeatedly generates actions from the agent and steps in the environment until all environments have reached a
        terminal state. Returns each trajectory in the form of rollouts.
        """
        meta_collector = {}

        num_games = self.env.num_parallel_instances

        agent_states, agent_masks, _, dones = self.env.reset()
        rollouts = [Rollout() for _ in range(num_games)]
        memory = None

        while not all(dones):
            agent_input = agent_states[0] # Hack for extractors
            agent_actions, memory = self.agent.step(agent_input, agent_masks, memory)

            if collect_metadata:
                for key, frame in self.agent.meta.items():
                    if key not in meta_collector:
                        meta_collector[key] = []

                    meta_collector[key].append(frame[0]) # first of batch

            env_action_lists = [self.agent_interface.convert_action(action) for action in agent_actions]

            # Feed actions to environment
            next_agent_states, next_masks, rewards, dones = self.env.step(env_action_lists)

            # Record info in rollouts
            for i in range(num_games):
                rollouts[i].add_step(state=agent_input[i],
                                     mask=agent_masks[i],
                                     action=agent_actions[i],
                                     reward=rewards[i],
                                     done=dones[i])
            agent_states, agent_masks = next_agent_states, next_masks

        # Add terminal state in rollbacks
        for i in range(num_games):
            agent_input = agent_states[0] # Hack for extractors
            rollouts[i].add_step(state=agent_input[i])

        # if collect_metadata:
        #     # ----- CONV --------
        #     key = 'meta_final_conv'
        #     frames = meta_collector[key]

        #     # fix range
        #     arr = np.array(frames)
        #     arr_range = np.max(arr) - np.min(arr)
        #     arr = (arr - np.min(arr)) / arr_range

        #     # to int
        #     data = (arr * 255).astype(np.uint8)
        #     # n_frames, xd, yd, n_channels = data.shape
        #     data = data.transpose(0, 3, 1, 2) 
        #     _, n_channels, xd, yd = data.shape
        #     out_x_dim = int(n_channels / 2)

        #     print(data.shape) # (240, 16, 28, 28)
        #     reshaped = np.concatenate(data.reshape(-1, 2, xd*out_x_dim, yd).transpose(1,3,2,0)).transpose(2,0,1)
        #     print(reshaped.shape)
        #     skvideo.io.vwrite('vids/conv-{}.mp4'.format(self.episode_count), 
        #         reshaped, outputdict={"-pix_fmt":"yuv420p"})

        if collect_metadata:
            # ----- SPATIAL PROBS --------
            key = 'meta_spatial_probs'
            frames = np.array(meta_collector[key])
            # fix range
            frames_range = np.max(frames) - np.min(frames)
            arr = (frames - np.min(frames)) / frames_range

            # data = (arr * 255).astype(np.uint8)

            # print(PlayerRelative)
            input_states = np.array(meta_collector['meta_state_input']) # [frames,x,y,channels]
            # input_summed = np.sum(input_states, axis=-1)
            # input_range = np.max(input_summed) - np.min(input_summed) + 0.01
            # input_arr = (input_summed - np.min(input_summed)) / input_range

            # print(np.max(input_states), np.min(input_states), np.max(input_summed), np.min(input_summed))
            # print(input_range)
            # print(np.max(input_arr), np.min(input_arr))

            three_channel = np.stack([arr, input_states[:,:,:,PlayerRelative.NEUTRAL], input_states[:,:,:,PlayerRelative.SELF]], axis=-1)

            skvideo.io.vwrite('vids/conv-{}.mp4'.format(self.episode_count), 
                (three_channel * 255).astype(np.uint8), outputdict={"-pix_fmt":"yuv420p"})

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
