import tensorflow as tf
import numpy as np
import trfl
import os


class Rollout:
    """ Contains data needed for training from a single trajectory of the environment.

    Attributes:
        states: List of numpy arrays of shape [*state_shape], representing every state at which an action was taken.
        actions: List of action indices generated by the agent's step function.
        rewards: List of scalar rewards, representing the reward recieved after performing the corresponding action at
            the corresponding state.
        masks: List of masks generated by the environment.
        bootstrap_state: A numpy array of shape [*state_shape]. Represents the terminal state in the trajectory and is
            used to bootstrap the advantage estimation.
    """
    def __init__(self):
        self.states = []
        self.rewards = []
        self.actions = []
        self.masks = []
        self.should_bootstrap = None
        self.bootstrap_state = None
        self.done = False

    def total_reward(self):
        """
        :return: The current sum of rewards recieved in the trajectory.
        """
        return np.sum(self.rewards)

    def add_step(self, state, action=None, reward=None, mask=None, done=None):
        """ Saves a step generated by the agent to the rollout.

        Once `add_step` sees a `done`, it stops adding subsequent steps. However, make sure to call `add_step` at
        least one more time in order to record the terminal state for bootstrapping. Only leave the keyword parameters
        as None if feeding in the terminal state.

        :param state: The state which the action was taken in.
        :param action: The action index of the action taken, generated by the agent.
        :param reward: The reward recieved from the environment after taken the action.
        :param mask: The action mask that was used during the step.
        :param done: Whether the action resulted in the environment reaching a terminal state
        """
        if not self.done:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.masks.append(mask)
            self.done = done
        elif self.bootstrap_state is None:
            self.bootstrap_state = state


class ActorCriticLearner:
    """ Implementation of generalized advantage actor critic for TensorFlow.
    """
    def __init__(self, environment, agent,
                 save_dir="./",
                 load_model=False,
                 gamma=0.96,
                 td_lambda=0.96,
                 learning_rate=0.0003):
        """
        :param environment: An instance of `MultipleEnvironment` to be used to generate trajectories.
        :param agent: An instance of `ActorCriticAgent` to be used to generate actions.
        :param save_dir: The directory to store rewards and weights in.
        :param load_model: True if the model should be loaded from `save_dir`.
        :param gamma: The discount factor.
        :param td_lambda: The value of lambda used in generalized advantage estimation. Set to 1 to behave like
            monte carlo returns.
        """
        self.env = environment
        self.num_games = self.env.num_instances
        self.agent = agent
        self.discount_factor = gamma
        self.td_lambda = td_lambda

        self.save_dir = save_dir
        self.weights_path = os.path.join(save_dir, 'model.ckpt')
        self.rewards_path = os.path.join(save_dir, 'rewards.txt')
        self.episode_counter = 0

        self.rollouts = [Rollout() for _ in range(self.num_games)]
        with self.agent.graph.as_default():
            self.rewards_input = tf.placeholder(tf.float32, [None])
            self.loss = self._ac_loss()
            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
            self.session = self.agent.session
            self.session.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
            if load_model:
                try:
                    self.load_model()
                except ValueError:
                    pass
            else:
                open(self.rewards_path, 'w').close()

    def train_episode(self):
        """ Trains the agent for single episode for each environment in the `MultipleEnvironment`.

        Training is synchronized such that all training happens after all agents have finished acting in the
        environment. Call this method in a loop to train the agent.
        """
        self.generate_trajectory()
        for i in range(self.num_games):
            rollout = self.rollouts[i]
            if rollout.done:
                feed_dict = {
                    self.rewards_input: rollout.rewards,
                    **self.agent.get_feed_dict(rollout.states, rollout.masks, rollout.actions, rollout.bootstrap_state)
                }

                loss, _ = self.session.run([self.loss, self.train_op], feed_dict=feed_dict)
                self._log_data(rollout.total_reward())
                self.rollouts[i] = Rollout()

    def generate_trajectory(self):
        """
        Repeatedly generates actions from the agent and steps in the environment until all environments have reached a
        terminal state. Stores the complete result from each trajectory in `rollouts`.
        """
        states, masks, _, _ = self.env.reset()
        memory = None
        while True:
            action_indices, memory = self.agent.step(states, masks, memory)
            new_states, new_masks, rewards, dones = self.env.step(action_indices)

            for i, rollout in enumerate(self.rollouts):
                rollout.add_step(states[i], action_indices[i], rewards[i], masks[i], dones[i])
            states = new_states
            masks = new_masks
            if all(dones):
                # Add in the done state for rollouts which just finished for calculating the bootstrap value.
                for i, rollout in enumerate(self.rollouts):
                    rollout.add_step(states[i])
                return

    def save_model(self):
        """
        Saves the current model weights in current `save_path`.
        """
        save_path = self.saver.save(self.session, self.weights_path)
        print("Model Saved in %s" % save_path)

    def load_model(self):
        """
        Loads the model from weights stored in the current `save_path`.
        """
        self.saver.restore(self.session, self.weights_path)
        print('Model Loaded')

    def _log_data(self, reward):
        self.episode_counter += 1
        with open(self.rewards_path, 'a+') as f:
            f.write('%d\n' % reward)

        if self.episode_counter % 50 == 0:
            self.save_model()

    def _ac_loss(self):
        num_steps = tf.shape(self.rewards_input)[0]
        discounts = tf.ones((num_steps, 1)) * self.discount_factor
        rewards = tf.expand_dims(self.rewards_input, axis=1)

        values = tf.expand_dims(self.agent.train_values(), axis=1)
        bootstrap = tf.expand_dims(self.agent.bootstrap_value(), axis=0)
        glr = trfl.generalized_lambda_returns(rewards, discounts, values, bootstrap, lambda_=self.td_lambda)
        advantage = tf.squeeze(glr - values)

        loss_actor = tf.reduce_mean(-tf.stop_gradient(advantage) * self.agent.train_log_probs())
        loss_critic = tf.reduce_mean(advantage ** 2)
        result = loss_actor + 0.5 * loss_critic
        return result
