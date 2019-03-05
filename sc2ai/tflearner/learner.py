import tensorflow as tf
import numpy as np
import trfl
import sys
import os


class Rollout:
    def __init__(self):
        self.states = []
        self.rewards = []
        self.actions = []
        self.masks = []
        self.should_bootstrap = None
        self.bootstrap_state = None
        self.done = False

    def total_reward(self):
        return np.sum(self.rewards)

    def add_step(self, state, action=None, reward=None, mask=None, done=None):
        if not self.done:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.masks.append(mask)
            self.done = done
        elif self.bootstrap_state is None:
            self.bootstrap_state = state


class Learner:
    def __init__(self, environment, agent,
                 save_dir="./",
                 load_model=False,
                 gamma=0.96,
                 td_lambda=0.96):

        self.env = environment
        self.num_games = self.env.num_instances
        self.agent = agent
        self.discount_factor = gamma
        self.td_lambda = td_lambda

        self.save_dir = save_dir
        self.weights_path = os.path.join(save_dir, 'model.ckpt')
        self.rewards_path = os.path.join(save_dir, 'rewards.txt')
        self.episode_counter = 0

        with self.agent.graph.as_default():
            self.rewards_input = tf.placeholder(tf.float32, [None])
            self.loss = self.ac_loss()
            self.rollouts = [Rollout() for _ in range(self.num_games)]
            self.train_op = tf.train.AdamOptimizer(0.0005).minimize(self.loss)
            self.session = self.agent.session
            self.session.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
            if load_model:
                self.load_model(self.weights_path)
            else:
                open(self.rewards_path, 'w').close()

    def ac_loss(self):
        num_steps = tf.shape(self.rewards_input)[0]
        discounts = tf.ones((num_steps, 1)) * self.discount_factor
        bootstrap_values = tf.zeros((1,))  # TODO: replace with value for non finished games
        rewards = tf.expand_dims(self.rewards_input, axis=1)
        values = tf.expand_dims(self.agent.train_values(), axis=1)

        glr = trfl.generalized_lambda_returns(rewards, discounts, values, bootstrap_values, lambda_=self.td_lambda)
        advantage = tf.squeeze(glr)

        loss_actor = tf.reduce_mean(-advantage * self.agent.train_log_probs())
        loss_critic = tf.reduce_mean(advantage ** 2)
        self._state = self.agent.state_input
        self._mask = self.agent.mask_input
        self._action = self.agent.action_input
        self._spacial = self.agent.spacial_input
        self._loss_actor = loss_actor
        self._loss_critic = loss_critic
        self._advantage = advantage
        self._glr = glr
        self._values = values
        self._rewards = rewards
        result = loss_actor + 0.5 * loss_critic
        return result

    def generate_trajectory(self):
        states, masks, _, _ = self.env.reset()
        while True:
            action_indices = self.agent.step(states, masks)
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

    def save_model(self, path):
        save_path = self.saver.save(self.session, path)
        print("Model Saved in %s" % save_path)

    def load_model(self, path):
        self.saver.restore(self.session, path)
        print('Model Loaded')

    def train_episode(self):
        self.generate_trajectory()
        for i in range(self.num_games):
            rollout = self.rollouts[i]
            if rollout.done:
                feed_dict = {
                    self.rewards_input: rollout.rewards,
                    **self.agent.get_feed_dict(rollout.states, rollout.masks, rollout.actions)
                }

                fetches = [self._loss_actor, self._loss_critic, self._advantage,
                           self._glr, self._values, self._rewards, self._state, self._mask, self._action, self._spacial]
                fetches = fetches + [
                    self.agent._comp,
                    self.agent._spacial_log_probs,
                    self.agent._nonspacial_log_probs,
                    self.agent._probs_x,
                    self.agent._probs_y,
                    self.agent._final_log_prob,
                    self.agent.spacial_probs_x[0],
                    self.agent.spacial_probs_y[0],
                    self.agent._spacial_indexed_screen_x,
                    self.agent._modulo
                ]

                results = self.session.run([self.loss, self.train_op] + fetches, feed_dict=feed_dict)
                _loss, _, _loss_actor, _loss_critic, _advantage, _glr, _values, _rewards, _state, \
                _mask, _action, _spacial, _comp, _spacial_log_probs, _nonspacial_log_probs, _probs_x, _probs_y, \
                _final_log_prob, _all_spacial_x, _all_spacial_y, \
                    _spacial_indexed_screen_x, _modulo = results

                self.log_data(rollout.total_reward())
                self.rollouts[i] = Rollout()

    def log_data(self, reward):
        self.episode_counter += 1
        with open(self.rewards_path, 'a+') as f:
            f.write('%d\n' % reward)

        if self.episode_counter % 50 == 0:
            self.save_model(self.weights_path)
