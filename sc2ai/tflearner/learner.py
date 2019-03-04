import tensorflow as tf
import numpy as np
import trfl
import sys


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

        with self.agent.graph.as_default():
            self.rewards_input = tf.placeholder(tf.float32, [None])
            self.loss = self.ac_loss()
            self.rollouts = [Rollout() for _ in range(self.num_games)]
            self.train_op = tf.train.AdamOptimizer(0.001).minimize(self.loss)
            self.session = self.agent.session
            self.session.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()

    def ac_loss(self):
        num_steps = tf.shape(self.rewards_input)[0]
        discounts = tf.ones((num_steps, 1)) * self.discount_factor
        bootstrap_values = tf.zeros((1,))  # TODO: replace with value for non finished games
        rewards = tf.expand_dims(self.rewards_input, axis=1)
        values = tf.expand_dims(self.agent.train_values(), axis=1)

        glr = trfl.generalized_lambda_returns(rewards, discounts, values, bootstrap_values)
        advantage = tf.squeeze(glr)

        loss_actor = tf.reduce_mean(-advantage * self.agent.train_log_probs())
        loss_critic = tf.reduce_mean(advantage ** 2)
        # print_op = tf.print("loss tensors:",
        #                     "\nloss actor", loss_actor,
        #                     # "\nloss critic", loss_critic,
        #                     "\nadvantage", advantage,
        #                     "\nglr", glr,
        #                     '\nbootstrap values', bootstrap_values,
        #                     '\ndiscounts', discounts,
        #                     "\nvalues", values,
        #                     "\nrewards", rewards,
        #                     output_stream=sys.stdout)
        # with tf.control_dependencies([print_op]):
        result = loss_actor + 0.5 * loss_critic
        return result

    def generate_trajectory(self):
        states, masks, _, _ = self.env.reset()
        while True:
            action_indices = self.agent.step(states, masks)
            new_states, masks, rewards, dones = self.env.step(action_indices)

            for i, rollout in enumerate(self.rollouts):
                rollout.add_step(states[i], action_indices[i], rewards[i], masks[i], dones[i])
            states = new_states
            if all(dones):
                # Add in the done state for rollouts which just finished for calculating the bootstrap value.
                for i, rollout in enumerate(self.rollouts):
                    rollout.add_step(states[i])
                return

    def save_model(self):
        save_path = self.saver.save(self.session, 'saves/model.ckpt')
        print("Model Saved in %s" % save_path)

    def load_model(self):
        self.saver.restore(self.session, 'saves/model.ckpt')
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
                loss, _ = self.session.run([self.loss, self.train_op], feed_dict=feed_dict)
                print(rollout.total_reward())
                self.rollouts[i] = Rollout()
