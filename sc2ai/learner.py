import numpy as np
import torch
from sc2ai.actor_critic import ConvActorCritic


class Learner:
    def __init__(self, environment, agent_interface,
                 load_model=False,
                 use_cuda=True,
                 gamma=0.96,
                 td_lambda=0.96):
        self.reward_discount = gamma
        self.td_lambda = td_lambda

        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

        self.agent_interface = agent_interface
        self.actor = ConvActorCritic(num_actions=self.agent_interface.num_actions,
                                     screen_dimensions=self.agent_interface.screen_dimensions,
                                     state_shape=self.agent_interface.state_shape,
                                     dtype=self.dtype, device=self.device).to(self.device)
        if load_model:
            self.load()
        else:
            open('rewards.txt', 'w').close()

        self.env = environment
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.0015)
        self.episode_counter = 0

    def generate_trajectory(self):
        states, action_masks, rewards, next_dones = self.env.reset()
        while True:
            # Choose action from state
            state = torch.as_tensor(np.stack(states), device=self.device).type(self.dtype)
            action_masks = torch.as_tensor(np.stack(action_masks), device=self.device).type(self.dtype)
            action_indices, coords, entropys, log_action_probs, critic_values = self.actor(state, action_masks)
            dones = next_dones

            # Step to get reward and next state
            states, action_masks, rewards, next_dones = \
                self.env.step(zip(action_indices.cpu().numpy(), coords.cpu().numpy()))
            yield critic_values, rewards, entropys, log_action_probs, dones
            if all(dones):
                break

    def train_episode(self):
        data = list(zip(*[step for step in self.generate_trajectory()]))
        critic_vals, rewards, entropys, log_probs, dones = data
        critic_vals, entropys, log_probs = [torch.t(torch.stack(tensors))
                                            for tensors in (critic_vals, entropys, log_probs)]
        rewards, dones = [np.stack(array).T for array in (rewards, dones)]

        loss = torch.stack([self.actor_critic_loss(rewards[i], critic_vals[i], log_probs[i], entropys[i], dones[i])
                            for i in range(self.env.num_instances)])
        loss = loss.sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for reward in rewards:
            self.log_data(np.nansum(reward[:-1]))

    def actor_critic_loss(self, rewards, values, log_action_probs, entropys, dones, infinite_horizon=False):
        seq_length = dones.shape[0] - np.sum(dones)

        rewards = torch.as_tensor(rewards[:seq_length].astype(np.float32), device=self.device).type(self.dtype)
        td_errors = rewards + self.reward_discount * values[1:seq_length + 1] - values[:seq_length]

        if not infinite_horizon:
            td_errors[seq_length - 1] = rewards[seq_length - 1] - values[seq_length - 1]
        advantage = self.discount(td_errors, discount_factor=self.td_lambda * self.reward_discount)
        if infinite_horizon:
            normalizing_factor = 1 if self.td_lambda == 1 else 1 / (1 - self.td_lambda ** np.arange(seq_length, 0, -1))
            advantage = advantage * torch.as_tensor(normalizing_factor, device=self.device).type(self.dtype)
        actor_loss = -log_action_probs.type(self.dtype)[:seq_length] * advantage.data
        critic_loss = advantage.pow(2)
        return actor_loss.mean() + 0.5 * critic_loss.mean() - 0.001 * entropys[:seq_length].mean()

    @staticmethod
    def discount(values, discount_factor):
        prev = 0
        discounted = values.clone()
        for i in range(1, discounted.shape[0] + 1):
            discounted[-i] += prev * discount_factor
            prev = discounted[-i]
        return discounted

    def save(self):
        print('Saving weights')
        torch.save(self.actor.state_dict(), './weights/actor')

    def load(self):
        self.actor.load_state_dict(torch.load('./weights/actor'))

    def log_data(self, reward):
        with open('rewards.txt', 'a+') as f:
            f.write('%d\n' % reward)

        if self.episode_counter % 50 == 0:
            self.save()
        self.episode_counter += 1
