import numpy as np
import torch
from sc2ai.actor_critic import ConvActorCritic


class Learner:
    def __init__(self, environment, agent_interface, use_cuda=True):
        self.discount_factor = 0.99

        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

        self.agent_interface = agent_interface
        self.actor = ConvActorCritic(num_actions=self.agent_interface.num_actions,
                                     screen_shape=self.agent_interface.screen_shape,
                                     state_shape=self.agent_interface.state_shape, dtype=self.dtype).to(self.device)
        self.env = environment
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.0005)
        self.episode_counter = 0

    def generate_trajectory(self):
        states, action_masks, rewards, dones = self.env.reset()
        while not all(dones):
            state = torch.as_tensor(np.stack(states), device=self.device)
            action_masks = torch.as_tensor(np.stack(action_masks), device=self.device).type(self.dtype)

            action_indices, x, y, entropys, log_action_probs, critic_values = self.actor(state, action_masks)
            states, action_masks, rewards, dones = self.env.step(zip(action_indices, x, y))
            yield critic_values, rewards, entropys, log_action_probs, dones  # TODO: Check OBOE done

    def train_episode(self):
        data = list(zip(*[step for step in self.generate_trajectory()]))
        critic_vals, rewards, entropys, log_probs, dones = data
        critic_vals, entropys, log_probs = [torch.t(torch.stack(tensors))
                                            for tensors in (critic_vals, entropys, log_probs)]
        rewards, dones = [np.stack(array).T for array in (rewards, dones)]

        loss = torch.stack([self.actor_critic_loss(rewards[i], critic_vals[i], log_probs[i], entropys[i])
                            for i in range(self.env.num_instances)]).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for reward in rewards:
            self.log_data(np.sum(reward))

    def save(self):
        print('Saving weights')
        torch.save(self.actor.state_dict(), './weights/actor')

    def load(self):
        print('Loading weights')
        self.actor.load_state_dict(torch.load('./weights/actor'))

    def log_data(self, reward):
        with open('rewards.txt', 'a+') as f:
            f.write('%d\n' % reward)

        if self.episode_counter % 200 == 0:
            self.save()
        self.episode_counter += 1

    def actor_critic_loss(self, rewards, critic_values, log_action_probs, entropys):
        discounted_rewards = self.discount(rewards)
        advantage = discounted_rewards.type(self.dtype) - critic_values
        actor_loss = -log_action_probs.type(self.dtype) * advantage.data
        critic_loss = advantage.pow(2)

        return actor_loss.mean() + 0.5 * critic_loss.mean() - 0.01 * entropys.mean()

    def discount(self, rewards, discount_factor=0.99):
        prev = 0
        discounted_rewards = np.copy(rewards)
        for i in range(1, len(discounted_rewards)):
            discounted_rewards[-i] += prev * discount_factor
            prev = discounted_rewards[-i]
        return torch.as_tensor(np.array(discounted_rewards), device=self.device)
