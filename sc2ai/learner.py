import numpy as np
import torch
from sc2ai.actor_critic import ConvActorCritic
from pysc2.env.environment import StepType


class Learner:
    def __init__(self, data_queue, pipes, use_cuda=True):
        self.discount_factor = 0.99

        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

        # This is the state shape for the mini-map, represented in channels_first order.
        state_shape = [2, 84, 84]
        self.num_actions = 2
        self.resolution = 1
        screen_shape = (84 // self.resolution, 64 // self.resolution)

        self.data_queue = data_queue
        self.pipes = pipes

        self.actor = ConvActorCritic(num_actions=self.num_actions,
                                     screen_shape=screen_shape,
                                     state_shape=state_shape, dtype=self.dtype).to(self.device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.0005)

        # Define all input placeholders
        self.episode_counter = 0

        self.train_tensors = []
        for _ in self.pipes:
            self.train_tensors.append({
                'rewards': [],
                'log_action_probs': [],
                'entropys': [],
                'critic_values': []
            })
        self.train_loop()

    def train_loop(self):
        while True:
            states = []
            rewards = []
            masks = []
            step_types = []
            pids = []
            move_ids = []

            for pipe in self.pipes:
                state, reward, mask, step_type, pid, move_id = pipe.recv()

                states.append(state)
                rewards.append(reward)
                masks.append(mask)
                pids.append(pid)
                step_types.append(step_type)
                move_ids.append(move_id)

            states = torch.as_tensor(np.stack(states), device=self.device)
            masks = torch.as_tensor(np.stack(masks), device=self.device).type(self.dtype)
            rewards = np.stack(rewards)

            action_indices, x, y, entropys, log_action_probs, critic_values = self.actor(states, masks)

            for i, pid in enumerate(pids):
                self.pipes[pid].send((action_indices[i].item(), x[i].item(), y[i].item(), move_ids[i]))

                if step_types[i] != StepType.FIRST:
                    self.train_tensors[pid]['rewards'].append(rewards[i])

                if step_types[i] != StepType.LAST:
                    self.train_tensors[pid]['log_action_probs'].append(log_action_probs[i])
                    self.train_tensors[pid]['entropys'].append(entropys[i])
                    self.train_tensors[pid]['critic_values'].append(critic_values[i])

            # This only works if they all have the same episode lengths
            if step_types[0] == StepType.LAST:
                # loss_val = torch.Tensor(0, device=device).type(dtype)
                total_loss = None

                for pid in range(len(self.pipes)):
                    data = self.train_tensors[pid]
                    self.log_data(data['rewards'])
                    loss_val = self.actor_critic_loss(**data)
                    total_loss = loss_val if total_loss is None else total_loss + loss_val
                    self.train_tensors[pid] = {
                        'rewards': [],
                        'log_action_probs': [],
                        'entropys': [],
                        'critic_values': []
                    }
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

    def save(self):
        print('Saving weights')
        torch.save(self.actor.state_dict(), './weights/actor')

    def load(self):
        print('Loading weights')
        self.actor.load_state_dict(torch.load('./weights/actor'))

    def log_data(self, reward_list):
        with open('rewards.txt', 'a+') as f:
            rewards = np.sum(reward_list)
            f.write('%d\n' % rewards)

        if self.episode_counter % 200 == 0:
            self.save()
        self.episode_counter += 1

    def actor_critic_loss(self, rewards, critic_values, log_action_probs, entropys):
        discounted_rewards = self.discount(rewards)
        advantage = discounted_rewards.type(self.dtype) - torch.stack(critic_values)
        actor_loss = -torch.stack(log_action_probs).type(self.dtype) * advantage.data
        critic_loss = advantage.pow(2)

        return actor_loss.mean() + 0.5 * critic_loss.mean() - 0.01 * torch.stack(entropys).mean()

    def discount(self, rewards, discount_factor=0.99):
        prev = 0
        discounted_rewards = np.copy(rewards)
        for i in range(1, len(discounted_rewards)):
            discounted_rewards[-i] += prev * discount_factor
            prev = discounted_rewards[-i]

        return torch.as_tensor(np.array(discounted_rewards), device=self.device)

