import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical


class ConvActorCritic(torch.nn.Module):
    def __init__(self, num_actions, screen_dimensions, state_shape, dtype=torch.FloatTensor, device=torch.device('cpu')):
        super().__init__()
        self.num_actions = num_actions
        self.state_shape = state_shape
        self.screen_dimensions = screen_dimensions
        self.dtype = dtype
        num_input_channels = state_shape[0]

        self.conv1 = torch.nn.Conv2d(num_input_channels, 32, 5, stride=3)
        self.conv2 = torch.nn.Conv2d(32, 32, 5, stride=3)
        self.conv3 = torch.nn.Conv2d(32, 16, 3)
        self.linear1 = torch.nn.Linear(576, 32)
        self.head_non_spacial = torch.nn.Linear(32, self.num_actions)
        self.head_spacials = torch.nn.ModuleList()
        for dimension in screen_dimensions:
            self.head_spacials.append(torch.nn.Linear(32, dimension).to(device))
        self.head_critic = torch.nn.Linear(32, 1)

    @staticmethod
    def sample(probs):
        categorical = Categorical(probs)
        chosen_action_index = categorical.sample()
        log_action_prob = categorical.log_prob(chosen_action_index)
        return chosen_action_index, log_action_prob, categorical.entropy()

    def forward(self, state, action_mask, epsilon=0):
        num_steps = state.shape[0]

        logits = F.relu(self.conv1(state))
        logits = F.relu(self.conv2(logits))
        logits = F.relu(self.conv3(logits))
        logits = logits.view(num_steps, -1)
        logits = F.relu(self.linear1(logits))

        non_spacial_probs = F.softmax(self.head_non_spacial(logits), dim=-1)
        spacial_probs = []
        for head_spacial in self.head_spacials:
            spacial_probs.append(F.softmax(head_spacial(logits), dim=-1))
        critic_value = self.head_critic(logits)

        # Calculate amount to add to each action such that there is epsilon probability of performing a random action
        epsilon = np.min([1, epsilon + 0.00001])
        extra_prob = ((epsilon / (1 - epsilon)) * torch.sum(non_spacial_probs.data * action_mask, dim=-1, keepdim=True)).type(self.dtype)
        extra_prob /= torch.sum(action_mask, dim=-1, keepdim=True)
        masked_probs = (non_spacial_probs + extra_prob) * action_mask

        # Normalize probability distribution
        masked_probs = masked_probs / torch.sum(masked_probs, dim=-1, keepdim=True)

        non_spacial_index, non_spacial_log_prob, non_spacial_entropy = self.sample(masked_probs)
        data = zip(*[self.sample(probs) for probs in spacial_probs])
        spacial_coords, spacial_log_probs, spacial_entropys = [torch.stack(tensors, dim=-1) for tensors in data]
        joint_entropy = non_spacial_entropy + spacial_entropys.sum(dim=-1)

        # Calculate log probability for actions. Each pair of spacial actions corresponds to a non-spacial action.
        joint_log_prob = non_spacial_log_prob
        for i in range(int(len(self.screen_dimensions) / 2)):
            joint_log_prob += spacial_log_probs[:, i*2:i*2+2].sum(dim=-1) * (non_spacial_index == i).type(self.dtype)
        return non_spacial_index, spacial_coords, joint_entropy, joint_log_prob, torch.squeeze(critic_value, dim=-1)
