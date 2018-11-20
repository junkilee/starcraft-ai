import torch
import torch.nn.functional as F
from torch.distributions import Categorical


class ConvActorCritic(torch.nn.Module):
    def __init__(self, num_actions, screen_shape, state_shape, dtype=torch.FloatTensor):
        super().__init__()
        self.num_actions = num_actions
        self.state_shape = state_shape
        self.dtype = dtype
        num_input_channels = state_shape[0]

        self.conv1 = torch.nn.Conv2d(num_input_channels, 32, 5, stride=3)
        self.conv2 = torch.nn.Conv2d(32, 32, 5, stride=3)
        self.conv3 = torch.nn.Conv2d(32, 16, 3)
        self.linear1 = torch.nn.Linear(576, 32)
        self.head_non_spacial = torch.nn.Linear(32, self.num_actions)
        self.head_spacial_x = torch.nn.Linear(32, screen_shape[0])
        self.head_spacial_y = torch.nn.Linear(32, screen_shape[1])
        self.head_critic = torch.nn.Linear(32, 1)

    @staticmethod
    def sample(probs):
        categorical = Categorical(probs)
        chosen_action_index = categorical.sample()
        log_action_prob = categorical.log_prob(chosen_action_index)
        return chosen_action_index, log_action_prob, categorical.entropy()

    def forward(self, state, action_mask):
        num_steps = state.shape[0]

        logits = F.relu(self.conv1(state))
        logits = F.relu(self.conv2(logits))
        logits = F.relu(self.conv3(logits))
        logits = logits.view(num_steps, -1)
        logits = F.relu(self.linear1(logits))

        non_spacial_probs = F.softmax(self.head_non_spacial(logits), dim=-1)
        spacial_x_probs = F.softmax(self.head_spacial_x(logits), dim=-1)
        spacial_y_probs = F.softmax(self.head_spacial_y(logits), dim=-1)
        critic_value = self.head_critic(logits)

        masked_probs = (non_spacial_probs + 0.00001) * action_mask
        masked_probs = masked_probs / torch.sum(masked_probs, dim=-1, keepdim=True)

        non_spacial_index, non_spacial_log_prob, non_spacial_entropy = self.sample(masked_probs)
        x, x_log_prob, x_entropy = self.sample(spacial_x_probs)
        y, y_log_prob, y_entropy = self.sample(spacial_y_probs)

        joint_entropy = non_spacial_entropy + x_entropy + y_entropy
        joint_log_prob = (x_log_prob + y_log_prob) * (non_spacial_index == 0).type(self.dtype) \
                            + non_spacial_log_prob

        return non_spacial_index, x, y, joint_entropy, joint_log_prob, torch.squeeze(critic_value, dim=-1)
