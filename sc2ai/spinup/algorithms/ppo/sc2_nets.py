import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from sc2ai.spinup.algorithms.ppo.core import mlp
from sc2ai.envs.actions import ActionVectorType
import math


class SC2Actor(nn.Module):
    def __init__(self, previous_modules, hidden_units, action_spec, action_mask, device):
        super().__init__()
        self._previous_modules = previous_modules
        self._hidden_units = hidden_units
        self._action_spec = action_spec
        self._action_mask = action_mask
        self.logit_nets = list()
        self.device = device
        self.to(device)

        print("action register-----------------------------")
        for action_tuple in self._action_spec:
            print(action_tuple)
            if action_tuple[0] is ActionVectorType.ACTION_TYPE:
                self.logit_nets += [
                    torch.nn.Sequential(previous_modules, torch.nn.Linear(hidden_units, action_tuple[1])).to(device)]
            elif action_tuple[0] is ActionVectorType.SCALAR:
                self.logit_nets += [
                    torch.nn.Sequential(previous_modules, torch.nn.Linear(hidden_units, action_tuple[1])).to(device)]
            elif action_tuple[0] is ActionVectorType.SPATIAL:
                xy_size = int(math.sqrt(action_tuple[1]))
                self.logit_nets += \
                    [(torch.nn.Sequential(previous_modules, torch.nn.Linear(hidden_units, xy_size)).to(device),
                      torch.nn.Sequential(previous_modules, torch.nn.Linear(hidden_units, xy_size)).to(device))]
            else:
                raise Exception("Such ActionVectorType is not defined.")
        print("action register-----------------------------")

    def distributions(self, obs):
        distributions = []
        _id = 0
        for action_tuple in self._action_spec:
            if action_tuple[0] is ActionVectorType.ACTION_TYPE or action_tuple[0] is ActionVectorType.SCALAR:
                distributions += [Categorical(logits=self.logit_nets[_id](obs))]
            elif action_tuple[0] is ActionVectorType.SPATIAL:
                distributions += [(Categorical(logits=self.logit_nets[_id][0](obs)),
                                   Categorical(logits=self.logit_nets[_id][1](obs)))]
            else:
                raise Exception("Such ActionVectorType is not defined.")
            _id += 1
        return distributions

    def log_prob_from_distributions(self, pis, acts):
        log_probs = list()
        masks = list()
        for i, act in enumerate(acts):
            masks.append(self._action_mask[int(act[0].item())])
        masks = torch.tensor(masks, device=self.device, dtype=torch.int32)

        for i, distribution in enumerate(pis):
            action_tuple = self._action_spec[i]
            if isinstance(distribution, tuple):
                xy_size = int(math.sqrt(action_tuple[1]))
                #print(act[_id], xy_size)
                log_probs.append((distribution[0].log_prob(acts[:, i] // xy_size) +
                                 distribution[1].log_prob(acts[:, i] % xy_size)))
            else:
                log_probs.append(distribution.log_prob(acts[:, i]))
        return torch.sum(torch.mul(torch.stack(log_probs, 1), masks), 1)

    def sample(self, pis):
        a_vector = list()
        for i, distribution in enumerate(pis):
            action_tuple = self._action_spec[i]
            if isinstance(distribution, tuple):
                xy_size = int(math.sqrt(action_tuple[1]))
                a_vector.append(distribution[0].sample() * xy_size + distribution[1].sample())
            else:
                a_vector.append(distribution.sample())
        act_id = a_vector[0].item()
        #print(self._action_mask)
        #print(torch.stack(a_vector, 1),
        #      torch.tensor(self._action_mask[act_id], device=self.device, dtype=torch.int32).unsqueeze(0))
        return torch.mul(torch.stack(a_vector, 1),
                         torch.tensor(self._action_mask[act_id], device=self.device, dtype=torch.int32).unsqueeze(0))

    def forward(self, obs, act=None):
        pis = self.distributions(obs)
        logp_a = None
        if act is not None:
            logp_a = self.log_prob_from_distributions(pis, act)
        return pis, logp_a


class SC2Critic(nn.Module):
    def __init__(self, previous_modules, hidden_units, device):
        super().__init__()
        self.v_net = torch.nn.Sequential(previous_modules, torch.nn.Linear(hidden_units, 1)).to(device)
        self.device = device

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)  # Critical to ensure v has the right shape.


class SC2AtariNetActorCritic(nn.Module):
    def __init__(self, observation_space, action_spec=None, action_mask=None, hidden_units=256, activation=nn.ReLU,
                 device=torch.device('cpu'),
                 usegnn=False):
        super().__init__()
        self.device = device
        if usegnn:
            print("Use gnn base for actor!")
            self._convs_sequence = self._build_gconv_layers(observation_space, hidden_units, device)
        else:
            self._convs_sequence = self._build_sequential_layers(observation_space, hidden_units, activation, device)
        self._build_policy(self._convs_sequence, hidden_units, action_spec, action_mask)
        self._build_critic(self._convs_sequence, hidden_units)
        self.to(device=device)

    def _build_policy(self, convs_sequence, hidden_units, action_spec, action_mask):
        self.pi = SC2Actor(convs_sequence, hidden_units, action_spec, action_mask, self.device)
        self.pi.to(device=self.device)

    def _build_critic(self, convs_sequence, hidden_units):
        self.v = SC2Critic(convs_sequence, hidden_units, self.device)
        self.v.to(device=self.device)

    def _build_sequential_layers(self, observation_space, hidden_units, activation, device):
        self.conv1 = nn.Conv2d(observation_space['feature_screen'].shape[0], 16, 8, stride=4).to(device)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2).to(device)  # 32 channel 9,9
        return nn.Sequential(self.conv1,
                             activation(),
                             self.conv2,
                             activation(),
                             nn.Flatten(),
                             nn.Linear(32 * 9 * 9, hidden_units),
                             nn.ReLU()).to(device)

    def _build_gconv_layers(self, observation_space, hidden_units, device):
        return GNNBase(observation_space['feature_screen'].shape[0], hidden_units).to(device)


    def step(self, obs):
        with torch.no_grad():
            pis = self.pi.distributions(obs)
            a = self.pi.sample(pis)
            logp_a = self.pi.log_prob_from_distributions(pis, a)
            v = self.v(obs)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def act(self, obs):
        return self.step(obs)[0]


class SC2FullyConvActor(nn.Module):
    def __init__(self, previous_modules, hidden_units, action_spec, device):
        super().__init__()
        self._previous_modules = previous_modules
        self._hidden_units = hidden_units
        self._action_spec = action_spec
        self.logit_nets = list()
        self.device = device
        self.to(device)

        print("action register-----------------------------")
        for action_tuple in self._action_spec:
            print(action_tuple)
            if action_tuple[0] is ActionVectorType.ACTION_TYPE:
                self.logit_nets += [
                    torch.nn.Sequential(previous_modules, torch.nn.Linear(hidden_units, action_tuple[1])).to(device)]
            elif action_tuple[0] is ActionVectorType.SCALAR:
                self.logit_nets += [
                    torch.nn.Sequential(previous_modules, torch.nn.Linear(hidden_units, action_tuple[1])).to(device)]
            elif action_tuple[0] is ActionVectorType.SPATIAL:
                xy_size = int(math.sqrt(action_tuple[1]))
                self.logit_nets += \
                    [(torch.nn.Sequential(previous_modules, torch.nn.Linear(hidden_units, xy_size)).to(device),
                      torch.nn.Sequential(previous_modules, torch.nn.Linear(hidden_units, xy_size)).to(device))]
            else:
                raise Exception("Such ActionVectorType is not defined.")
        print("action register-----------------------------")


class SC2FullyConvActorCritic(SC2AtariNetActorCritic):
    def _build_sequential_layers(self, hidden_units, activation, device):
        pass

# class SC2FullyConvLSTMActorCritic(nn.Module):





def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class GNNBase(nn.Module):
    """
    This is for extracting the global attribute from observation/context inputs
    """
    def __init__(self, input_channel, num_outputs):
        super().__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('leaky_relu', 0.1))
        # here the input shape is in numpy img format! channel last!!
        C = input_channel

        self.feature_extractor = nn.Sequential(
            init_(nn.Conv2d(C, 64, 8, stride=4)), nn.LeakyReLU(.1),
            init_(nn.Conv2d(64, 510, 4, stride=2)), nn.LeakyReLU(.1),
        )
        self.attention1 = RelationalAttention(510 + 2, 256, 256, 4, maxout=True)
        self.attention2 = RelationalAttention(256, 128, 128, 4, maxout=True)
        self.mapping = nn.Sequential(
            init_(nn.Linear(512, num_outputs)),
            nn.LeakyReLU(.1)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.attention1(x)
        return self.mapping(x)


class RelationalAttention(nn.Module):
    '''
    Transpose the input data batch BxCxHxW into object embeddings of shape Bx(HxW)xC and process with multihead dot product attention. 
    The output shape will be BxHxWx(C+2) since the module will attach two extra dimensions to represent the location of each object in the original HxW frame.
    Note that d_model = (C+2)

    The output shape is
    BxHxWx(C+2) if maxout=False
    Bx(C+2) if maxout=True
    '''

    def __init__(self, d_model, d_kq, d_v, n_heads, drop_prob=0.0, maxout=False):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_kq
        self.d_q = d_kq
        self.d_v = d_v
        self.n_heads = n_heads
        self.maxout = maxout

        self.gnn = MultiHeadAttention(d_model, d_kq, d_v, n_heads, drop_prob)
        # self.gnn = MultiHeadAttention(512, 256, 256, 4)

    def forward(self, x):
        # add object position as additional GNN feature
        b_sz, n_channel, height, width = x.size()
        entity_embed = x.reshape(b_sz, n_channel, -1).transpose(1, 2)
        coord = []
        for i in range(height*width):
            # add coordinate and normalize
            coord.append([float(i//width)/height, (i%width)/width])
        coord = torch.tensor(coord, device=entity_embed.device).view(1, -1, 2).repeat(b_sz, 1, 1)
        entity_embed = torch.cat((entity_embed, coord), dim=2)
        
        out = F.relu(self.gnn(entity_embed))
        if self.maxout:
            # only output the max value no index
            out = torch.max(out, dim=1)[0]

        return out


class MultiHeadAttention(nn.Module):
    '''Multi-Head Self Attention module used by GNN'''
    
    def __init__(self, d_model, d_kq, d_v, n_heads, drop_prob=0.0):
        super(MultiHeadAttention, self).__init__()
        init_ = lambda m: self.param_init(m, nn.init.orthogonal_, nn.init.calculate_gain('linear'))
        self.d_model = d_model
        self.d_k = d_kq
        self.d_q = d_kq
        self.d_v = d_v
        self.n_heads = n_heads

        self.linear_k = init_(nn.Linear(self.d_model, self.d_k*n_heads, bias=False))
        self.linear_q = init_(nn.Linear(self.d_model, self.d_q*n_heads, bias=False))
        self.linear_v = init_(nn.Linear(self.d_model, self.d_v*n_heads, bias=False))
        self.normalize = np.sqrt(self.d_k)
        self.linear_output = nn.Sequential(
                                  nn.Linear(self.d_v*n_heads, self.d_model, bias=False),
                             )
        
        # Assume that the dimension of linear_k/q/v are all the same
        self.layer_norm_embed = nn.LayerNorm(self.d_k*n_heads, eps=1e-6)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.atten_dropout = nn.Dropout(drop_prob)
        
    def param_init(self, module, weight_init, gain=1):
        weight_init(module.weight.data, gain=gain)
        return module
    
    def forward(self, entity_embeds_raw):
        b_sz, num_entities = entity_embeds_raw.size(0), entity_embeds_raw.size(1)
        # (batch_size, num_entities, d_model) -> (batch_size*num_entities, d_model)
        entity_embeds = entity_embeds_raw.reshape(-1, self.d_model)
        # (batch_size*num_entities, d_k*n_heads) -> (batch_size, num_entities, n_heads, d_k)
        embed_q = self.layer_norm_embed(self.linear_q(entity_embeds)).view(b_sz, num_entities, self.n_heads, self.d_q)
        embed_k = self.layer_norm_embed(self.linear_k(entity_embeds)).view(b_sz, num_entities, self.n_heads, self.d_k)
        embed_v = self.layer_norm_embed(self.linear_v(entity_embeds)).view(b_sz, num_entities, self.n_heads, self.d_v)

        residual_v = embed_v
        # swap n_head dim with num_entities
        # ->(batch_size, n_heads, num_entities, d_k)
        embed_q2 = embed_q.transpose(1,2)
        embed_k2 = embed_k.transpose(1,2)
        embed_v2 = embed_v.transpose(1,2)
        
        # Scaled Dot Product Attention(for each head)
        tmp = torch.matmul(embed_q2, embed_k2.transpose(2, 3))/self.normalize
        # -> (batch_size, n_heads, num_entities, num_entities_prob)
        weights = self.atten_dropout(F.softmax(tmp, dim=-1))
        # weights = self.atten_dropout(F.softmax(tmp, dim=1)) #this is the previous old/wrong implementation
        new_v = torch.matmul(weights, embed_v2)
        
        # Concatenate over head dimensioins
        # (batch_size, n_heads, num_entities, d_k) -> (batch_size, num_entities, n_heads*d_k)
        new_v = new_v.transpose(1, 2).contiguous().view(b_sz, num_entities, -1)
        new_v = self.linear_output(new_v)
        
        # residual
        output = new_v + entity_embeds_raw
        output = self.layer_norm(output).view(b_sz, num_entities, self.d_model)
        
        return output