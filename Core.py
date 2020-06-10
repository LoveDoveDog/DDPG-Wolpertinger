import random
import numpy as np
import torch
import torch.optim as opt
import torch.nn.functional as fun
from collections import deque, namedtuple
from AC import Actor, Critic
from sympy.utilities.iterables import multiset_permutations


def hard_update(target, source):
    for target_item, source_item in zip(target.parameters(), source.parameters()):
        target_item.data.copy_(source_item.data)


def soft_update(target, source, tau):
    for target_item, source_item in zip(target.parameters(), source.parameters()):
        target_item.data.copy_(tau * source_item.data + (1 - tau) * target_item.data)


def nearest(proto_action, channel_occupy, message_num):
    proto_action = proto_action * message_num
    int_action = np.rint(proto_action).astype(int)
    for index, i in enumerate(channel_occupy):
        if i == 1:
            int_action[index] = 0
    valid_index = []
    valid_action = []
    for index, action in enumerate(int_action):
        if action < 0:
            int_action[index] = 0
        if action > message_num:
            int_action[index] = message_num
        if action > 0:
            valid_index.append(index)
            valid_action.append(action)

    if len(valid_action) == 0:
        near_actions = np.zeros([1, len(int_action)])
    else:
        valid_actions = np.zeros([1, len(valid_action)])
        for i in multiset_permutations(valid_action):
            valid_actions = np.vstack((valid_actions, i))
        near_actions = np.zeros([valid_actions.shape[0], len(int_action)])
        for index, i in enumerate(valid_index):
            near_actions[:, i] = valid_actions[:, index]
    return near_actions


def noise(action, channel_occupy, message_num, epsilon):
    channel_occupy = channel_occupy.data.numpy()
    for index, i in enumerate(channel_occupy[0, :]):
        if i == 0:
            if random.uniform(0, 1) > epsilon:
                action[index] = random.randint(0, message_num)
    return action


class Core(object):
    def __init__(self, state_dim, action_dim, message_num, channel_num, args):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.message_num = message_num
        self.channel_num = channel_num
        self.actor_node1 = args.actor_node1
        self.actor_node2 = args.actor_node2
        self.critic_node1 = args.critic_node1
        self.critic_node2 = args.critic_node2
        self.lr_actor = args.lr_actor
        self.lr_critic = args.lr_critic
        self.tau = args.tau
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.folder = args.folder
        self.name = args.name

        self.device = args.device

        self.actor = Actor(self.state_dim, self.action_dim, self.actor_node1, self.actor_node2)
        self.actor_target = Actor(self.state_dim, self.action_dim, self.actor_node1, self.actor_node2)
        self.actor_opt = opt.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic = Critic(self.state_dim, self.action_dim, self.critic_node1, self.critic_node2)
        self.critic_target = Critic(self.state_dim, self.action_dim, self.critic_node1, self.critic_node2)
        self.critic_opt = opt.Adam(self.critic.parameters(), lr=self.lr_critic)

        self.begin_cuda()

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

    def update_policy(self, experiences):
        states, channel_occupys, actions, rewards, next_states, next_channel_occupys = experiences
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        next_target_actions = self.wolpertinger_action(next_states, next_channel_occupys, status='target')
        next_target_actions = torch.from_numpy(next_target_actions).to(self.device)
        next_target_q_values = self.critic_target(next_states, next_target_actions)
        evaluated_q_values = rewards + self.gamma * next_target_q_values
        target_q_values = self.critic(states, actions)
        self.critic.zero_grad()
        critic_loss = fun.mse_loss(evaluated_q_values, target_q_values)
        critic_loss.backward()
        self.critic_opt.step()

        self.actor.zero_grad()
        actor_loss = -self.critic(states, self.actor(states))
        actor_loss = actor_loss.mean()
        actor_loss.backward()
        self.actor_opt.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    def random_action(self, message_num, channel_occupy):
        action = np.zeros(len(channel_occupy), dtype=int)
        for index, i in enumerate(channel_occupy):
            if i == 0:
                action[index] = random.randint(0, message_num)
        return action

    def wolpertinger_action(self, states, channel_occupys, status='execute'):
        if status == 'execute':
            states_tensor = states.to(self.device)
            proto_actions_tensor = self.actor(states_tensor)
            proto_actions = proto_actions_tensor.cpu()
            next_actions = np.empty(proto_actions.size())
            for state, channel_occupy, proto_action, index in zip(states, channel_occupys, proto_actions,
                                                                  np.arange(next_actions.shape[0])):
                state = state.data.numpy()
                channel_occupy = channel_occupy.data.numpy()
                proto_action = proto_action.data.numpy()
                near_actions = nearest(proto_action, channel_occupy, self.message_num)
                state_list = np.vstack([state for i in np.arange(near_actions.shape[0])])
                state_list_tensor = torch.from_numpy(state_list).to(self.device)
                near_actions_tensor = torch.from_numpy(near_actions).to(self.device)
                near_values = self.critic(state_list_tensor, near_actions_tensor)
                action_index = torch.argmax(near_values)
                next_actions[index, :] = near_actions[action_index, :]
            next_actions = next_actions[0, :]
            next_actions = noise(next_actions, channel_occupys, self.message_num, self.epsilon)

        elif status == 'target':
            proto_actions_tensor = self.actor_target(states)
            states = states.cpu()
            proto_actions = proto_actions_tensor.cpu()
            next_actions = np.empty(proto_actions.size())
            for state, channel_occupy, proto_action, index in zip(states, channel_occupys, proto_actions,
                                                                  np.arange(next_actions.shape[0])):
                state = state.data.numpy()
                channel_occupy = channel_occupy.data.numpy()
                proto_action = proto_action.data.numpy()
                near_actions = nearest(proto_action, channel_occupy, self.message_num)
                state_list = np.vstack([state for i in np.arange(near_actions.shape[0])])
                state_list_tensor = torch.from_numpy(state_list).to(self.device)
                near_actions_tensor = torch.from_numpy(near_actions).to(self.device)
                near_values = self.critic_target(state_list_tensor, near_actions_tensor)
                action_index = torch.argmax(near_values)
                next_actions[index, :] = near_actions[action_index, :]

        return next_actions

    def begin_eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def begin_cuda(self):
        self.actor.to(self.device)
        self.actor_target.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)

    def to_device(self, tensor):
        if self.device is not None:
            tensor = tensor.to(self.device)
        return tensor

    def save_model(self):
        torch.save(self.actor.state_dict(), self.folder+'/'+self.name+'actor.pt')
        torch.save(self.critic.state_dict(), self.folder+'/'+self.name+'critic.pt')

    def load_model(self):
        self.actor.load_state_dict(torch.load(self.folder+'/'+self.name+'actor.pt'))
        self.critic.load_state_dict(torch.load(self.folder+'/'+self.name+'critic.pt'))


class ReplayBuffer(object):
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)
        self.experience \
            = namedtuple("Experience",
                         field_names=
                         ["state", "channel_occupy", "action", "reward", "next_state", "next_channel_occupy"])

    def add(self, state, channel_occupy, action, reward, next_state, next_channel_occupy):
        action = action.astype(float)
        new_experience = self.experience(state, channel_occupy, action, reward, next_state, next_channel_occupy)
        self.memory.append(new_experience)

    def sample(self):
        experiences = random.sample(self.memory, self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences]))
        channel_occupys = torch.from_numpy(np.vstack([e.channel_occupy for e in experiences]))
        actions = torch.from_numpy(np.vstack([e.action for e in experiences]))
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences]))
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences]))
        next_channel_occupys = torch.from_numpy(np.vstack([e.next_channel_occupy for e in experiences]))

        return [states, channel_occupys, actions, rewards, next_states, next_channel_occupys]
