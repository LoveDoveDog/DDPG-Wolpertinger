import numpy as np
import torch
import argparse


def train(iteration, agent, env):
    step = 0
    state, channel_occupy, _ = env.reset()
    state_tensor = torch.from_numpy(state)
    reward_list = []
    while step < iteration:
        if step < agent.buffer_size:
            action = agent.random_action(agent.message_num, channel_occupy)
        else:
            action = agent.wolpertinger_action(state_tensor, torch.from_numpy(channel_occupy).unsqueeze(0))
        next_state, next_channel_occupy, reward = env.update(action)
        agent.memory.add(state, channel_occupy, action, reward, next_state, next_channel_occupy)
        state = next_state
        channel_occupy = next_channel_occupy
        state_tensor = torch.from_numpy(state)
        if step >= agent.buffer_size:
            experiences = agent.memory.sample()
            agent.update_policy(experiences)
        if step > agent.buffer_size and (step - agent.buffer_size) % 10 == 0:
            agent.save_model()
        step += 1
        reward_list.append(reward)
        if step % 200 == 0:
            print('Step: ', step)
            print('Reward: ', np.array(reward_list[-200:]).mean())
            with open(agent.folder+'/'+agent.name+'.npy', 'wb') as f:
                np.save(f, np.array(reward_list))
        if step == 30000:
            agent.epsilon = 0.98


def valid(iteration, agent, env):
    agent.begin_eval()
    step = 0
    state, channel_occupy, _ = env.reset()
    state = torch.from_numpy(state)
    agent.load_model()
    while step < iteration:
        action = agent.wolpertinger_action(state, torch.from_numpy(channel_occupy).unsqueeze(0))
        state, channel_occupy, reward = env.update(action)
        state = torch.from_numpy(state)
        step += 1
        print('Step: ', step)
        print('Reward: ', reward)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trade', default=1, type=float)
    parser.add_argument('--message_num', default=4, type=int)
    parser.add_argument('--message_keep', default=4, type=int)
    parser.add_argument('--message_lambda', default=[20, 20, 20, 20], type=float)
    parser.add_argument('--message_slot', default=[2, 2, 2, 2], type=int)
    parser.add_argument('--channel_num', default=8, type=int)
    parser.add_argument('--energy_mean', default=[100, 500], type=float)
    parser.add_argument('--energy_variance', default=1, type=float)

    parser.add_argument('--actor_node1', default=10, type=int)
    parser.add_argument('--actor_node2', default=5, type=int)
    parser.add_argument('--critic_node1', default=10, type=int)
    parser.add_argument('--critic_node2', default=5, type=int)
    parser.add_argument('--lr_actor', default=0.01, type=float)
    parser.add_argument('--lr_critic', default=0.01, type=float)
    parser.add_argument('--tau', default=0.001, type=float)
    parser.add_argument('--gamma', default=0.9, type=float)
    parser.add_argument('--epsilon', default=0.95, type=float)
    parser.add_argument('--buffer_size', default=1000, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--random_seed', default=2, type=int)
    parser.add_argument('--folder', default='Data', type=str)
    parser.add_argument('--name', default='Exp1', type=str)
    parser.add_argument('--train_iteration', default=300000, type=int)
    parser.add_argument('--valid_iteration', default=2000, type=int)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    return args
