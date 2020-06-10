import random
import numpy as np
import torch

from Env import Env
from Core import Core
from InterAct_GPU import train, valid, parser

if __name__ == '__main__':
    args = parser()
    args.energy_mean = [0, 0]
    args.name = 'id1'
    if args.random_seed > 0:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)

    env = Env(args)

    if torch.cuda.is_available():
        args.device = torch.device('cuda:0')

    agent = Core(env.state_dim, env.action_dim, env.message_num, env.channel_num, args)
    train(args.train_iteration, agent, env)
    # valid(args.valid_iteration, agent, env)
