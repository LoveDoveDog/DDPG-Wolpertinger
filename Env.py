import numpy as np


class Env(object):

    def __init__(self, args):
        self.trade = args.trade

        self.message_num = args.message_num
        self.message_keep = args.message_keep
        self.message_lambda = args.message_lambda
        self.message_slot = args.message_slot
        self.channel_num = args.channel_num
        self.energy_mean = np.random.uniform(args.energy_mean[0],
                                             args.energy_mean[1], [self.message_num, self.channel_num])
        self.energy_variance = args.energy_variance

        self.channel_state = np.zeros([self.message_num, self.channel_num], dtype=int)
        self.table = np.zeros([self.message_num, self.message_keep], dtype=int)
        self.arrival = np.zeros(self.message_num, dtype=int)
        self.energy = np.zeros([self.message_num, self.channel_num], dtype=float)

        # self.state_dim = self.message_num*self.message_keep + self.message_num*self.channel_num
        self.state_dim = self.message_num*self.message_keep + 2*self.message_num*self.channel_num
        self.action_dim = self.channel_num

    def reset(self):
        self.channel_state = np.zeros([self.message_num, self.channel_num], dtype=int)
        self.table = np.zeros([self.message_num, self.message_keep], dtype=int)
        self.energy = np.random.normal(self.energy_mean, self.energy_variance)
        return self.update(np.zeros(self.channel_num))

    def update(self, action):
        action = np.rint(action).astype(int)
        energy_cost = 0
        for index, i in enumerate(action):
            if i > 0:
                energy_cost += self.energy[i - 1, index] * self.message_slot[i - 1]
                self.table[i-1, :] = 0
        delay_cost = 0
        for i in np.arange(self.message_num):
            delay_cost += np.sum(self.table[i, :] * (1 + np.arange(self.message_keep)))
        reward = - self.trade * energy_cost - delay_cost

        for index in np.arange(self.message_num):
            self.arrival[index] = np.random.poisson(self.message_lambda[index])
        for index in np.arange(self.message_keep):
            if index == 0:
                self.table[:, -index - 1] = self.table[:, -index - 1] + self.table[:, -index - 2]
            elif index < self.message_keep - 1:
                self.table[:, -index - 1] = self.table[:, -index - 2]
            else:
                self.table[:, -index - 1] = self.arrival

        self.channel_state[self.channel_state > 0] += 1
        for index, i in enumerate(action):
            if i > 0:
                self.channel_state[i - 1, index] = 1
        for i in np.arange(self.message_num):
            line = self.channel_state[i, :]
            line[line == self.message_slot[i]] = 0
            self.channel_state[i, :] = line

        self.energy = np.random.normal(self.energy_mean, self.energy_variance)

        channel_occupy = np.zeros(self.channel_num, dtype=int)
        channel_occupy[np.sum(self.channel_state, axis=0) > 0] = 1

        normal_table = np.zeros([self.message_num, self.message_keep])
        for i in np.arange(self.message_num):
            normal_table[i, :] = (self.table[i, :]-self.message_lambda[i])/np.sqrt(self.message_lambda[i])
        normal_energy = (self.energy-self.energy_mean)/np.sqrt(self.energy_variance)

        tem_a = np.hstack(normal_table)
        tem_c = np.hstack(self.channel_state)
        tem_d = np.hstack(normal_energy)
        tem_e = np.hstack([tem_a, tem_c, tem_d])
        next_state = np.zeros([1, self.state_dim])
        next_state[0, :] = tem_e

        return [next_state, channel_occupy, reward]
