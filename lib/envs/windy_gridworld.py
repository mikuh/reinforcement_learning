import gym
import numpy as np
import sys
from gym.envs.toy_text import discrete
from io import StringIO

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class WindyGridworldEnv(discrete.DiscreteEnv):

    metadata = {'render.modes': ['human', 'ansi']}

    def _limit_coordinates(self, coord):
        """限制坐标，不能跑到格子外面去
        """
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta, winds):
        """计算转移概率
        Args:
            current: 当前坐标
            delta:  坐标移动
            winds: 风强度映射表
        Returns:
            一个list的tuple (转移概率，新状态，即时奖励， 是否完结)
        """
        new_position = np.array(current) + np.array(delta) + np.array([-1, 0]) * winds[tuple(current)]
        new_position = self._limit_coordinates(new_position).astype(int)    # 新的坐标
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)   # 根据坐标得到新的状态编号
        is_done = tuple(new_position) == (3, 7)                             # 达到这个点代表结束
        return [(1.0, new_state, -1.0, is_done)]

    def __init__(self):
        self.shape = (7, 10)            # 格子形状

        nS = np.prod(self.shape)        # 状态数量
        nA = 4                          # 行动数量

        # 风的强度
        winds = np.zeros(self.shape)
        winds[:, [3, 4, 5, 8]] = 1
        winds[:, [6, 7]] = 2

        # 计算转移概率
        P = {}
        for s in range(nS):
            position = np.unravel_index(s, self.shape)  # 每个状态对应的矩阵坐标
            P[s] = {a: [] for a in range(nA)}
            P[s][UP] = self._calculate_transition_prob(position, [-1, 0], winds)
            P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1], winds)
            P[s][DOWN] = self._calculate_transition_prob(position, [1, 0], winds)
            P[s][LEFT] = self._calculate_transition_prob(position, [0, -1], winds)

        # 我们总是从状态(3, 0) 点开始走
        isd = np.zeros(nS)      # 初始状态分布
        isd[np.ravel_multi_index((3, 0), self.shape)] = 1.0

        super(WindyGridworldEnv, self).__init__(nS, nA, P, isd)

    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            # print(self.s)
            if self.s == s:
                output = " x "
            elif position == (3, 7):
                output = " T "
            else:
                output = " o "

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += "\n"

            outfile.write(output)
        outfile.write("\n")