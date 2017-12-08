import numpy as np
import sys
from gym.envs.toy_text import discrete
from io import StringIO

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3



class GridworldEnv(discrete.DiscreteEnv):
    """格子世界环境
    初始化的时候将计算好所有格子状态采取每个动作的（状态转移概率， 下一个状态， 立即回报， 是否终结）的信息
    """

    def __init__(self, shape=[4, 4]):
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')

        self.shape = shape      # 环境形状

        nS = np.prod(shape)     # 状态个数
        nA = 4                  # 行动个数

        MAX_Y = shape[0]
        MAX_X = shape[1]

        P = {}
        grid = np.arange(nS).reshape(shape)                 # 构建格子
        it = np.nditer(grid, flags=['multi_index'])         # 外部循环迭代器

        while not it.finished:
            s = it.iterindex                                # 格子状态
            y, x = it.multi_index                           # 格子坐标

            P[s] = {a: [] for a in range(nA)}               # 每个状态所有动作的响应放在一个[]中 {0: [], 1: [], 2: [], 3: []}
            is_done = lambda s: s == 0 or s == (nS - 1)     # 是否终结
            reward = 0.0 if is_done(s) else -1.0            # 立即回报

            # 每个状态的 状态转移概率、下一个状态、回报、 是否终结
            if is_done(s):
                P[s][UP] = [(1.0, s, reward, True)]
                P[s][RIGHT] = [(1.0, s, reward, True)]
                P[s][DOWN] = [(1.0, s, reward, True)]
                P[s][LEFT] = [(1.0, s, reward, True)]
            # 非终止状态
            else:
                # 下一个状态 想要走出格子的话就停留在原处
                ns_up = s if y == 0 else s - MAX_X
                ns_right = s if x == (MAX_X - 1) else s + 1
                ns_down = s if y == (MAX_Y - 1) else s + MAX_X
                ns_left = s if x == 0 else s - 1
                P[s][UP] = [(1.0, ns_up, reward, is_done(ns_up))]
                P[s][RIGHT] = [(1.0, ns_right, reward, is_done(ns_right))]
                P[s][DOWN] = [(1.0, ns_down, reward, is_done(ns_down))]
                P[s][LEFT] = [(1.0, ns_left, reward, is_done(ns_left))]

            it.iternext()

            # 初始状态分布式一致的
            isd = np.ones(nS) / nS

            self.P = P

            super(GridworldEnv, self).__init__(nS, nA, P, isd)


    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            if self.s == s:
                output = " x "
            elif s == 0 or s == self.nS - 1:
                output = " T "
            else:
                output = " o "

            if x == 0:
                output = output.lstrip()
            if x == self.shape[1] - 1:
                output = output.rstrip()

            outfile.write(output)

            if x == self.shape[1] - 1:
                outfile.write("\n")

            it.iternext()