"""
试用一下Q-learning 经过500个episode学习到的状态-行动价值函数Q
通过打印行走路线，可以很容易的看出 现在的基于行动价值函数Q的贪婪策略是最佳的策略
"""
from lib.envs.cliff_walking import CliffWalkingEnv
import pickle
import numpy as np
env = CliffWalkingEnv()

with open("Q-learning-policy.pic", 'rb') as f:
    Q = pickle.load(f)

# 初始状态

state = env.reset()
env.render()


while True:
    # 在某个状态采取某个行动之后的观察
    state, reward, is_done, prob = env.step(np.argmax(Q[state]))
    env.render()
    if is_done:
        break
