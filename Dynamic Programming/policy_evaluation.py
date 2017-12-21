"""
格子世界例子的策略评估
"""

import numpy as np
from lib.envs.gridworld import GridworldEnv


def policy_evaluation(policy, env, discount_factor=1.0, theta=0.00001):
    """给定一个策略policy
    环境env已知，即知道状态转移概率等
    折扣因子discount_factor默认设置为1.0
    状态价值计算精度theta=0.00001 迭代变化小于这个的时候停止迭代
    """
    # 初始化状态价值V
    V = np.zeros(env.nS)
    n = 0
    while True:
        n +=1
        delta = 0
        # For each state, perform a "full backup"
        for s in range(env.nS):
            v = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    v += action_prob * prob * (reward + discount_factor * V[next_state])

            delta = max(delta, np.abs(v - V[s]))    # 所有状态点最大的变化
            V[s] = v
        print("第{}次迭代".format(n))
        if delta < theta:
            break

    return np.array(V)



if __name__ == '__main__':
    env = GridworldEnv()  # 初始化格子环境模型
    # 给定一个随机策略pi
    pi = np.ones([env.nS, env.nA]) / env.nA
    # 给定策略的状态价值
    v = policy_evaluation(pi, env)
    print("Value Function:")
    print(v.reshape(env.shape))