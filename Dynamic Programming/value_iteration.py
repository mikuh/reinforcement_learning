"""
格子世界价值迭代
不需要给定一个策略函数
直接按照最大价值来决策
"""

import numpy as np
from lib.envs.gridworld import GridworldEnv


def value_iteration(env, discount_factor=1.0,  theta=0.00001):
    def one_step_lookahead(state, V):
        """
        Args:
            state: 要考虑的状态
            V: 当前估算的状态价值函数

        Returns:
            每个行动的期望价值
        """
        A = np.zeros(env.nA)  # 期望行动价值
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A

    # 初始化状态价值V
    V = np.zeros(env.nS)
    n = 0
    while True:
        n += 1
        # Stopping condition
        delta = 0
        # Update each state...
        for s in range(env.nS):
            # 前看一步找到最佳行动
            A = one_step_lookahead(s, V)        # 行动
            best_action_value = np.max(A)       # 最佳行动价值函数
            delta = max(delta, np.abs(best_action_value - V[s]))
            # 更新状态价值函数
            V[s] = best_action_value
        print("第{}次迭代".format(n))
        # 达到精度
        if delta < theta:
            break
    # 使用最优价值函数创建一个确定的策略
    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        # 向前看一步 找到当前最佳行动
        A = one_step_lookahead(s, V)
        best_action = np.argmax(A)
        # 总是采取最佳行动
        policy[s, best_action] = 1.0

    return policy, V


if __name__ == '__main__':
    env = GridworldEnv()
    # 通过价值迭代得到的策略
    policy, v = value_iteration(env)

    print("Policy Probability Distribution:")
    print(policy)
    print("")

    print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
    print(np.reshape(np.argmax(policy, axis=1), env.shape))
    print("")

    print("Value Function:")
    print(v)
    print("")

    print("Reshaped Grid Value Function:")
    print(v.reshape(env.shape))
    print("")