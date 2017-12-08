"""
格子世界策略迭代
贪婪策略的状态价值和原策略的状态价值相比较，不一样则修改策略再次迭代
"""

import numpy as np
from lib.envs.gridworld import GridworldEnv
from policy_evaluation import policy_evaluation

def policy_improvement(env, policy_eval_fn=policy_evaluation, discount_factor=1.0):
    # 初始化一个随机策略
    policy = np.ones([env.nS, env.nA]) / env.nA
    # 迭代改进策略
    while True:
        V = policy_eval_fn(policy, env, discount_factor)
        # 策略稳定
        policy_stable = True

        # 遍历每个状态
        for s in range(env.nS):
            # 按照策略的概率选择行动
            chosen_a = np.argmax(policy[s])
            action_values = np.zeros(env.nA)
            # 计算每个行动的行动价值
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += prob * (reward + discount_factor * V[next_state])
            # 最佳行动是状态价值最大的那个行动
            best_a = np.argmax(action_values)

            if chosen_a != best_a:
                # 只要有一个状态的策略采取的行动与最佳行动不一致，就需要继续迭代
                policy_stable = False
            # 将策略更新为：最佳行动概率为1 其余为0
            policy[s] = np.eye(env.nA)[best_a]

        # 策略稳定了就返回
        if policy_stable:
            return policy, V


if __name__ == '__main__':
    env = GridworldEnv()
    policy, v = policy_improvement(env)

    print("Policy Probability Distribution:")
    print(policy)
    print("")

    print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
    print(np.reshape(np.argmax(policy, axis=1), env.shape))
    print("")

    print("Reshaped Value Function:")
    print(v.reshape(env.shape))
    print("")