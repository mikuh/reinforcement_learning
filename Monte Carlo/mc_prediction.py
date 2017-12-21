import gym
import matplotlib
import numpy as np
import sys

from collections import defaultdict
from lib.envs.blackjack import BlackJackEnv
from lib import plotting


env = BlackJackEnv()


def mc_prediction(policy, env, num_episodes, discount_factor=1.0):
    """
    蒙特卡洛预测算法，为一个给定策略计算价值函数
    Args:
        policy: A function that maps an observation to action probabilities.
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.

    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    # Keeps track of sum and count of returns for each state 跟踪每个状态的总和和次数
    # to calculate an average. We could use an array to save all 计算平均值，可以用一个数组来存储所有数据
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    # 最终的价值函数
    V = defaultdict(float)

    for i_episode in range(1, num_episodes + 1):
        # 打印出我们在哪个episode 方便调试
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # 生成一个 episode. 是一个 [(state, action, reward),...]
        # An episode is an array of (state, action, reward) tuples
        episode = []
        state = env.reset()
        for t in range(100):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        # 找到所有访问过的 episode
        # 将所有的state转换成tuple好作为dict的key
        states_in_episode = set([tuple(x[0]) for x in episode])
        for state in states_in_episode:
            # Find the first occurrence of the state in the episode
            # 找到episode中第一次出现的state的索引
            first_occurrence_idx = next(i for i, x in enumerate(episode) if x[0] == state)
            # Sum up all rewards since the first occurrence
            # 从第一次出现开始的所有奖励求和
            G = sum([x[2] * (discount_factor ** i) for i, x in enumerate(episode[first_occurrence_idx:])])
            # Calculate average return for this state over all sampled episodes
            # 在所有的episodes上计算平均值
            returns_sum[state] += G
            returns_count[state] += 1.0
            V[state] = returns_sum[state] / returns_count[state]

    return V


def sample_policy(observation):
    """
    A policy that sticks if the player score is >= 20 and hits otherwise.
    """
    score, dealer_score, usable_ace = observation
    return 0 if score >= 20 else 1



V_10k = mc_prediction(sample_policy, env, num_episodes=10000)
plotting.plot_value_function(V_10k, title="10,000 Steps")

V_500k = mc_prediction(sample_policy, env, num_episodes=500000)
plotting.plot_value_function(V_500k, title="500,000 Steps")