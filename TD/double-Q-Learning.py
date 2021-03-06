"""
Double Q Learning 算法
"""

import itertools
import matplotlib
import numpy as np
import pickle
import sys
from collections import defaultdict
from lib.envs.cliff_walking import CliffWalkingEnv
from lib import plotting

matplotlib.style.use('ggplot')



env = CliffWalkingEnv()


def make_epsilon_greedy_policy(Q1, Q2, epsilon, nA):
    """
    创建一个 epsilon-greedy policy based on given Q1-function and Q2-function and epsilon
    Args:
        Q1: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        Q2: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q1[observation] + Q2[observation])
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def double_q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    Double Q learning 算法 为了规避 最大化偏差
    """

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q1 = defaultdict(lambda: np.zeros(env.action_space.n))
    Q2 = defaultdict(lambda: np.zeros(env.action_space.n))

    # 记录有用的统计信息
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # The policy we're following
    # 我们所遵循的epsilon-greedy 策略
    policy = make_epsilon_greedy_policy(Q1, Q2, epsilon, env.action_space.n)

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        # Reset the environment and pick the first action
        state = env.reset()

        # One step in the environment
        # total_reward = 0.0
        for t in itertools.count():

            # Take a step
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            # TD Update
            if np.random.random() > 0.5:
                best_next_action = np.argmax(Q1[next_state])
                td_target = reward + discount_factor * Q2[next_state][best_next_action]
                td_delta = td_target - Q1[state][action]
                Q1[state][action] += alpha * td_delta
            else:
                best_next_action = np.argmax(Q2[next_state])
                td_target = reward + discount_factor * Q1[next_state][best_next_action]
                td_delta = td_target - Q2[state][action]
                Q2[state][action] += alpha * td_delta

            if done:
                break

            state = next_state

    return Q1, Q2, stats


Q1, Q2, stats = double_q_learning(env, 500)

# with open("Q-learning-policy.pic", 'wb') as f:
#     pickle.dump(dict(Q), f)
plotting.plot_episode_stats(stats)