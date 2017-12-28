"""
Q-Learning 算法
是一个Off-policy 的方法，它基于epsilon-greedy policy 来选择行动进行采样，但是并不基于该策略来估算价值
而是根据argmax Q(state) 贪婪选择 行动价值最高的动作来作为 下一步的行动价值的学习目标，于是更新Q的方式是：
Q[state][action] <-- Q[state][action] + alpha * (reward + discount_factor * (Q[next_state][best_next_action] - Q[state][action]))
这一点与sarsa方法不同，sarsa方法是基于epsilon-greedy policy 选择行动来采样，同时下一步行动价值的目标也是通过epsilon-greedy policy  来进行估算的，也就是说更新Q的方式是：
Q[state][action] <-- Q[state][action] + alpha * (reward + discount_factor * (Q[next_state][epsilon-greedy policy next_action] - Q[state][action]))
"""
import gym
import itertools
import matplotlib
import numpy as np
import pickle
import sys


if "../" not in sys.path:
  sys.path.append("../")

from collections import defaultdict
from lib.envs.cliff_walking import CliffWalkingEnv
from lib import plotting

matplotlib.style.use('ggplot')



env = CliffWalkingEnv()


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    Q-Learning 算法：Off-policy TD 控制，遵循一个epsilon-greedy policy 找到最佳贪婪策略
    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # The policy we're following
    # 我们所遵循的epsilon-greedy 策略
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

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
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            if done:
                break

            state = next_state

    return Q, stats


Q, stats = q_learning(env, 500)

with open("Q-learning-policy.pic", 'wb') as f:
    pickle.dump(dict(Q), f)
# plotting.plot_episode_stats(stats)