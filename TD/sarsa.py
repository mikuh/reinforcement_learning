"""
sarsa算法总结
sarsa 是 on-policy方法 这种方法并不是求出最优策略 而是接近最优策略，因为为了能接近最优策略保留了探索
对所有的状态行为对s->a随机初始化行动价值函数Q,并且终止状态的Q=0
循环 n个episodes中每个episode
    初始化 S
    在S 通过epsilon-greedy 策略指定一个策略policy，
    循环 episode中的每一步
        基于起始状态S，根据当前epsilon-greedy,Q 得到policy, 根据policy采取行动A， 观察到即时回报R，和状态转移S'
        在S' 在通过 epsilon-greedy 指定的策略policy 采取行动 A'
        这样 我们就得到了S 新的估计行动价值 td_target = reward + discount_factor * Q[next_state][next_action]
        然后通过这个公式来更新状态行动价值 Q[state][action] += alpha * (td_target - Q[state][action])
        如果达到指定步数或者达到终止状态，就结束这个episode
"""
import itertools
import matplotlib
import numpy as np
import sys


from collections import defaultdict
from lib.envs.windy_gridworld import WindyGridworldEnv
from lib import plotting

matplotlib.style.use('ggplot')


env = WindyGridworldEnv()


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    基于给定的状态动作价值函数和epsilon创建一个epsilon-greedy策略
    Args:
        Q: 一个字典从 state -> action 映射到 values.
            Each value is a numpy array of length nA (see below)
        epsilon: 选择一个随机状态的概率 . float between 0 and 1.
        nA: 在环境中的动作个数.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA     # 其他动作的概率
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)               # 贪婪动作的概率
        return A

    return policy_fn


def sarsa(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    SARSA算法: on-policy TD 控制  找到最优epsilon-greedy 策略
    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
        A tuple (Q, stats).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
        Q 是行动价值函数，一个映射字典 state映射到action value
        stats 是一个 episodeStats对象 由 episode长度 和 episode奖励 两部分组成
    """

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    # 追踪每一个episode的步长和奖励用于统计
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # 我们遵循的策略
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        # 重置环境和获取第一个行动
        # 因为在WindyGridworldEnv类中定义的概率分布，初始都会从(3,0)这个固定点开始
        state = env.reset()
        action_probs = policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        # 一个episode中的一步
        for t in itertools.count():
            # 执行一步看看
            next_state, reward, done, _ = env.step(action)

            # 获取下一个行动
            next_action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)

            # 更新统计表
            stats.episode_rewards[i_episode] += reward      # 总回报
            stats.episode_lengths[i_episode] = t            # 总步数

            # TD 更新
            td_target = reward + discount_factor * Q[next_state][next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            if done:
                break

            action = next_action
            state = next_state

    return Q, stats


Q, stats = sarsa(env, 200)

plotting.plot_episode_stats(stats)