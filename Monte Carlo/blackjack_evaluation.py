from lib.envs.blackjack import BlackJackEnv


env = BlackJackEnv()


def print_observation(observation):
    """打印观察
    """
    score, dealer_score, usable_ace = observation
    print("Player Score: {} (Usable Ace: {}), Dealer Score: {}".format(
          score, usable_ace, dealer_score))

def strategy(observation):
    """策略
    如果是20点以上就停止要牌，否则要牌
    """
    score, dealer_score, usable_ace = observation
    # Stick (action 0) if the score is > 20, hit (action 1) otherwise
    return 0 if score >= 20 else 1


end_reward_list = []
for i_episode in range(20):
    observation = env.reset()   # 初始观察
    for t in range(100):
        print_observation(observation)
        action = strategy(observation)  # 行动
        print("Taking action: {}".format(["Stick", "Twist"][action]))
        observation, reward, done, _ = env.step(action)
        if done:
            print_observation(observation)
            end_reward_list.append((observation, reward))
            print("Game end. Reward: {}\n".format(float(reward)))
            break

print("最终的状态回报情况：")
print(end_reward_list)