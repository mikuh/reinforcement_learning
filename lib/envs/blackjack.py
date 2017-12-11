"""
21点模型

"""
import gym
from gym import spaces
from gym.utils import seeding


def cmp(a, b):
    return int((a > b)) - int((a < b))


# 牌组 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

def draw_card(np_random):
    """随机抽牌
    """
    return np_random.choice(deck)


def draw_hand(np_random):
    """初始手牌，直接发两张
    """
    return [draw_card(np_random), draw_card(np_random)]


def usable_ace(hand):
    """是否使用手牌ace的特殊效果，手上有ace并且牌面不爆炸的话就用
    """
    return 1 in hand and sum(hand) + 10 <= 21


def sum_hand(hand):
    """返回当前的点数
    """
    if usable_ace(hand):
            return sum(hand) + 10
    return sum(hand)


def is_bust(hand):
    """点数是否爆炸
    """
    return sum_hand(hand) > 21

def score(hand):  # What is the score of this hand (0 if bust)
    return 0 if is_bust(hand) else sum_hand(hand)


def is_natural(hand):  # Is this hand a natural blackjack?
    return sorted(hand) == [1, 10]


class BlackJackEnv(gym.Env):
    """21点环境
    """
    def __init__(self, natural=False):
        # 行动空间
        self.action_space = spaces.Discrete(2)
        # 观察
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),                    # 当前点数
            spaces.Discrete(11),                    # 庄家明牌
            spaces.Discrete(2)))                    # 我有没有ace
        # 随机因子
        self._seed()

        # Flag to payout 1.5 on a "natural" blackjack win, like casino rules
        # Ref: http://www.bicyclecards.com/how-to-play/blackjack/
        self.natural = natural
        # Start the first game
        self._reset()           # Number of
        self.nA = 2             # 两种行动  要牌和不要


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action)
        if action:  # hit: add a card to players hand and return
            self.player.append(draw_card(self.np_random))       # 增加一张牌
            if is_bust(self.player):
                done = True                                     # 终止
                reward = -1                                     # 奖励
            else:
                done = False
                reward = 0
        else:  # stick: play out the dealers hand, and score
            done = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(draw_card(self.np_random))
            reward = cmp(score(self.player), score(self.dealer))
            if self.natural and is_natural(self.player) and reward == 1:
                reward = 1.5
        return self._get_obs(), reward, done, {}


    def _get_obs(self):
        """当前的观察
        return 一个tuple(我的点数, 庄家明牌, 我是否有ace)
        """
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player))

    def _reset(self):
        # 初始化庄家和玩家手牌
        self.dealer = draw_hand(self.np_random)
        self.player = draw_hand(self.np_random)

        # 不到12的话自动要牌
        while sum_hand(self.player) < 12:
            self.player.append(draw_card(self.np_random))

        return self._get_obs()



if __name__ == '__main__':
    for i in range(100):
        np_random, seed = seeding.np_random(None)
        print(np_random, seed)
        print(draw_card(np_random))