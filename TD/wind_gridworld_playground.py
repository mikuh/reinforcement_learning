"""
演示有风格子世界的每步动作转移以及即时回报等..
"""
from lib.envs.windy_gridworld import WindyGridworldEnv


UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


env = WindyGridworldEnv()


print(env.reset())
env.render()

# for i in range(12):
#     print(env.step(RIGHT))
#     env.render()