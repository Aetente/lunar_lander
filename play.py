
import sys
# pylint: disable=F0401

import matplotlib.pyplot as plt
from collections import deque
from dqn import QNetwork
import gym
import numpy as np
import torch.nn as nn
import torch
import random


gamma = 0.95
batch_size = 64
step_c = 4000  # afer how many fixes we update the second model
experience_size = 1000000
training_frequency = 4
# 64


loaded_model = None
if len(sys.argv) > 1:
    loaded_model = sys.argv[1]


def discrete_action_to_continious_array(possible_actions, action):
    return possible_actions[action]


def play(n_episodes=100, max_t=1000):
    # for first train eps start is 1
    # for second eps start is 0.5

    # also I changed max_t to 500, because it would take a long time to land

    possible_actions = [
        # don't move
        [0, 0],

        # up
        [0.1, 0],
        [0.2, 0],
        [0.3, 0],
        [0.5, 0],
        [0.6, 0],
        [0.7, 0],
        [0.8, 0],
        [0.9, 0],
        [1, 0],

        # left
        [0, -0.6],
        [0, -0.7],
        [0, -0.8],
        [0, -0.9],

        # right
        [0, 0.6],
        [0, 0.7],
        [0, 0.8],
        [0, 0.9],

        # up-left
        # [0.8, -0.8],
        # [0.8, -0.65],
        # [0.6, -0.8],
        # [0.6, -0.65],
        # [0.7, -0.8],
        # [0.7, -0.65],

        # # up-right
        # [0.8, 0.8],
        # [0.8, 0.65],
        # [0.6, 0.8],
        # [0.6, 0.65],
        # [0.7, 0.8],
        # [0.7, 0.65]
    ]

    env = gym.make('LunarLanderContinuous-v2')
    # env = gym.make('LunarLander-v2')
    # action_space = env.action_space
    # print("action size=", action_space)
    n_actions = len(possible_actions)
    the_ship = QNetwork(state_size=8, action_size=n_actions)
    if (loaded_model):
        the_ship.load_state_dict(torch.load(loaded_model))

    the_ship.eval()
    scores_window = deque(maxlen=100)
    state_arr = []
    scores = []

    # run n games

    for i_episode in range(1, n_episodes+1):

        state = env.reset()
        score = 0

        # sample random transitions

        for t in range(max_t):
            state_arr = [state]
            input_s = torch.tensor(state_arr)
            qs = the_ship(input_s)
            the_ship.train()
            _, action = torch.max(qs.detach(), 1)
            action = action.detach().numpy()[0]

            env.render()
            action_continious = discrete_action_to_continious_array(
                possible_actions, action)
            next_state, reward, done, _ = env.step(action_continious)

            score += reward

            state = next_state

            if done:
                break

        scores_window.append(score)
        scores.append(score)

        mean_score = np.mean(scores_window)

        print('\rEpisode {}\tAverage Score: {:.2f}, Last reward: {}'.format(
            i_episode, mean_score, reward), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}, Last reward: {}'.format(
                i_episode, mean_score, reward))
    return scores


# scores = train()
# for second train use different values
scores = play()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
