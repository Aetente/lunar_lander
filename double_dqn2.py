
import sys
# pylint: disable=F0401

import matplotlib.pyplot as plt
from collections import deque
from dqn_net import QNetwork
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


def make_batch_from_experience(experience, batch_size):
    # experience_batch = random.sample(experience, batch_size)
    if (len(experience) <= batch_size):
        experience_batch = experience
    else:
        experience_batch = random.sample(experience, batch_size)
    state_array = []
    action_array = []
    reward_array = []
    next_state_array = []
    done_array = []

    for e in experience_batch:
        state_array.append(e[0])
        action_array.append(e[1])
        # change reward value to make the agent not try to commit suicide all the time
        reward_array.append(e[2])
        next_state_array.append(e[3])
        done_array.append(e[4])

    return [state_array, action_array, reward_array, next_state_array, done_array]


def train(n_episodes=3000, max_t=1000, eps_start=1.0, eps_end=0.015, eps_decay=0.99, lr=0.001, tau=0.5):
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
    the_ship2 = QNetwork(state_size=8, action_size=n_actions)

    if (loaded_model):
        the_ship.load_state_dict(torch.load(loaded_model))
        the_ship2.load_state_dict(torch.load(loaded_model))

    loss_function = nn.MSELoss()
    # loss_function = torch.nn.SmoothL1Loss()  # huber loss
    optimizer = torch.optim.Adam(the_ship.parameters(), lr=lr)
    # decreasing learning rate to 0.000001 definetly helped
    # it was for a big amount of actions
    # for when saving model to train it later I used 0.001 for learning rate and got mean score of -58 per 100 episodes
    # for the second attempt this learning rates gets the model stuck with mean score around -50 and -70 per 100 episodes

    experience = deque(maxlen=experience_size)

    scores = []

    scores_window = deque(maxlen=100)
    eps = eps_start

    the_ship.train()
    the_ship2.eval()

    net_fixes = 0
    samples = 0

    # run n games

    print("number of actions", n_actions)
    print("eps_start", eps_start)
    print("learning rate", lr)

    for i_episode in range(1, n_episodes+1):

        state = env.reset()
        score = 0

        # sample random transitions

        for t in range(max_t):
            samples += 1

            # run the actual 1 game
            if (random.random() < eps):
                # explore
                action = random.randrange(n_actions)
            else:
                # evaluate policy
                # there is a bug with batch normalization, where it demands more than 1 element in array of batch
                # I will look into that later if I have time
                # or switch to using dropout
                state_arr = [state]
                input_s = torch.tensor(state_arr)
                the_ship.eval()
                qs = the_ship(input_s)
                the_ship.train()
                _, action = torch.max(qs.detach(), 1)
                action = action.detach().numpy()[0]

            # just in case, I increased the amount of discrete actions
            # decreased batch size
            # used dropout
            # it seemed to work much better than before
            # also I removed random sampling, may be I should get it back

            # if (i_episode > 2500):
            # env.render()
            action_continious = discrete_action_to_continious_array(
                possible_actions, action)
            next_state, reward, done, _ = env.step(action_continious)
            # next_state, reward, done, _ = env.step(action)

            score += reward

            # reward *= reward
            # if (reward <= -100):
            #     reward *= 2

            # # we sample random actions to avoid correlations
            # if (random.random() > 0.2):
            #     # if True:
            #     # if (not done):
            #     #     reward = -10
            #     state_array.append(state)
            #     action_array.append(action)
            #     # change reward value to make the agent not try to commit suicide all the time
            #     reward_array.append(reward)
            #     next_state_array.append(next_state)
            #     done_value = 1
            #     if (done == True):
            #         done_value = 0
            #     done_array.append(done_value)

            done_value = 1
            if (done == True):
                done_value = 0

            experience.append([state, action, reward, next_state, done_value])

            state = next_state

            # when batch is full fix the net
            # if (len(state_array) >= batch_size):
            if (samples % training_frequency == 0):
                # if (samples > experience_size and samples % training_frequency == 0):
                state_array, action_array, reward_array, next_state_array, done_array = make_batch_from_experience(
                    experience, batch_size)

                net_fixes += 1
                # if (net_fixes % step_c == 0):  # update second model every C steps
                #     the_ship2.load_state_dict(the_ship.state_dict())

                all_curr_Qs = the_ship(torch.tensor(state_array))
                curr_Q = all_curr_Qs.gather(
                    dim=1, index=torch.tensor(action_array, dtype=torch.int64).unsqueeze(1)).squeeze(1)
                # .gather() is to gather the q-value for the chosen action

                next_Q = the_ship2(torch.tensor(next_state_array))
                max_next_Q = torch.max(next_Q, 1)[0]
                expected_Q = torch.tensor(reward_array) + \
                    torch.tensor(done_array) * gamma * max_next_Q
                # print("curr_Q", curr_Q)
                # print("expected_Q", expected_Q)

                loss = loss_function(curr_Q, expected_Q.detach().float())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                for target_param, param in zip(the_ship2.parameters(), the_ship.parameters()):
                    target_param.data.copy_(
                        tau * param + (1 - tau) * target_param)

            if done:
                break

        scores_window.append(score)
        scores.append(score)

        eps = max(eps_end, eps_decay*eps)

        mean_score = np.mean(scores_window)

        print('\rEpisode {}\tAverage Score: {:.2f}, Last reward: {}, Net fixes {}, eps: {}'.format(
            i_episode, mean_score, reward, net_fixes, eps), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}, Last reward: {}, Net fixes {}, eps: {}'.format(
                i_episode, mean_score, reward, net_fixes, eps))
        if mean_score >= 200.0:
            print('\nnSolved {:d} episodes\tAverage Score: {:.2f}'.format(
                i_episode-100, mean_score))
            torch.save(the_ship.state_dict(), 'checkpoint_win.pth')
            break
        if i_episode >= n_episodes - 1:
            print('\nEnvironment aaaaaaaaaaaa {:d} episodes!\tAverage Score: {:.2f}'.format(
                i_episode-100, mean_score))
            torch.save(the_ship.state_dict(), 'checkpoint_double_dqn_2.pth')
            break
    return scores


scores = train()
# for second train use different values
# scores = train(n_episodes=3000, max_t=500, eps_start=0.7,
#                eps_end=0.015, eps_decay=0.99, lr=0.00001)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
