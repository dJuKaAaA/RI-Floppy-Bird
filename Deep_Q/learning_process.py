from network import Agent
import numpy as np
from game2 import Game
import cv2
import torch
from settings import WINDOW_WIDTH, WINDOW_HEIGHT

def pre_processing(image, width, height):
    image = cv2.cvtColor(cv2.resize(image, (width, height)), cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    return image[None, :, :].astype(np.float32)


def test_learning(n_games, agent):
    
    scores, eps_history = [], []
    for i in range(n_games):
        score = 0
        alive = True
        game = Game()
        observation, reward, alive = game.newGame()
        # observation = pre_processing(observation[:WINDOW_WIDTH, :WINDOW_HEIGHT], 84, 84)
        # print(observation)
        # observationT = torch.from_numpy(observation)
        # state = torch.cat(tuple(observationT for _ in range(4)))
        # print(state)
        # state = torch.cat(tuple(observationT for _ in range(4)))[None, :, :, :]
        # print(state)

        while alive:
            action = agent.chose_action(observation)
            observation_, reward, alive = game.updateSprites(action)
            # observation_ = pre_processing(observation_[:WINDOW_HEIGHT, :WINDOW_HEIGHT], 84, 84)
            # observationT_ = torch.from_numpy(observation_)
            # new_state = state = torch.cat(tuple(observationT_ for _ in range(4)))[None, :, :, :]
            score += reward
            # agent.store_transition(observation, action=action, reward=reward, state_=observation_, done=alive)
            # agent.learn()
            agent.save_gotten_information(torch.tensor(observation), action=action, reward=reward, state_=torch.tensor(observation_), done=alive)
            observation = observation_
            # state = new_state
        scores.append(score)
        eps_history.append(agent.epsilon)
        avg = np.mean(scores)
        print('epizoda ', i, ' rezultat %.2f ' %score, ' prosek %.2f ' %avg,
              ' epsilon %.2f' %agent.epsilon)

if __name__=="__main__":
    agent = Agent(gamma=0.9, epsilon=1.0, lr = 0.003, input_dims=[4], batch_siye=64,
                  n_actions=2)
    test_learning(31, agent)
