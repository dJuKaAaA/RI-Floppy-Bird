from network import Agent
import numpy as np
from game2 import Game

def test_learning():
    agent = Agent(gamma=0.9, epsilon=1.0, lr = 0.003, input_dims=[8], batch_siye=64,
                  n_actions=2)
    scores, eps_history = [], []
    n_games = 5
    for i in range(n_games):
        score = 0
        done = False
        observation = Game()
        observation.newGame()
        while not done:
            action = agent.chose_action(observation)
            observation_, reward, done = observation.updateSprites(0)
            score += reward
            agent.store_transition(observation, action=action, reward=reward, state_=observation_, done=done)
            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)
        avg = np.mean(scores)
        print('epizoda ', i, ' rezultat %.2f ' %score, ' prosek %.2f ' %avg,
              ' epsilon %.2f' %agent.epsilon)

if __name__=="__main__":
    test_learning()