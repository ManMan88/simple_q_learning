from q_learning_gym import QlearningGym
import gym
import matplotlib.pyplot as plt
import numpy as np
from time import sleep

def run_training(env,learning_method,episodes):
    acumm_reward_array = []
    for i_episode in range(episodes):
        observation = env.reset()
        done = False
        acumm_reward = 0
        while not done:
            # env.render()
            action = learning_method.choose_stochastic_action_for_state(observation)
            last_observation = observation
            observation, reward, done, info = env.step(action)
            learning_method.update_rule(last_observation,action,reward,observation)
            learning_method.set_stochastic_policy()
            acumm_reward += reward
        
        acumm_reward_array.append(acumm_reward)
        print("finised episode number: " + str(i_episode+1) + " out of " + str(episodes))

    return acumm_reward_array

def run_trained_agent(env,learning_method,episodes):
    acumm_reward_array = []
    for i_episode in range(episodes):
        observation = env.reset()
        done = False
        acumm_reward = 0
        while not done:
            env.render()
            sleep(0.2)
            action = learning_method.choose_deterministic_action_for_state(observation)
            observation, reward, done, info = env.step(action)
            acumm_reward += reward
        acumm_reward_array.append(acumm_reward)

    return acumm_reward_array

if __name__ == "__main__":
    # Chose environment
    env = gym.make('Taxi-v3')
    alpha = 0.1
    eps = 0.1
    gama = 1.0

    ############################
    ##### Traind new agent #####
    ############################
    q_learning_method = QlearningGym(env.action_space,env.observation_space,alpha,eps,gama)
    num_of_episodes = 5000
    acumm_reward_array = run_training(env,q_learning_method,num_of_episodes)
    env.close()
    np.save("taxi.npy",q_learning_method.Q)

    plt.plot(range(num_of_episodes), acumm_reward_array)
    plt.xlabel('episode number')
    plt.ylabel('accumelated reward')
    plt.grid()
    plt.show()

    ############################
    ######## Test agent ########
    ############################
    q_learning_method = QlearningGym(env.action_space,env.observation_space,alpha,eps,gama)
    q_learning_method.set_q(np.load("taxi.npy"))
    q_learning_method.set_deterministic_policy()
    num_of_episodes = 10
    acumm_reward_array = run_trained_agent(env,q_learning_method,num_of_episodes)
    env.close()