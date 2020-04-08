# Q learning for openai's gym simple examples
import gym
import numpy as np

class QlearningGym:
    def __init__(self,action_space,observation_space,alpha,eps,gama):
        self.alpha = alpha # step size parameter
        self.eps = eps # epsilon greedy parameter
        self.gama = gama # discounting rate

        # Check states and actions are discrete
        assert action_space == gym.spaces.Discrete(action_space.n)
        assert observation_space == gym.spaces.Discrete(observation_space.n)

        # set the number of states and action
        self.action_num = action_space.n
        self.states_num = observation_space.n

        # Initialize Q estimate for all states and actions
        self.Q = np.zeros((self.states_num,self.action_num))
        # initialize policy
        self.e_greedy_policy = np.zeros((self.states_num,self.action_num))
        self.deterministic_policy = np.zeros((self.states_num,self.action_num))
        self.set_stochastic_policy()
        self.set_deterministic_policy()

    def set_stochastic_policy(self):
        for state_ind in range(self.states_num):
            max_val = np.amax(self.Q[state_ind])
            max_indices = np.argwhere(self.Q[state_ind]==max_val)
            max_len = float(len(max_indices))
            if max_len < self.action_num:
                max_prob = (1-self.eps)/max_len
                min_prob = self.eps/float(self.action_num-max_len)
                self.e_greedy_policy[state_ind] = np.ones((self.action_num))*min_prob
                self.e_greedy_policy[state_ind][max_indices] = max_prob
            else:
                prob = 1/float(self.action_num)
                self.e_greedy_policy[state_ind] = np.ones((self.action_num))*prob

    def set_deterministic_policy(self):
        for state_ind in range(self.states_num):
            max_val = np.amax(self.Q[state_ind])
            max_indices = np.argwhere(self.Q[state_ind]==max_val)
            max_len = float(len(max_indices))
            if max_len < self.action_num:
                max_prob = 1/max_len
                self.deterministic_policy[state_ind] = np.zeros((self.action_num))
                self.deterministic_policy[state_ind][max_indices] = max_prob
            else:
                prob = 1/float(self.action_num)
                self.deterministic_policy[state_ind] = np.ones((self.action_num))*prob

    def choose_stochastic_action_for_state(self,state_ind):
        actions = self.e_greedy_policy[state_ind]
        rand_action = np.random.rand()
        sum = 0
        for ind in range(len(actions)):
            sum += actions[ind]
            if sum > rand_action:
                return ind

    def choose_deterministic_action_for_state(self,state_ind):
        actions = self.deterministic_policy[state_ind]
        max_indices = np.argwhere(self.deterministic_policy[state_ind]==np.amax(actions))
        return np.random.choice(max_indices.flatten())

    def update_rule(self,state,action,reward,next_state):
        max_q = np.amax(self.Q[next_state])
        self.Q[state,action] += self.alpha*(reward + self.gama*max_q - self.Q[state,action])

    def set_q(self,q):
        self.Q = q