import sys
import random
import numpy as np


# let us build a markov decision process class
# based on the transition_probs & rewards format above
class MDP:
    def __init__(self, transition_probs, rewards):
        """
        Defines an MDP. Compatible with gym Env.
        :param transition_probs: transition_probs[s][a][s_next] = P(s_next | s, a)
            A dict[state -> dict] of dicts[action -> dict] of dicts[next_state -> prob]
            For each state and action, probabilities of next states should sum to 1
            If a state has no actions available, it is considered terminal
        :param rewards: rewards[s][a][s_next] = r(s,a,s')
            A dict[state -> dict] of dicts[action -> dict] of dicts[next_state -> reward]
            The reward for anything not mentioned here is zero.
        """

        self._transition_probs = transition_probs
        self._rewards = rewards
        self.n_states = len(transition_probs)
    
    def reset(self):
        """ reset the game, return the initial state"""
        self._current_state = np.random.choice(list(self._transition_probs.keys())[:-1])

    def get_all_states(self):
        """ return a tuple of all possiblestates """
        return tuple(self._transition_probs.keys())

    def get_possible_actions(self, state):
        """ return a tuple of possible actions in a given state """
        return tuple(self._transition_probs.get(state, {}).keys())

    def is_terminal(self, state):
        """ return True if state is terminal or False if it isn't """
        return len(self.get_possible_actions(state)) == 0

    def get_next_states(self, state, action,):
        """ return a list of next_state values """
        assert action in self.get_possible_actions(state), "cannot do action %s from state %s" % (action, state)
        return list(self._transition_probs[state][action].keys())

    def get_transition_prob(self, state, action, next_state):
        """ return P(next_state | state, action) """
        return self._transition_probs.get(state, {}).get(action, {}).get(next_state, 0.0)

    def get_reward(self, state, action, next_state):
        """ return the reward you get for taking action in state and landing on next_state"""
        assert action in self.get_possible_actions(state), "cannot do action %s from state %s" % (action, state)
        return self._rewards.get(state, {}).get(action, {}).get(next_state, 0.0)

    def step(self, action):
        """ take action, return next_state, reward, is_done, empty_info """
        s = self._current_state
        possible_states = self.get_next_states(s, action)
        probs = [self.get_transition_prob(s, action, v) for v in possible_states]
        next_state = np.random.choice(possible_states, p=probs)
        reward = self.get_reward(self._current_state, action, next_state)
        is_done = self.is_terminal(next_state)
        self._current_state = next_state
        
        return next_state, reward, is_done
    
    def generate_episodes(self,start_state=None):
        """ generates a single episode """
        is_done = False
        if start_state:
            self._current_state = start_state
        else:
            self.reset()
        episode = []
        while not is_done:
            s = self._current_state
            a = np.random.choice(self.get_possible_actions(s))
            next_state, reward, is_done = self.step(a)
            episode.append((s, a, reward, next_state))
        return episode
    
    
# test cases
# mdp = MDP(transition_probs, rewards)

# assert mdp.get_all_states() == ('s0', 's1', 's2')

# assert mdp.get_possible_actions('s0') == ('a0', 'a1')

# assert mdp.is_terminal('s0') == False

# assert mdp.get_next_states('s1', 'a1') == ['s1', 's2']

# assert mdp.get_reward('s1', 'a0', 's0') == 5

# assert mdp.get_transition_prob('s1', 'a1', 's0') == 0
# assert mdp.get_transition_prob('s1', 'a0', 's0') == 0.7