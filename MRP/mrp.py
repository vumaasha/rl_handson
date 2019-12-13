import sys
import random
import numpy as np


# let us build a markov decision process class
# based on the transition_probs & rewards format above
class MRP:
    def __init__(self, states, transition_probs, rewards):
        """
        Defines an MDP. Compatible with gym Env.

        :param states: the list of state names
        :param transition_probs: transition_probs[s][s_next] = P(s_next | s)
            A dict[state -> dict] of of dicts[next_state -> prob]
            For each state, probabilities of next states should sum to 1

        :param rewards: rewards[s] = r(s,s')
            A dict[state -> dict] of dicts[action -> dict] of dicts[next_state -> reward]
            The reward for anything not mentioned here is zero.
        """
        self.states = (states)
        self._transition_probs = np.array(transition_probs)
        self._rewards = rewards
        self.n_states = len(transition_probs)
    
    def reset(self):
        """ reset the game, return the initial state"""
        self._current_state = np.random.choice(self.n_states-1)

    def get_all_states(self):
        """ return a tuple of all possiblestates """
        return tuple(self.states)

    def is_terminal(self, state):
        """ return True if state is terminal or False if it isn't """
        # is the only possible next state is the same
        return self.get_next_states(state)[0] == state

    def get_next_states(self, state):
        """ return a list of next_state values """
        return np.where(self._transition_probs[state, :])[0]

    def get_transition_prob(self, state, next_state):
        """ return P(next_state | state, action) """
        return self._transition_probs[state, next_state]

    def step(self):
        """ take action, return next_state, reward, is_done, empty_info """
        s = self._current_state
        possible_states = self.get_next_states(s)
        probs = [self.get_transition_prob(s, v) for v in possible_states]
        next_state = np.random.choice(possible_states, 
                                      p=probs)
        reward = self._rewards[s]
        is_done = self.is_terminal(next_state)
        self._current_state = next_state
        
        return next_state, reward, is_done, {}
    
    def generate_episodes(self,start_state=None):
        is_done = False
        if start_state:
            self._current_state = self.states.index(start_state)
        else:
            self.reset()
        episode = []
        while not is_done:
            s = self._current_state
            next_state, reward, is_done, _ = self.step()
            episode.append((self.states[s], 
                            reward, self.states[next_state]))
        return episode