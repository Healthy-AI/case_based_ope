import pandas as pd
import numpy as np

class AICPolicy():
    def __init__(self, path):
        policy = pd.read_csv(path, header=None)
        self.policy = policy.to_numpy()
        
    def predict_proba(self, states):
        probas = []
        for s in states:
            probas.append(self.policy[int(s), :])
        return np.array(probas)

class RandomPolicy():
    def __init__(self, n_actions=25, p=0.01):
        self.n_actions = n_actions
        self.p = p

    def predict_proba(self, states):
        n_states = len(states)
        probas = np.abs(np.zeros((n_states, self.n_actions))-self.p/(self.n_actions-1))
        for i in range(n_states):
            a = np.random.randint(low=0, high=self.n_actions)
            probas[i, a] = 1-self.p
        return probas

class ZeroDrugPolicy():
    def __init__(self, n_actions=25, p=0.01):
        self.n_actions = n_actions
        self.p = p

    def predict_proba(self, states):
        n_states = len(states)
        probas = np.abs(np.zeros((n_states, self.n_actions))-self.p/(self.n_actions-1))
        probas[:, 0] = 1-self.p
        return probas
