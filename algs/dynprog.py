import numpy as np
from env.mdp import MarkovDecisionProcess

class ValueIterator():
    def __init__(self, MDP:MarkovDecisionProcess) -> None:
        self.MDP : MarkovDecisionProcess = MDP
        self.V_t : np.ndarray = np.zeros((self.MDP.n)) # |S|=n sized array for V(s)
    
    def optimality_operator(self):
        T = self.MDP.R + self.MDP.gamma * np.einsum('ijk,k',self.MDP.P_sa,self.V_t)
        self.V_t = np.max(T,1)
        
    def solve(self,steps=10):
        for s in range(steps):
            self.optimality_operator()
        return self.V_t