import numpy as np
import scipy.optimize as opt

from env.mdp import MarkovDecisionProcess

class ValueIterator():
    def __init__(self, MDP:MarkovDecisionProcess) -> None:
        self.MDP : MarkovDecisionProcess = MDP
        self.V_t : np.ndarray = np.zeros((self.MDP.n)) # |S|=n sized array for V(s)
        
    def solve(self,steps:int=10)->np.ndarray:
        for s in range(steps):
            self.V_t = self.MDP.optimality_operator(self.V_t)
        return self.V_t
    
    def recover_policy(self)->np.ndarray:
        T = self.MDP.R + self.MDP.gamma * np.einsum('ijk,k',self.MDP.P_sa,self.V_t)
        return np.argmax(T,1)
        

class PolicyIterator():
    def __init__(self, MDP:MarkovDecisionProcess) -> None:
        self.MDP : MarkovDecisionProcess = MDP
        self.V_t : np.ndarray = np.zeros((self.MDP.n)) # |S|=n sized array for V(s)
        self.pi_t : np.ndarray = np.zeros((self.MDP.n,),dtype=np.uint32) # arbitrary initial policy pi
        
    def solve(self,steps:int=10, value_steps:int = 10)->np.ndarray:
        for s in range(steps):
            for k in range(value_steps): # policy evaluation
                self.V_t = self.MDP.expectation_operator(self.pi_t,self.V_t)
            self.pi_t = self.recover_policy() # policy update
        return self.pi_t, self.V_t
    
    def recover_policy(self)->np.ndarray:
        T = self.MDP.R + self.MDP.gamma * np.einsum('ijk,k',self.MDP.P_sa,self.V_t)
        return np.argmax(T,1)
        