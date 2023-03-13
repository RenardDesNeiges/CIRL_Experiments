import numpy as np
import scipy.optimize as opt
from einops import rearrange

from env.mdp import MarkovDecisionProcess

class PrimalLP():
    def __init__(self, MDP:MarkovDecisionProcess) -> None:
        self.MDP : MarkovDecisionProcess = MDP
        self.V_opt = None # optimal value function
        
    def solve(self,)->np.ndarray:
        E = rearrange(
            np.array([  np.identity(self.MDP.n) 
                        for _ in range(self.MDP.m)], dtype=np.float64),
                        's a n -> (s a) n')        
        P = rearrange(self.MDP.P_sa, 's a n -> (a s) n')
        A_ub = - (E-self.MDP.gamma*P)
        b_ub = - rearrange(self.MDP.R, 's a -> (a s)')
        res = opt.linprog(self.MDP.init_distrib*(1-self.MDP.gamma),
                          A_ub, b_ub)
        self.V_opt = res.x
        return self.V_opt
        
    def recover_policy(self)->np.ndarray:
        T = self.MDP.R + \
            self.MDP.gamma * np.einsum('ijk,k',self.MDP.P_sa,self.V_opt)
        return np.argmax(T,1) # determinist policy solution
    
    
class DualLP():
    def __init__(self, MDP:MarkovDecisionProcess) -> None:
        self.MDP : MarkovDecisionProcess = MDP
        self.mu_opt = None # optimal occupancy measure
        
    def solve(self,)->np.ndarray:
        E_t = rearrange(
            np.array([  np.identity(self.MDP.n) 
                        for _ in range(self.MDP.m)], dtype=np.float64),
                        's a n -> (s a) n').transpose()
        P_t = rearrange(self.MDP.P_sa, 's a n -> (a s) n').transpose()
        A_eq = (E_t-self.MDP.gamma * P_t) 
        b_eq = (1-self.MDP.gamma) * self.MDP.init_distrib
        r = rearrange(self.MDP.R, 's a -> (a s)')
        res = opt.linprog(-r, # negative since we want to maximize
                          A_eq=A_eq, b_eq=b_eq,bounds=(0,None))
        self.mu_opt = np.array([[res.x[a*self.MDP.n+s]
                                for a in range(self.MDP.m)]
                                for s in range(self.MDP.n)],dtype=np.float64)
        return self.mu_opt
        
    def recover_policy(self)->np.ndarray:
        T = np.einsum('ij,i->ij',self.mu_opt,1/np.sum(self.mu_opt,1))
        return np.argmax(T,1)