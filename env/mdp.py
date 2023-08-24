import numpy as np
from numpy import random as rd

class Sampler():
    def __init__(self, MDP) -> None:
        self.MDP : MarkovDecisionProcess = MDP
        self.s_t :int = None
        
    def reset(self, s_0: int = None):
        if not s_0:
            p = self.MDP.init_distrib.astype('float64')
            p /= np.sum(p)
            self.s_t = rd.choice(np.arange(self.MDP.n), p = p)
        else: 
            self.s_t = s_0
        return self.s_t # sets the initial state to 
    
    def step(self, action:int):
        reward = self.MDP.R[self.s_t,action]
        p = self.MDP.P_sa[self.s_t,action,:].astype('float64')
        p /= np.sum(p)
        self.s_t = rd.choice(np.arange(self.MDP.n), p = p)
        return self.s_t, reward

class MarkovDecisionProcess():
    def __init__(self,
            n  :int, 
            m :int, 
            gamma :float, 
            P_sa : np.ndarray,
            R : np.ndarray,
            init_distrib : np.ndarray,
            b : np.ndarray = None,
            Psi : np.ndarray = None,
            ) -> None:
        self.n : int = n
        self.m : int = m
        self.gamma : float = gamma
        self.P_sa : np.ndarray = P_sa
        self.R : np.ndarray = R
        self.init_distrib : np.ndarray = init_distrib
        self.b = b
        self.Psi = Psi

    def next_state_distribution(self, s:int, a:int)->np.ndarray:
        """Given a fixed state-action pair, gives the distribution on the next state.

        Args:
            s (int): current state s
            a (int): action a

        Returns:
            np.ndarray: n-sized array containing the distribution of the random variable s'
        """
        return self.P_sa[s,a,:]
    
    def optimality_operator(self, V:np.ndarray)->np.ndarray:
        """Bellman optimality operator
        Note that this operator has no meaning when the MDP is constrained
        
        Args:
            V (np.ndarray): value function

        Returns:
            np.ndarray: updated value function
        """
        T = self.R + self.gamma * np.einsum('ijk,k',self.P_sa,V)
        return np.max(T,1)
    
    def expectation_operator(self, pi:np.ndarray, V:np.ndarray)->np.ndarray:
        """Note that this operator has no meaning when the MDP is constrained
        """
        R_pi = np.array([self.R[i,p] for i, p in enumerate(pi)])
        P_s = np.array([self.P_sa[i,p,:] for i, p in enumerate(pi)])
        return R_pi + self.gamma * P_s @ V
