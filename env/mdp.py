import numpy as np

class MarkovDecisionProcess():
    def __init__(self,
            n  :int, 
            m :int, 
            gamma :float, 
            P_sa : np.ndarray,
            R : np.ndarray,
            ) -> None:
        self.n : int = n
        self.m : int = m
        self.gamma : float = gamma
        self.P_sa : np.ndarray = P_sa
        self.R : np.ndarray = R

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

        Args:
            V (np.ndarray): value function

        Returns:
            np.ndarray: updated value function
        """
        T = self.R + self.gamma * np.einsum('ijk,k',self.P_sa,V)
        return np.max(T,1)
    
    def expectation_operator(self, pi:np.ndarray, V:np.ndarray)->np.ndarray:
        R_pi = np.array([self.R[i,p] for i, p in enumerate(pi)])
        P_s = np.array([self.P_sa[i,p,:] for i, p in enumerate(pi)])
        return R_pi + self.gamma * P_s @ V
