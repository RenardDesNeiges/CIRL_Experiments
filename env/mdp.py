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
