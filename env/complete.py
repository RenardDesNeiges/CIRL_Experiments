import numpy as np
import numpy.linalg as la
import numpy.random as rd
from .mdp import MarkovDecisionProcess

class Complete(MarkovDecisionProcess):
    """Defines a complete MDP
    """
    def __init__(self,
                n  :    int = 2, 
                m  :    int = 2, 
                gamma : float = 0.9, 
                seed :  int = 1,
                rmax :  float = 10.,
                b    :  np.array = None,
                Psi  :  np.array = None,
                ) -> None:
        """
        Args:
            n (int, optional): State set size. Defaults to 2.
            m (int, optional): Action set size. Defaults to 2.
            gamma (float, optional): Discount factor. Defaults to 0.9.
            seed (int, optional): Random seed (for reproductibility). Defaults to 1.
            rmax (float, optional): Maximum reward. Defaults to 10..
        """
        
        rd.seed(seed)
        self.rmax   : float = rmax
        self.n      : int = n
        self.m      : int = m
        
        # TODO : allow for passing a pre-computed matrix to the MDP directly?
        
        P_sa : np.ndarray = np.array(  [[self._get_state_distrib() 
                                            for _  in range(self.n)      ] 
                                            for _  in range(self.m)             ],
                                     dtype=np.float64) # P ∈ (n x m x n)
        
        R : np.ndarray = np.array(  [[self._reward()
                                            for _  in range(self.m)             ] 
                                            for _  in range(self.n)             ],dtype=np.float64) # P ∈ (n x m)
        
        super().__init__(self.n,self.m,gamma,P_sa,R,self._get_state_distrib(),b=b,Psi=Psi)


    
    def _get_state_distrib(self,) -> np.ndarray:
        """Returns the markov transition probability P(sp|s,a) as a n=|S| sized vector
        that satisfies Σ_s' P(s'|s,a) = 1 

        Returns:
            np.ndarray: a valid disiribution for the P matrix col
        """
        d = rd.rand(self.n)
        d = d/la.norm(d,1) # make sure d is an actual distribution
        return d
        
    def _reward(self,)->float:
        return rd.rand()*self.rmax
    