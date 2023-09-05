import jax
from jax import numpy as jnp
from jax.numpy import linalg as jla

from einops import repeat
from itertools import accumulate

class MarkovDecisionProcess():
    def __init__(self,
            n  :int, 
            m :int, 
            gamma :float, 
            P_sa : jnp.ndarray,
            R : jnp.ndarray,
            init_distrib : jnp.ndarray,
            b : jnp.ndarray = None,
            Psi : jnp.ndarray = None,
            ) -> None:
        self.n : int = n
        self.m : int = m
        self.gamma : float = gamma
        self.P_sa : jnp.ndarray = P_sa
        self.R : jnp.ndarray = R
        self.init_distrib : jnp.ndarray = init_distrib
        self.b = b
        self.Psi = Psi

    def closed_loop_kernel(self, pi:jnp.ndarray)->jnp.ndarray:
        """Computes the closed loop transition kernel P of the mdp under policy pi.

        Args:
            pi (jnp.ndarray): policy for which we want to compute P

        Returns:
            jnp.ndarray: nm*nm-sized array containing the kernel
        """
        return jnp.einsum('sap,sa->sp',self.P_sa,pi)
    
    def state_occ_measure(self,pi):
        """Computes the state occupancy measure

        Args:
            pi (jnp.ndarray): policy for which we want to compute P

        Returns:
            jnp.ndarray: n-sized array containing the state occupancy measure
        """
        if self.n == 1:
            return jnp.array([1.])
        P_pi = self.closed_loop_kernel(pi)
        return (1-self.gamma)*(jla.inv((jnp.eye(self.n)-self.gamma*P_pi)).transpose()@self.init_distrib)

    def occ_measure(self,pi):
        """Computes the occupancy measure

        Args:
            pi (jnp.ndarray): policy for which we want to compute P

        Returns:
            jnp.ndarray: n*m-sized array containing the occupancy measure
        """
        mu_s = self.state_occ_measure(pi)
        return pi * repeat(mu_s, 's -> s new_axis', new_axis=self.m)

    def next_state_distribution(self, s:int, a:int)->jnp.ndarray:
        """Given a fixed state-action pair, gives the distribution on the next state.

        Args:
            s (int): current state s
            a (int): action a

        Returns:
            jnp.ndarray: n-sized array containing the distribution of the random variable s'
        """
        return self.P_sa[s,a,:]
    
    def optimality_operator(self, V:jnp.ndarray)->jnp.ndarray:
        """Bellman optimality operator
        Note that this operator has no meaning when the MDP is constrained
        
        Args:
            V (jnp.ndarray): value function

        Returns:
            jnp.ndarray: updated value function
        """
        T = self.R + self.gamma * jnp.einsum('ijk,k',self.P_sa,V)
        return jnp.max(T,1)
    
    def expectation_operator(self, pi:jnp.ndarray, V:jnp.ndarray)->jnp.ndarray:
        """Note that this operator has no meaning when the MDP is constrained
        """
        R_pi = jnp.array([self.R[i,p] for i, p in enumerate(pi)])
        P_s = jnp.array([self.P_sa[i,p,:] for i, p in enumerate(pi)])
        return R_pi + self.gamma * P_s @ V
