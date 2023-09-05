import jax
from jax import numpy as jnp

from typing import List, Tuple, Callable

from env.mdp import MarkovDecisionProcess

"""Note for further improvements: 
        - batch sampling should be vectorizable for a much faster implementation
"""

class Sampler():
    def __init__(self,  MDP:MarkovDecisionProcess, 
                        key:jax.random.KeyArray, 
                        batchsize:int, 
                        horizon:int) -> None:
        """Sampler class to sample MDPs

        Args:
            MDP (MarkovDecisionProcess): the MDP.
            key (jax.random.KeyArray): the initial jax PRNG key.
            batchsize (int): the batch size.
            horizon (int): the sampling horizon.
        """
        self.MDP : MarkovDecisionProcess = MDP
        self.b = batchsize
        self.h = horizon
        self.s_t :int = None
        self.key = key
        
    def _get_subkey(self)->jax.random.KeyArray:
        """Handles subkey splitting when sampling sequentially.

        Returns:
            jax.random.KeyArray: the key to use (inner key has been split)
        """
        key, subkey = jax.random.split(self.key)
        self.key = key
        return subkey
        
    def reset(self, s_0: int = None)->int:
        """Resets the MDP to some state.

        Args:
            s_0 (int, optional): Initial state. Defaults to None. If none is picked u.a.r. 

        Returns:
            int: the new state that was just set.
        """
        _key = self._get_subkey()
        if not s_0:
            p = self.MDP.init_distrib.astype('float64')
            p /= jnp.sum(p)
            self.s_t = jax.random.choice(_key,jnp.arange(self.MDP.n), p = p)
        else: 
            self.s_t = s_0
        return self.s_t 
    
    def step(self, action:int)->Tuple[int,float]:
        """Simulates one mdp step.

        Args:
            action (int): the action picked.

        Returns:
            Tuple[int,float]: a tuple containg the next state and the reward.
        """
        _key = self._get_subkey()
        reward = self.MDP.R[self.s_t,action]
        p = self.MDP.P_sa[self.s_t,action,:].astype('float64')
        p /= jnp.sum(p)
        self.s_t = jax.random.choice(_key,jnp.arange(self.MDP.n), p = p)
        return self.s_t, reward
        
    def trajectory(self,    pi:jnp.ndarray,
                            regularizer:Callable=None
                            )->List[Tuple[int,int,float]]: 
        """Samples a trajectory from the MDP.

        Args:
            pi (jnp.ndarray): The policy under which to sample.
            regularizer (Callable, optional): The regularizer to apply (for reward-discounting). Defaults to None.
        """
        def pick_action(pi,s):
            p = pi[s]; p /= jnp.sum(p)
            sk = self._get_subkey()
            return jax.random.choice(sk,jnp.arange(self.MDP.m), p = p)
            
        traj = []
        r_t = 0
        s_t = self.reset()
        for _ in range(self.h):
            a_t = pick_action(pi,s_t)
            traj += [(s_t,a_t,r_t)]
            s_t, r_t = self.step(a_t)
            if regularizer is not None:
                r_t -= regularizer(pi[s_t,:])
        return traj
    
    def batch(self,  pi:jnp.ndarray,
                            regularizer:Callable=None
                            )->Tuple[jnp.ndarray,jnp.ndarray,jnp.ndarray]:
        """Samples a batch of trajectories.

        Args:
            pi (jnp.ndarray): the policy under which to sample the trajectories.
            regularizer (Callable, optional): The regularizer to apply (for reward-discounting). Defaults to None.

        Returns:
            Tuple[jnp.ndarray,jnp.ndarray,jnp.ndarray]: the batch in the form (states,actions,rewards)
        """
        
        raw_batch=[self.trajectory(pi,regularizer) for _ in range(self.b)]
        s_batch = jnp.array([[e[0] for e in t] for t in raw_batch])
        a_batch = jnp.array([[e[1] for e in t] for t in raw_batch])
        r_batch = jnp.array([[e[2] for e in t] for t in raw_batch])
        return (s_batch,a_batch,r_batch)