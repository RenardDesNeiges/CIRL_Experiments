"""env.sample

Contains the Sampler class for sampling batches from MDPs.

Titouan Renard, September 2023
"""
import jax
from jax import numpy as jnp

from typing import List, Tuple, Callable

from env.mdp import MarkovDecisionProcess

class Sampler():
    def __init__(self,  MDP:MarkovDecisionProcess, 
                        batchsize:int, 
                        horizon:int,
                        key:jax.random.KeyArray=jax.random.PRNGKey(0)) -> None:
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
        
        def initState(key):
            return jax.random.choice(key,
                                        jnp.arange(self.MDP.n),
                                        p=self.MDP.init_distrib)
        def nextAction(state,key):
            return jax.random.choice(key,
                                        jnp.arange(self.MDP.m),
                                        p=pi[state,:])
        def nextState(state,action,key):
            p = self.MDP.P_sa[state,action,:]
            return jax.random.choice(key,
                                        jnp.arange(self.MDP.n), 
                                        p = p)
        def getRewards(state,action):
            _r = self.MDP.R[state,action] 
            if regularizer is not None:
                _r -= regularizer(pi[state,:])§
            return _r
        
        
        @jax.jit
        def _inner_loop(it,el):
            _ = el
            s,key = it
            key, subkey = jax.random.split(key)
            keyArray = jax.random.split(subkey, self.b)
            a = jax.vmap(nextAction)(s,keyArray)
            r = jax.vmap(getRewards)(s,a)
            key, subkey = jax.random.split(key)
            keyArray = jax.random.split(subkey, self.b)
            _s = jax.vmap(nextState)(s,a,keyArray)
            return (_s,key), (s,a,r)

        keyArray = jax.random.split(self._get_subkey(), self.b)
        s = jax.vmap(initState)(keyArray)
        init = (s,self._get_subkey())
        _, result = jax.lax.scan(_inner_loop,init,None,length=self.h)
        
        # because I want batch to be the leading axis
        return tuple(jnp.swapaxes(e,0,1) for e in result)