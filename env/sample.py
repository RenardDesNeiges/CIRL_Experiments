import jax
from jax import numpy as jnp
from jax.numpy import linalg as jla

from env.mdp import MarkovDecisionProcess

class Sampler():
    def __init__(self, MDP, key) -> None:
        self.MDP : MarkovDecisionProcess = MDP
        self.s_t :int = None
        self.key = key
        
    def _get_subkey(self):
        key, subkey = jax.random.split(self.key)
        self.key = key
        return subkey
        
    def reset(self, s_0: int = None):
        _key = self._get_subkey()
        if not s_0:
            p = self.MDP.init_distrib.astype('float64')
            p /= jnp.sum(p)
            self.s_t = jax.random.choice(_key,jnp.arange(self.MDP.n), p = p)
        else: 
            self.s_t = s_0
        return self.s_t # sets the initial state to 
    
    def step(self, action:int):
        _key = self._get_subkey()
        reward = self.MDP.R[self.s_t,action]
        p = self.MDP.P_sa[self.s_t,action,:].astype('float64')
        p /= jnp.sum(p)
        self.s_t = jax.random.choice(_key,jnp.arange(self.MDP.n), p = p)
        return self.s_t, reward