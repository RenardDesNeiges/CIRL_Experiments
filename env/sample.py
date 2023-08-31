import jax
from jax import numpy as jnp
from jax.numpy import linalg as jla

from env.mdp import MarkovDecisionProcess


"""
Important Notes on the design of a proper sampling module!


Sampler should be able to sample trajectories and batches
Sampler should contain a batch size and a horizon parameter

"""

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
    
""" Generic sampling routines for MDPs """
def sample_trajectory(pi,mdp,smp,H,key,regularizer=None):
    def pick_action(pi,s,mdp):
        p = pi[s]; p /= jnp.sum(p)
        return jax.random.choice(key,jnp.arange(mdp.m), p = p)
        
    traj = []
    r_t = 0
    s_t = smp.reset()
    for _ in range(H):
        a_t = pick_action(pi,s_t,mdp)
        traj += [(s_t,a_t,r_t)]
        s_t, r_t = smp.step(a_t)
        if regularizer is not None:
            r_t -= regularizer(pi[s_t,:])
    return traj
def sample_batch(pi,mdp,smp,H,B,key,regularizer=None):
    subkeys = jax.random.split(key,B)
    return [sample_trajectory(pi,mdp,smp,H,k,regularizer) for k in subkeys]