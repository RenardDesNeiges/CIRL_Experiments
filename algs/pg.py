import jax
from jax import numpy as jnp

from typing import Callable

from env.mdp import MarkovDecisionProcess
from env.sample import Sampler
from algs.grads import vanillaGradOracle, naturalGradOracle, gpomdp

"""Parameters initialization methods"""
def initDirectPG(   k: jax.random.KeyArray,
                    mdp:MarkovDecisionProcess)->Callable:
    """Initializes direct policy gradient parameters.

    Args:
        k (jax.random.KeyArray): jax PRNG key
        mdp (MarkovDecisionProcess): MDP
    """
    def init(k):
        p = jax.random.uniform(k,(mdp.n,mdp.m))
        r = mdp.R
        return {
            'policy': p,
            'reward': r,
        }
        
    return lambda : init(k)


"""Gradient computation wrappers"""
def exactVanillaPG( J:Callable,
                    mdp:MarkovDecisionProcess,
                    pFun:Callable,
                    rFun:Callable,
                    reg:Callable)->Callable:
    pGrad = vanillaGradOracle(J,mdp,pFun,rFun, reg)
    def grad(key,p):
        _ = key
        _pg = pGrad(p)
        _rg = jnp.zeros_like(p['reward'])
        return {
            'policy' : _pg,
            'reward' : _rg,
        }
    return grad

def exactNaturalPG( J:Callable,
                    mdp:MarkovDecisionProcess,
                    pFun:Callable,
                    rFun:Callable,
                    reg:Callable)->Callable:
    pGrad = naturalGradOracle(J,mdp,pFun,rFun, reg)
    def grad(key,p):
        _ = key
        _pg = pGrad(p)
        _rg = jnp.zeros_like(p['reward'])
        return {
            'policy' : _pg,
            'reward' : _rg,
        }
    return grad

def stochVanillaPG( J:Callable,
                    mdp:MarkovDecisionProcess,
                    smp:Sampler,
                    pFun:Callable,
                    rFun:Callable,
                    reg:Callable)->Callable:
    pGrad = gpomdp(mdp,pFun,smp)
    def grad(key,p):
        smp.key = key
        batch = smp.batch(pFun(p['policy']),reg)
        _pg = pGrad(batch,p)
        _rg = jnp.zeros_like(p['reward'])
        return {
            'policy' : _pg,
            'reward' : _rg,
        }
    return grad

"""Gradient preprocessors"""
def pgClipProcessor(plr:float,ct:float)->Callable:
    def proc(g):
        return {
            'policy': jnp.clip(plr*g['policy'],a_min=-ct,a_max=ct), 
            'reward': g['reward'], # not learning, just implement identity
            }
    return proc