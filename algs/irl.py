import jax
from jax import numpy as jnp
from jax.numpy import linalg as jla

from typing import Callable

from env.mdp import MarkovDecisionProcess
from env.sample import Sampler
from algs.utils import flatten
from algs.projs import euclidean_l1ball, euclidean_l2ball
from algs.grads import vanillaGradOracle, rewardGradOracle, naturalGradOracle



"""Parameters initialization methods"""
def initDirectIRL(   k: jax.random.KeyArray,
                    mdp:MarkovDecisionProcess)->Callable:
    """Initializes direct policy gradient parameters.

    Args:
        k (jax.random.KeyArray): jax PRNG key
        mdp (MarkovDecisionProcess): MDP
    """
    def init(k):
        k, sk = jax.random.split(k)
        p = jax.random.uniform(sk,(mdp.n,mdp.m))
        k, sk = jax.random.split(k) # we need to instantiate a param vector
        r = jax.random.uniform(sk,(mdp.n,mdp.m))
        return {
            'policy': p,
            'reward': r,
        }
        
    return lambda : init(k)

"""Gradient computation wrappers"""
def exactVanillaIRL( J:Callable,
                    mdp:MarkovDecisionProcess,
                    pFun:Callable,
                    rFun:Callable,
                    reg:Callable)->Callable:
    pGrad = vanillaGradOracle(J,mdp,pFun,rFun, reg)
    rGrad = rewardGradOracle(J,mdp,pFun,rFun,reg)
    def grad(key,p):
        _ = key
        _pg = pGrad(p)
        _rg = rGrad(p)
        return {
            'policy' : _pg,
            'reward' : _rg,
        }
    return grad

"""Gradient computation wrappers"""
def exactNaturalIRL(J:Callable,
                    mdp:MarkovDecisionProcess,
                    pFun:Callable,
                    rFun:Callable,
                    reg:Callable)->Callable:
    pGrad = naturalGradOracle(J,mdp,pFun,rFun, reg)
    rGrad = rewardGradOracle(J,mdp,pFun,rFun,reg)
    def grad(key,p):
        _ = key
        _pg = pGrad(p)
        _rg = rGrad(p)
        return {
            'policy' : _pg,
            'reward' : _rg,
        }
    return grad


"""Gradient preprocessors"""
def irlDefaultProcessor(plr:float,rlr:float,ct:float)->Callable:
    def proc(g):
        return {
            'policy': jax.numpy.nan_to_num(
                    plr*g['policy'],
            copy=False, nan=0.0), 
            'reward': -rlr*g['reward'], # not learning, just implement identity
            }
    return proc


"""Projection operations"""
def irlL2Proj(r:float, t:float)->Callable:
    def proj(x):
        x['policy']=euclidean_l2ball(x['policy'],t) # ensure this is a distribution
        x['reward']=euclidean_l2ball(x['reward'],r)
        return x
    return proj

