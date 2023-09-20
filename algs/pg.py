import jax
from jax import numpy as jnp
from jax.numpy import linalg as jla

from typing import Callable

from env.mdp import MarkovDecisionProcess
from env.sample import Sampler
from algs.utils import flatten
from algs.grads import vanillaGradOracle, naturalGradOracle, exactFIMOracle, gpomdp, fimEstimator
from algs.projs import euclidean_l2ball

"""Parameters initialization methods"""
def initDirectPG(   k: jax.random.KeyArray,
                    mdp:MarkovDecisionProcess,
                    init_radius:float=1                )->Callable:
    """Initializes direct policy gradient parameters.

    Args:
        k (jax.random.KeyArray): jax PRNG key
        mdp (MarkovDecisionProcess): MDP
    """
    def init(k):
        p = jax.random.uniform(k,(mdp.n,mdp.m))*init_radius - init_radius/2
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
    pGrad = vanillaGradOracle(J,mdp,pFun,rFun, reg) 
    
    # TODO: compute exact FIM and then invert in that function (to allow for logging the FIM)
    def grad(key,p):
        _ = key
        _g = pGrad(p)
        _shape = p['policy'].shape
        _fim = exactFIMOracle(mdp,pFun, p['policy'])
        _pg = jnp.reshape(jla.pinv(_fim)@flatten(_g),_shape)
        _rg = jnp.zeros_like(p['reward'])
        # Here we append additional values to the grad object for logging purposes
        # specifically we pass the fisher-information related parameters
        return {
            'policy' : _pg,
            'exact_fim' : _fim,
            'exact_fim_pinv' : jla.pinv(_fim),
            'reward' : _rg,
        }
    return grad

def stochVanillaPG( J:Callable,
                    mdp:MarkovDecisionProcess,
                    smp:Sampler,
                    pFun:Callable,
                    rFun:Callable,
                    reg:Callable,
                    fim_reg:float)->Callable:
    pGrad = gpomdp(mdp,pFun,smp,reg)
    def grad(key,p):
        smp.key = key
        batch = smp.batch(pFun(p['policy']),regularizer=reg)
        _pg = pGrad(batch,p)
        _rg = jnp.zeros_like(p['reward'])
        return {
            'policy' : _pg,
            'reward' : _rg,
        }
    return grad

def stochNaturalPG( J:Callable,
                    mdp:MarkovDecisionProcess,
                    smp:Sampler,
                    pFun:Callable,
                    rFun:Callable,
                    reg:Callable,
                    fim_reg:float,)->Callable:
    pGrad = gpomdp(mdp,pFun,smp,reg)
    fim = fimEstimator(mdp,pFun,fim_reg)
    def grad(key,p):
        smp.key = key
        _exact_fim = exactFIMOracle(mdp,pFun,p['policy'])
        batch = smp.batch(pFun(p['policy']),regularizer=reg)
        _g = pGrad(batch,p); _fim = fim(batch,p)
        _shape = _g.shape
        _pg = jnp.reshape(jla.pinv(_fim)@flatten(_g),_shape)
        _rg = jnp.zeros_like(p['reward'])
        # Here we append additional values to the grad object for logging purposes
        # specifically we pass the fisher-information related parameters
        return { 
            'raw_policy_grads' :_g,
            'policy' : _pg,
            'reward' : _rg,
            'fim'    : _fim,
            'fim_pinv'    : jla.pinv(_fim),
            'exact_fim'    : _exact_fim,
            'exact_fim_pinv'    : jla.pinv(_exact_fim),
        }
    return grad

"""Gradient preprocessors"""
def pgDefaultProcessor(plr:float,ct:float)->Callable:
    def proc(g):
        return {
            'policy': jax.numpy.nan_to_num(
                    plr*g['policy'],
            copy=False, nan=0.0), 
            'reward': g['reward'], # not learning, just implement identity
            }
    return proc


"""Projection operations"""
def pgL2Proj(t:float)->Callable:
    def proj(x):
        x['policy']=euclidean_l2ball(x['policy'],t) # ensure this is a distribution
        return x
    return proj