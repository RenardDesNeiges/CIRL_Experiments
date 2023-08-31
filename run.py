import jax
from jax import numpy as jnp
import jax.nn as nn

from typing import Callable

import matplotlib.pyplot as plt

from env.mdp import MarkovDecisionProcess
from env.gridworld import Gridworld
from algs.opt import Optimizer
from algs.grads import vanillaGradOracle, naturalGradOracle
from algs.returns import J

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

def exactVanillaPG( J:Callable,
                    mdp:MarkovDecisionProcess,
                    pFun:Callable,
                    rFun:Callable,
                    reg:Callable)->Callable:
    pGrad = vanillaGradOracle(J,mdp,pFun,rFun, reg)
    def grad(batch,p):
        _pg = pGrad(batch,p)
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
    def grad(batch,p):
        _pg = pGrad(batch,p)
        _rg = jnp.zeros_like(p['reward'])
        return {
            'policy' : _pg,
            'reward' : _rg,
        }
    return grad

def pgClipProcessor(plr:float,ct:float)->Callable:
    def proc(g):
        return {
            'policy': jnp.clip(plr*g['policy'],a_min=-ct,a_max=ct), 
            'reward': g['reward'], # not learning, just implement identity
            }
    return proc
    
    
def main():
    key = jax.random.PRNGKey(0) 
    
    LR = 1
    STEPS = 20
    
    """Defining an MDP"""
    R = 100; P = -300; goals = [((2,0),R),((1,0),P),((1,1),P)]
    gridMDP = Gridworld(3,3,0.1,0.9,goals=goals,obstacles=[]) 
    gridMDP.init_distrib =  jnp.exp(jax.random.uniform(key,(gridMDP.n,))) / \
        jnp.sum(jnp.exp(jax.random.uniform(key,(gridMDP.n,))))

    """Defining the relevant function"""
    pFun = lambda p : nn.softmax(p,axis=1)  # policy function
    rFun = lambda r : r                     # reward function
    reg = None                              # regularizer function (or lack thereof)
    
    init = initDirectPG(key,gridMDP)                # init function
    grad = exactVanillaPG(J,gridMDP,pFun,rFun,reg)     # gradient function
    proc = pgClipProcessor(LR,1e3)
    
    """Defining the logger"""
    def logger( params, grads, step, i):
        return {
            'J'         : J(gridMDP,pFun(params['policy'])),
            'params'    : params,
            'grads'     : grads,
            'step'      : step,
            'iter'      : i
        }
            
    """Optimizing"""
    opt = Optimizer(init=init,grad=grad,proc=proc,log=logger)
    _, log = opt.train(key,STEPS,True)
    
    """Plotting the results"""
    i = [e['iter'] for e in log]
    j = [e['J'] for e in log]
    plt.plot(i,j)
    plt.show()
    
    
if __name__ == "__main__":
    main()