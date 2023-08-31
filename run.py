import jax
from jax import numpy as jnp
import jax.nn as nn

import matplotlib.pyplot as plt

from env.gridworld import Gridworld
from algs.opt import Optimizer, defaultLogger
from algs.returns import J

def main():
    key = jax.random.PRNGKey(0) 
    
    LR = 1
    STEPS = 20
    
    """Defining an MDP"""
    R = 100; P = -300; goals = [((2,0),R),((1,0),P),((1,1),P)]
    gridMDP = Gridworld(3,3,0.1,0.9,goals=goals,obstacles=[]) 
    gridMDP.init_distrib =  jnp.exp(jax.random.uniform(key,(gridMDP.n,))) / \
        jnp.sum(jnp.exp(jax.random.uniform(key,(gridMDP.n,))))

    pFun = lambda p : nn.softmax(p,axis=1) # policy function
    init = lambda : {'policy': jax.random.uniform(key,(gridMDP.n,gridMDP.m))}
    grad = lambda _, p : jax.jit(jax.grad(lambda p : J(gridMDP,pFun(p['policy']))))(p)
    proc = lambda g : {'policy': LR*g['policy']}
    
    def logger( params, grads, step, i):
        return {
            'J'         : J(gridMDP,pFun(params['policy'])),
            'params'    : params,
            'grads'     : grads,
            'step'      : step,
            'iter'      : i
        }
            
    
    opt = Optimizer(init=init,grad=grad,proc=proc,log=logger)
    _, log = opt.train(key,STEPS)
    
    perf = [e['J'] for e in log]
    plt.plot(perf)
    plt.show()
    
    
if __name__ == "__main__":
    main()