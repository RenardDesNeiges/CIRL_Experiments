import jax
from jax import numpy as jnp
import jax.nn as nn

import matplotlib.pyplot as plt

from env.gridworld import Gridworld
from algs.opt import Optimizer, initDirectPG
from algs.grads import naturalGradOracle
from algs.returns import J

def main():
    key = jax.random.PRNGKey(0) 
    
    LR = 5e-3
    STEPS = 20
    
    """Defining an MDP"""
    R = 100; P = -300; goals = [((2,0),R),((1,0),P),((1,1),P)]
    gridMDP = Gridworld(3,3,0.1,0.9,goals=goals,obstacles=[]) 
    gridMDP.init_distrib =  jnp.exp(jax.random.uniform(key,(gridMDP.n,))) / \
        jnp.sum(jnp.exp(jax.random.uniform(key,(gridMDP.n,))))

    """Defining the relevant function"""
    pFun = lambda p : nn.softmax(p,axis=1) # policy function
    init = initDirectPG(key,gridMDP)
    pGrad = naturalGradOracle(J,gridMDP,pFun,lambda x:x, None)
    def grad(batch,p):
        _pg = pGrad(batch,p)
        _rg = jnp.zeros_like(p['reward'])
        return {
            'policy' : _pg,
            'reward' : _rg,
        }
        
    proc = lambda g : {'policy': LR*g['policy'], 'reward': g['reward']}
    
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