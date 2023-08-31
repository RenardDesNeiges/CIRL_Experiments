import jax
from jax import numpy as jnp
import jax.nn as nn
from jax.config import config; config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt

from env.gridworld import Gridworld
from algs.opt import Optimizer
from algs.pg import *
from algs.returns import J

from env.sample import Sampler
    
def main():
    key = jax.random.PRNGKey(0) 
    
    LR = 1e-2
    CLIP_THRESH = 1e2
    STEPS = 30
    BATCH = 40
    HORIZON = 20
    
    """Defining an MDP"""
    R = 100; P = -300; goals = [((2,0),R),((1,0),P),((1,1),P)]
    gridMDP = Gridworld(3,3,0.1,0.9,goals=goals,obstacles=[]) 
    gridMDP.init_distrib =  jnp.exp(jax.random.uniform(key,(gridMDP.n,))) / \
        jnp.sum(jnp.exp(jax.random.uniform(key,(gridMDP.n,))))
        
    """Defining a sampler for the MDP"""
    smp = Sampler(MDP=gridMDP,key=key,batchsize=BATCH,horizon=HORIZON)

    """Defining the relevant function"""
    pFun = lambda p : nn.softmax(p,axis=1)  # policy function
    rFun = lambda r : r                     # reward function
    reg = None                              # regularizer function (or lack thereof)
    
    init = initDirectPG(key,gridMDP)                    # init function
    grad = stochNaturalPG(J,gridMDP,smp,pFun,rFun,reg)      # gradient function
    # grad = exactNaturalPG(J,gridMDP,pFun,rFun,reg)      # gradient function
    proc = pgClipProcessor(LR,CLIP_THRESH)
    
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