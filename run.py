import jax
from jax import numpy as jnp
from jax.numpy import linalg as jla
import jax.nn as nn
from jax.config import config; config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt

from env.gridworld import Gridworld,gridplot
from algs.opt import Optimizer
from algs.pg import *
from algs.irl import *
from algs.returns import J, IRLLagrangian
from algs.utils import shannonEntropy

from env.sample import Sampler
    
def main_pg():
    key = jax.random.PRNGKey(0) 
    
    LR = 1e-2
    CLIP_THRESH = 1e2
    STEPS = 30
    BATCH = 40
    HORIZON = 20
    BETA = 2
    
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
    reg = lambda p : BETA * shannonEntropy(p)
    
    init = initDirectPG(key,gridMDP)                    # init function
    # grad = stochNaturalPG(J,gridMDP,smp,pFun,rFun,reg)      # gradient function
    grad = exactNaturalPG(J,gridMDP,pFun,rFun,reg)      # gradient function
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
    opt = Optimizer(init=init,grad=grad,proc=proc,log=logger,proj=lambda x: x)
    op, log = opt.train(key,STEPS,True)
    
    """Plotting the results"""
    i = [e['iter'] for e in log]
    j = [e['J'] for e in log]
    plt.plot(i,j)
    plt.show()
    
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    gridplot(gridMDP,ax,goals=goals,stochastic_policy=pFun(op['policy']))
    plt.show()
    
def getExpertPolicy(mdp,key,beta):
    LR = 5e-3
    CLIP_THRESH = 5e2
    STEPS = 40
    pFun = lambda p : nn.softmax(p,axis=1)  # policy function
    rFun = lambda r : r                     # reward function
    reg = lambda p : beta * shannonEntropy(p)
    
    init = initDirectPG(key,mdp)                    # init function
    grad = exactNaturalPG(J,mdp,pFun,rFun,reg)      # gradient function
    proc = pgClipProcessor(LR,CLIP_THRESH)
    
    opt = Optimizer(init=init,grad=grad,proc=proc,log=lambda x,_g,_s,i : None,proj=lambda x: x)
    p, _ = opt.train(key,STEPS,True)
    return pFun(p['policy'])
    
def main_irl():
    key = jax.random.PRNGKey(0) 
    
    PLR = 1
    RLR = 1e1
    CLIP_THRESH = 5e2
    STEPS = 400
    W_RADIUS = 1
    BETA = 3
    
    """Defining an MDP"""
    R = 100; P = -300; 
    # goals = [((2,0),R),((1,0),P),((1,1),P)]
    goals = [((2,0),R)]
    gridMDP = Gridworld(3,3,0.1,0.9,goals=goals,obstacles=[]) 
    gridMDP.init_distrib = jnp.exp(jax.random.uniform(key,(gridMDP.n,))) / \
        jnp.sum(jnp.exp(jax.random.uniform(key,(gridMDP.n,))))
        
    """Recovering the expert policy (that we are going recover to clone)"""
    print("Computing the expert's policy")
    expertPolicy = getExpertPolicy(gridMDP,key,BETA)
    L = IRLLagrangian(expertPolicy)
    
    """Defining the relevant function"""
    pFun = lambda p : nn.softmax(p,axis=1)  # policy function
    rFun = lambda r : r                     # reward function (for now we just parametrize directly)
    reg = lambda p : BETA * shannonEntropy(p)

    init = initDirectIRL(key,gridMDP)                   # init function
    grad = exactVanillaIRL(L,gridMDP,pFun,rFun,reg)      # gradient function 
    proc = irlClipProcessor(PLR,RLR,CLIP_THRESH)        # gradient processing
    proj = irlL2Proj(W_RADIUS)


    def policyReconstructionError(policy):
        # TODO: create a metrics module
        diff = policy-expertPolicy
        return jnp.sum(jla.norm(diff,ord=1,axis=1))

    """Defining the logger"""
    def logger( params, grads, step, i):
        return {
            'L'         : L(gridMDP,pFun(params['policy']),rFun(params['reward']),reg), 
            'PRE'       : policyReconstructionError(pFun(params['policy'])),
            'params'    : params,
            'grads'     : grads,
            'step'      : step,
            'iter'      : i
        }
            
    """Optimizing"""
    opt = Optimizer(init=init,grad=grad,proc=proc,log=logger,proj=proj)
    o, log = opt.train(key,STEPS,True)

    ls = [e['L'] for e in log]
    pre = [e['PRE'] for e in log]
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].plot(ls)
    ax[0].set_title('L')
    ax[1].plot(pre)
    ax[1].set_title('PRE')
    fig.tight_layout()
    plt.show()
    
    fig, ax = plt.subplots(2,3,figsize=(15,10))
    
    gridplot(gridMDP,ax[0,0],stochastic_policy=expertPolicy,goals=goals)
    ax[0,0].set_title('expert')
    gridplot(gridMDP,ax[0,1],stochastic_policy=pFun(init()['policy']),goals=goals)
    ax[0,1].set_title('IRL pre-training')
    gridplot(gridMDP,ax[0,2],stochastic_policy=pFun(o['policy']),goals=goals)
    ax[0,2].set_title('IRL post-training')
    
    ax[1,0].set_title('expert')
    gridplot(gridMDP,ax[1,0],scalar=jnp.sum(gridMDP.R,axis=1))
    ax[1,1].set_title('IRL pre-training')
    gridplot(gridMDP,ax[1,1],scalar=jnp.sum(rFun(log[0]['params']['reward']),axis=1))
    ax[1,2].set_title('IRL post-training')
    gridplot(gridMDP,ax[1,2],scalar=jnp.sum(rFun(log[-1]['params']['reward']),axis=1))
    
    fig.tight_layout()
    plt.show()
    
    
    
    print(log)
    
if __name__ == "__main__":
    main_irl()

# fig,ax = plt.subplots(1,1,figsize=(5,5))
# gridplot(gridMDP,ax,goals=goals,stochastic_policy=expertPolicy)
# plt.show()
