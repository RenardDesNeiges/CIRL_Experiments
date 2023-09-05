import jax
from jax import numpy as jnp
from jax.numpy import linalg as jla
import jax.nn as nn
from jax.config import config; config.update("jax_enable_x64", True)

from typing import Callable

from env.mdp import MarkovDecisionProcess
from algs.utils import shannonEntropy
from algs.returns import J as Jfunction, IRLLagrangian
from algs.irl import initDirectIRL, exactNaturalIRL, irlClipProcessor, irlL2Proj
from algs.pg import initDirectPG, exactNaturalPG, pgClipProcessor
from algs.metrics import policyReconstructionError, normalizedRewardError
from algs.opt import Optimizer

# TODO : replace this with the PG trainer maybe
def getExpertPolicy(    key:jax.random.KeyArray,
                        mdp:MarkovDecisionProcess,
                        ret:Callable,
                        pFun:Callable,
                        rFun:Callable,
                        reg:Callable)->jnp.ndarray:
    STEPS=40
    LR = 5e-3
    CLIP_THRESH = 5e2
    init = initDirectPG(key,mdp)                    
    grad = exactNaturalPG(ret,mdp,pFun,rFun,reg)      
    proc = pgClipProcessor(LR,CLIP_THRESH)
    
    opt = Optimizer(init=init,grad=grad,proc=proc,log=lambda x,_g,_s,i : None,proj=lambda x: x)
    p, _ = opt.train(key,STEPS,True)
    return pFun(p['policy'])


class IRL_Trainer():
    def __init__(self,  mdp: MarkovDecisionProcess, 
                        policy_lr:float = 2e-3,
                        reward_lr:float = 1,
                        clip_thresh:float = 5e2,
                        w_radius:float = 1,
                        beta:float = 5,
                        pFun: Callable = lambda p : nn.softmax(p,axis=1),
                        rFun: Callable = lambda r : r,
                        reg: Callable = shannonEntropy,
                        ret:  Callable = Jfunction,
                        expertPolicy: jnp.ndarray=None,
                        key:jax.random.KeyArray=None, 
                        expertTrainer: Callable = getExpertPolicy,
                        ) -> None:

        if key is not None:
            self.key = key
        else:   
            self.key = jax.random.PRNGKey(0)
            
        self.mdp = mdp
        
        self.clip_tresh = clip_thresh
        self.beta = beta
        self.w_radius = w_radius
        self.policy_lr = policy_lr
        self.reward_lr = reward_lr
        
        self.ret = ret
        self.pFun = pFun
        self.rFun = rFun
        self.expertPolicy = expertPolicy
        self.reg = lambda p : - self.beta * reg(p)
        self.expertTrainer = expertTrainer
        

    def train(self, stepcount:int, pbar: bool = True):
        
        """Recovering the expert policy (that we are going recover to clone)"""
        print("Computing the expert's policy")
        if self.expertPolicy is None:
            assert self.expertTrainer is not None
            self.expertPolicy = getExpertPolicy(self.key,
                                                self.mdp,
                                                self.ret,
                                                self.pFun,
                                                self.rFun,
                                                self.reg)

        """Defining the optimizer functions"""
        L = IRLLagrangian(self.expertPolicy,self.ret)
        init = initDirectIRL(self.key,self.mdp)                 
        grad = exactNaturalIRL(L,self.mdp,
                               self.pFun,self.rFun,
                               self.reg)      
        proc = irlClipProcessor(self.policy_lr,
                                self.reward_lr,
                                self.clip_tresh)        
        proj = irlL2Proj(self.w_radius)

        """Defining the metrics functions"""
        pre = policyReconstructionError
        nre = normalizedRewardError

        """Defining the logger"""
        def logger( params, grads, step, i):
            return {
                'L'         : L(self.mdp,self.pFun(params['policy']),self.rFun(params['reward']),self.reg), 
                'PRE'       : pre(self.pFun(params['policy'])),
                'NRE'       : nre(self.rFun(params['reward'])),
                'params'    : params,
                'grads'     : grads,
                'step'      : step,
                'iter'      : i
            }
                    

        """Optimizing"""
        opt = Optimizer(init=init,grad=grad,proc=proc,log=logger,proj=proj)
        optimizers, trace = opt.train(self.key,stepcount,pbar)
        
        return optimizers, trace