import jax
from jax import numpy as jnp
from jax.numpy import linalg as jla
import jax.nn as nn
from jax.config import config; config.update("jax_enable_x64", True)

from typing import Callable, Any, Dict, List, Tuple

import matplotlib.pyplot as plt

from env.mdp import MarkovDecisionProcess
from env.gridworld import gridplot
from algs.utils import shannonEntropy
from algs.returns import J as Jfunction, IRLLagrangian
from algs.irl import initDirectIRL, exactNaturalIRL, irlClipProcessor, irlL2Proj
from algs.pg import initDirectPG, exactNaturalPG, pgClipProcessor
from algs.metrics import policyReconstructionError, normalizedRewardError
from algs.opt import Optimizer

from abc import ABC, abstractmethod

# TODO : replace this with the PG trainer maybe
def getExpertPolicy(    key:jax.random.KeyArray,
                        mdp:MarkovDecisionProcess,
                        ret:Callable,
                        pFun:Callable,
                        rFun:Callable,
                        reg:Callable)->jnp.ndarray:
    STEPS=60
    LR = 1
    CLIP_THRESH = 5e2
    init = initDirectPG(key,mdp)                    
    grad = exactNaturalPG(ret,mdp,pFun,rFun,reg)      
    proc = pgClipProcessor(LR,CLIP_THRESH)
    
    opt = Optimizer(init=init,grad=grad,proc=proc,log=lambda x,_g,_s,i : None,proj=lambda x: x)
    p, _ = opt.train(key,STEPS,True)
    return pFun(p['policy'])

class PG_Trainer():
    def __init__(self,  mdp: MarkovDecisionProcess, 
                        policy_lr:float = 2e-3,
                        clip_thresh:float = 5e2,
                        beta:float = 0.1,
                        pFun: Callable = lambda p : nn.softmax(p,axis=1),
                        rFun: Callable = lambda w : w,
                        reg: Callable = shannonEntropy,
                        ret:  Callable = Jfunction,
                        key:jax.random.KeyArray=None, 
                        init_params: Callable = initDirectPG,
                        gradients: Callable = exactNaturalPG,
                        grad_proc: Callable = pgClipProcessor,
                        proj: Callable = lambda x : x,
                        ) -> None:
        """Trainer class for policy gradient problems.
        
        TODO : write the docstring
        """        
        
        """Randomness"""
        if key is not None:
            self.key = key
        else:   
            self.key = jax.random.PRNGKey(0)
            
        """MDP"""
        self.mdp = mdp
        
        """Hyperparameters"""
        self.clip_tresh = clip_thresh
        self.beta = beta
        self.policy_lr = policy_lr
        
        """Problem specific functions"""
        self.ret = ret
        self.pFun = pFun
        self.rFun = rFun
        self.reg = lambda p : - self.beta * reg(p)
        
        """Optimizer required functions"""
        self.init_params = init_params
        self.gradients = gradients
        self.grad_proc = grad_proc
        self.proj = proj

    def train(self, stepcount:int, pbar: bool = True)->Tuple[Dict[str,Any],List[Dict[str,Any]]]:
        """Training method.

        Args:
            stepcount (int): number of training steps
            pbar (bool, optional): if true, uses a tqdm progress bar. Defaults to True.

        Returns:
            Tuple[Dict[str,Any],List[Dict[str,Any]]]: The optimized variables (in a dictionary) and the training traces (the logs).
        """

        """Defining the optimizer functions"""
        init = self.init_params(self.key,self.mdp)                 
        grad = self.gradients(  self.ret,self.mdp,
                                self.pFun,self.rFun,
                                self.reg)      
        proc = self.grad_proc(  self.policy_lr,
                                self.clip_tresh)        
        proj = self.proj

        """Defining the logger"""
        def logger( params, grads, step, i):
            return {
                'J'         :    self.ret(self.mdp,
                                self.pFun(params['policy']),
                                self.rFun(params['reward']),
                                self.reg), 
                'params'    : params,
                'grads'     : grads,
                'step'      : step,
                'iter'      : i
            }
                    

        """Optimizing"""
        opt = Optimizer(init=init,grad=grad,proc=proc,log=logger,proj=proj)
        optimizers, trace = opt.train(self.key,stepcount,pbar)
        
        return optimizers, trace

class IRL_Trainer():
    def __init__(self,  mdp: MarkovDecisionProcess, 
                        policy_lr:float = 2e-3,
                        reward_lr:float = 1,
                        clip_thresh:float = 5e2,
                        w_radius:float = 1,
                        beta:float = 0.1,
                        pFun: Callable = lambda p : nn.softmax(p,axis=1),
                        rFun: Callable = lambda r : r,
                        reg: Callable = shannonEntropy,
                        ret:  Callable = Jfunction,
                        expertPolicy: jnp.ndarray=None,
                        key:jax.random.KeyArray=None, 
                        expertTrainer: Callable = getExpertPolicy,
                        lagrangian: Callable = IRLLagrangian,
                        init_params: Callable = initDirectIRL,
                        gradients: Callable = exactNaturalIRL,
                        grad_proc: Callable = irlClipProcessor,
                        proj: Callable = irlL2Proj,
                        ) -> None:
        """Trainer class for inverse reinforcement learning problems.

        Args:
            mdp (MarkovDecisionProcess): mdp to optimize on.
            policy_lr (float, optional): policy learning rate. Defaults to 2e-3.
            reward_lr (float, optional): reward learning rate. Defaults to 1.
            clip_thresh (float, optional): reward gradient clipping threshold. Defaults to 5e2.
            w_radius (float, optional): reward class radius. Defaults to 1.
            beta (float, optional): regularization factor. Defaults to 5.
            pFun (_type_, optional): policy parameterization. Defaults to lambdap:nn.softmax(p,axis=1).
            rFun (_type_, optional): reward parameterization. Defaults to lambdar:r.
            reg (Callable, optional): regularizer function. Defaults to shannonEntropy.
            ret (Callable, optional): return function. Defaults to J.
            expertPolicy (jnp.ndarray, optional): expert policy. Defaults to None (and if none trains its own expert policy).
            key (jax.random.KeyArray, optional): pseudorandom key. Defaults to None.
            expertTrainer (Callable, optional): expert trainig function. Defaults to getExpertPolicy.
            lagrangian (Callable, optional): lagrangian function. Defaults to IRLLagrangian.
            init_params (Callable, optional): parameter initialization function. Defaults to initDirectIRL.
            gradients (Callable, optional): gradient computation function. Defaults to exactNaturalIRL.
            grad_proc (Callable, optional): gradient processing fucntion. Defaults to irlClipProcessor.
            proj (Callable, optional): projection. Defaults to irlL2Proj.
        """        
        
        """Randomness"""
        if key is not None:
            self.key = key
        else:   
            self.key = jax.random.PRNGKey(0)
            
        """MDP"""
        self.mdp = mdp
        
        """Hyperparameters"""
        self.clip_tresh = clip_thresh
        self.beta = beta
        self.w_radius = w_radius
        self.policy_lr = policy_lr
        self.reward_lr = reward_lr
        
        """Problem specific functions"""
        self.ret = ret
        self.pFun = pFun
        self.rFun = rFun
        self.reg = lambda p : - self.beta * reg(p)
        
        """Expert policy"""
        self.expertPolicy = expertPolicy
        self.expertTrainer = expertTrainer
        
        """Optimizer required functions"""
        self.lagrangian = lagrangian
        self.init_params = init_params
        self.gradients = gradients
        self.grad_proc = grad_proc
        self.proj = proj

    def train(self, stepcount:int, pbar: bool = True)->Tuple[Dict[str,Any],List[Dict[str,Any]]]:
        """Training method.

        Args:
            stepcount (int): number of training steps
            pbar (bool, optional): if true, uses a tqdm progress bar. Defaults to True.

        Returns:
            Tuple[Dict[str,Any],List[Dict[str,Any]]]: The optimized variables (in a dictionary) and the training traces (the logs).
        """        

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
        L = self.lagrangian(self.expertPolicy,self.ret)
        init = self.init_params(self.key,self.mdp)                 
        grad = self.gradients(L,self.mdp,
                                self.pFun,self.rFun,
                                self.reg)      
        proc = self.grad_proc(  self.policy_lr,
                                self.reward_lr,
                                self.clip_tresh)        
        proj = self.proj(self.w_radius)

        """Defining the metrics functions"""
        pre = policyReconstructionError(self.expertPolicy)
        nre = normalizedRewardError(self.mdp.R)

        """Defining the logger"""
        def logger( params, grads, step, i):
            return {
                'L'         : L(self.mdp,
                                self.pFun(params['policy']),
                                self.rFun(params['reward']),
                                self.reg), 
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
    
class TracePlotter(ABC):
    @abstractmethod
    def plotScalar(ax,key,trace,title=None):
        """Plots a scalar from a training trace.
        """        
        iters = [e['iter'] for e in trace]
        ls = [e[key] for e in trace]
        ax.plot(iters,ls)
        if title is None:
            ax.set_title(key)
        else:
            ax.set_title(title)

    @abstractmethod
    def plotGradNorms(ax,key,trace,title=None):
        """Plots gradient norms.
        """
        iters = [e['iter'] for e in trace]
        gnorm = [jla.norm(e['grads'][key]) for e in trace]
        ax.plot(iters,gnorm)
        if title is None:
            ax.set_title(f'{key} gradient norms')
        else:
            ax.set_title(title)
            
    @abstractmethod
    def gworldIRLPlot(  trainer, 
                        opts:Dict[str,Any], 
                        trace: List[Dict[str,Any]], 
                        fsize:Tuple[str,str]=(15,10)):
        
        fig, ax = plt.subplots(2,3,figsize=fsize)
        gridplot(trainer.mdp,ax[0,0],stochastic_policy=trainer.expertPolicy
        ,goals=trainer.mdp.goals); ax[0,0].set_title('Expert Policy')
        gridplot(trainer.mdp,ax[0,1],stochastic_policy=
                trainer.pFun(trace[0]['params']['policy'])
                ,goals=trainer.mdp.goals); ax[0,1].set_title('Pre-training Policy')
        gridplot(trainer.mdp,ax[0,2],stochastic_policy=
                trainer.pFun(trace[-1]['params']['policy'])
                ,goals=trainer.mdp.goals); ax[0,2].set_title('Learned IRL Policy')
        gridplot(   trainer.mdp,ax[1,0],
                    scalar=jnp.sum(trainer.mdp.R,axis=1),
                    goals=trainer.mdp.goals); ax[1,0].set_title('Expert reward')
        gridplot(trainer.mdp,ax[1,1],
                scalar=jnp.sum(trainer.rFun(trace[0]['params']['reward']),axis=1),
                goals=trainer.mdp.goals); ax[1,1].set_title('Pre training reward')
        gridplot(trainer.mdp,ax[1,2],
                scalar=jnp.sum(trainer.rFun(trace[-1]['params']['reward']),axis=1),
                goals=trainer.mdp.goals); ax[1,2].set_title('Learned reward')

        fig.tight_layout()
        return fig, ax  