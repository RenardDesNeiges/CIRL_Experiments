import jax
from jax import numpy as jnp
from jax.numpy import linalg as jla
import jax.nn as nn

from typing import Callable, Any, Dict, List, Tuple

from env.mdp import MarkovDecisionProcess
from env.sample import Sampler
from algs.utils import flatten

"""Exact gradient oracles
--> all gradient functions have the prototype 
f: (batch [Array],  parameters [Dict of jnp.arrays]) -> grad [jnp.array]

BUUUUT, the functions here are generators that return functions with that prototype, so they themselves have a different prototype
"""

def vanillaGradOracle(  J:Callable,
                        mdp:MarkovDecisionProcess,
                        pFun:Callable,
                        rFun:Callable,
                        reg:Callable
                        )->Callable[[Dict[str,jnp.ndarray]],jnp.ndarray]:
    """ Generates a (vanilla) policy gradient oracle function for some mdp,
        policy parametrization and reward function, supports regularized
        functions.

    Args:
        J (Callable): Return function.
        mdp (MarkovDecisionProcess): MDP to compute the grad on.
        pFun (Callable): Policy parametrization.
        rFun (Callable): Reward parametetrization.
        reg (Callable): regularizer.

    Returns:
        Callable[[Dict[str,jnp.ndarray]],jnp.ndarray]: the gradient function.
    """
    def grad_function(p):
        reward = rFun(p['reward'])
        grad = jax.grad(
            lambda theta: J(mdp,pFun(theta),reward,reg)
            )
        return grad(p['policy'])
    
    return jax.jit(grad_function)

def exactFIMOracle(mdp: MarkovDecisionProcess,pFun : Callable, _p:jnp.array)->jnp.ndarray:
    """Computes the exact FIM for some policy parameters and parameterization on some MDP.

    Args:
        mdp (MarkovDecisionProcess): the mdp.
        pFun (Callable): the policy parametrization.
        _p (jnp.array): the policy parameters.
        
    Returns:
        jnp.ndarray: the FIM.
    """
    v = jax.jacfwd(lambda p : flatten(jnp.log(pFun(p))))(_p) # computing the jacobian (step 1)
    jac = jnp.reshape(v,(mdp.n*mdp.m,mdp.n*mdp.m)) # flatten last two dimensions (step 1)
    bop = jnp.einsum('bi,bj->bji',jac,jac) # batch outer-product (step 2)
    return jnp.einsum('bij,b->ij',bop,
                        flatten(mdp.occ_measure(pFun(_p)))) # fisher information matrix (I hope) (step 3)


def naturalGradOracle(  J:Callable,
                        mdp:MarkovDecisionProcess,
                        pFun:Callable,
                        rFun:Callable,
                        reg:Callable
                        )->Callable[[List[Any],Dict[str,jnp.ndarray]],jnp.ndarray]:
    """ Generates a natural policy gradient oracle function for some mdp,
        policy parametrization and reward function, supports regularized
        functions.

    Args:
        J (Callable): Return function.
        mdp (MarkovDecisionProcess): MDP to compute the grad on.
        pFun (Callable): Policy parametrization.
        rFun (Callable): Reward parametetrization.
        reg (Callable): regularizer.

    Returns:
        Callable[[List[Any],Dict[str,jnp.ndarray]],jnp.ndarray]: the gradient function.
    """
    def grad_function(p):
        _shape = p['policy'].shape
        reward = rFun(p['reward'])
        grad = jax.grad(
            lambda theta: J(mdp,pFun(theta),reward,reg)
            )
        f_inv = jla.pinv(exactFIMOracle(mdp,pFun, p['policy']))
        g = grad(p['policy'])
        return jnp.reshape(f_inv@flatten(g),_shape)
    
    return jax.jit(grad_function)


def rewardGradOracle(   J:Callable,
                        mdp:MarkovDecisionProcess,
                        pFun:Callable,
                        rFun:Callable,
                        reg:Callable
                        )->Callable[[Dict[str,jnp.ndarray]],jnp.ndarray]:
    """ Generates a reward gradient oracle function for some mdp,
        policy parametrization and reward function, supports regularized
        functions.

    Args:
        J (Callable): Return function.
        mdp (MarkovDecisionProcess): MDP to compute the grad on.
        pFun (Callable): Policy parametrization.
        rFun (Callable): Reward parametetrization.
        reg (Callable): regularizer.

    Returns:
        Callable[[Dict[str,jnp.ndarray]],jnp.ndarray]: the gradient function.
    """
    def grad_function(p):
        policy = pFun(p['policy'])
        return jax.grad(lambda w : J(mdp,policy,rFun(w),reg))(p['reward'])
    
    return jax.jit(grad_function)


def monteCarloRewardGrad(   J:Callable,
                            mdp:MarkovDecisionProcess,
                            pFun:Callable,
                            rFun:Callable,
                            reg:Callable,
                            smp:Callable,
                            expertPolicy: jnp.ndarray,
                            )->Callable[[Dict[str,jnp.ndarray]],jnp.ndarray]:
    def rSA(w):
        return lambda s,a : rFun(w)[s,a]
    def batchReturn(batch_s,batch_a):
        gammas = jnp.repeat(jnp.expand_dims( mdp.gamma**jnp.arange(batch_s.shape[1]),
                                    0),batch_s.shape[0],0)
        return lambda w : jnp.sum(gammas * jax.vmap(rSA(w))(batch_s,batch_a))\
            /batch_s.shape[0]
    def batchGrad(batch_s,batch_a):
        return lambda w  : jax.grad(batchReturn(batch_s,batch_a))(w)
    
    # 1. Sample a reference expert dataset  
    # Implementation decision ! For now the batch size of the expert dataset is the same as the gradient sampling, maybe find a way of having an expert specific sampler later
    expert_s, expert_a, _ = smp.batch(expertPolicy,regularizer=reg)
    # 2. Compute its feature expectation function and store it
    expert_features = batchGrad(expert_s,expert_a) # this is a function of weights w
    
    # 3. Return a gradient function that 
    #       a. takes as input a batch and weights
    #       b. computes the learned policy-reward feature expectation
    #       c. recovers the gradients
    
    def grad(batch,p):
        policy_s, policy_a, _ = batch
        policy_features = batchGrad(policy_s,policy_a)
        return policy_features(p['reward']) - expert_features(p['reward'])
    
    return grad



"""Stochastic gradients estimators
--> all gradient functions have the prototype 
f: (batch [Array],  parameters [Dict of jnp.arrays]) -> grad [jnp.array]

Here the batch entry is actually used as these are stochastic estimators that require data to get a correct estimation.

"""

def gpomdp( mdp:MarkovDecisionProcess,
            pFun:Callable,
            smp:Sampler
            )->Callable[[Tuple[jnp.ndarray,jnp.ndarray,jnp.ndarray],
                         Dict[str,jnp.ndarray]],jnp.ndarray]:
    """ Generates a GPOMDP policy gradient estimator function for some mdp,
        policy parametrization using some sampler.

    Args:
        mdp (MarkovDecisionProcess): the MDP.
        pFun (Callable): the policy parametrization.
        smp (Sampler): the sampler.

    Returns:
        Callable[[Tuple[jnp.ndarray,jnp.ndarray,jnp.ndarray], Dict[str,jnp.ndarray]],jnp.ndarray]: the gradient evaluation function (which takes in a sampled batch as well as the parameters and returns the gradients).
    """
    def grad(batch,p):
        def g_log(s,a,theta,pFun):
            return jax.grad(lambda p : jnp.log(pFun(p))[s,a])(theta)
        _f = lambda s,a : g_log(s,a,p['policy'],pFun)
    
        s_batch, a_batch, r_batch = batch
        batch_grads = jax.vmap(jax.vmap(_f))(s_batch, a_batch) 
        summed_grads = jnp.cumsum(batch_grads,axis=1) 
        gamma_tensor = mdp.gamma**jnp.arange(smp.h) 
        gamma_tensor = jnp.repeat(jnp.repeat(jnp.repeat(
                    gamma_tensor[jnp.newaxis, :, jnp.newaxis, jnp.newaxis], 
                    smp.b, axis=0), summed_grads.shape[2],axis=2), 
                    summed_grads.shape[3],axis=3)
        reward_tensor = jnp.repeat(jnp.repeat(                      # here we repeat the 
                        r_batch[:, :, jnp.newaxis, jnp.newaxis],    # reward along the axes (2,3)
                        summed_grads.shape[2], axis=2),             # so we can elementwise 
                        summed_grads.shape[3],axis=3)               # multiply with the gradients
    
        gradient_tensor = summed_grads \
                        * reward_tensor \
                        * gamma_tensor

        return (1/smp.b)*jnp.sum(gradient_tensor,axis=(0,1))
    return jax.jit(grad)

"""Stochastic natural gradients and FIM estimation"""


def fimEstimator(mdp,pFun,reg_param=0):
    
    def fim(batch,p):
        pi = pFun(p['policy'])
        s_batch, _, _ = batch
        ps = jnp.bincount(flatten(s_batch))/flatten(s_batch).shape[0]
        psa = pi * jnp.repeat(jnp.expand_dims(ps,1), pi.shape[1],axis=1)
        sid = jnp.repeat(jnp.expand_dims(jnp.arange(mdp.n),1),mdp.m,axis=1)
        aid = jnp.repeat(jnp.expand_dims(jnp.arange(mdp.m),0),mdp.n,axis=0)

        def fim_sample(s,a):
            pg = flatten(jax.grad(lambda p : jnp.log(pFun(p))[s,a])(p['policy']))
            return jnp.outer(pg,pg)

        def element_op(p,s,a):
            return p*fim_sample(s,a)

        F = jnp.sum(jax.vmap(element_op)(flatten(psa),flatten(sid),flatten(aid)),axis=0)

        return F + jnp.eye(F.shape[0])*reg_param

    return fim

