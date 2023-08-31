import jax
from jax import numpy as jnp
from jax.numpy import linalg as jla
import jax.nn as nn

from typing import Callable, Any, Dict, List, Tuple

from env.mdp import MarkovDecisionProcess
from env.sample import Sampler
from algs.utils import flatten

CLIP_THRESH = 1e3

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
        grad = grad
        return grad(p['policy'])
    
    return jax.jit(grad_function)

def exact_fim_oracle(mdp: MarkovDecisionProcess,pFun : Callable, _p:jnp.array)->jnp.ndarray:
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
        f_inv = jla.pinv(exact_fim_oracle(mdp,pFun, p['policy']))
        g = grad(p['policy'])
        return jnp.reshape(f_inv@flatten(g),_shape)
    
    return jax.jit(grad_function)

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
                        r_batch[:, :, jnp.newaxis, jnp.newaxis],    # reward along the right 
                        summed_grads.shape[2], axis=2),             # axes so we can elementwise 
                        summed_grads.shape[3],axis=3)               # multiply with the gradients

        gradient_tensor = summed_grads[:,:-1,:,:] \
                        * reward_tensor[:,1:,:,:] \
                        * gamma_tensor[:,1:,:,:] 

        return (1/smp.b)*jnp.sum(gradient_tensor,axis=(0,1))
    return jax.jit(grad)

# def gpomdp(batch,theta,B,H,gamma,parametrization,reg=None):
#     #TODO implement regularizer
#     def g_log(theta,s,a):
#         return jax.grad(lambda p : jnp.log(parametrization(p))[s,a])(theta)
#     def trace_grad(batch,theta,b,h):
#         return jnp.sum(jnp.array([g_log(theta,e[0],e[1]) for e in batch[b][:h]]),axis=0)
#     def single_sample_gpomdp(batch,theta,b,H):
#         return jnp.sum(jnp.array([(gamma**h)*batch[b][h][2] \
#                     *trace_grad(batch,theta,b,h) for h in range(1,H)]),axis=0)
        
#     return (1/B)*jnp.sum(jnp.array([single_sample_gpomdp(batch,theta,b,H) \
#                             for b in range(B)]),axis=0)




def computeSlowMonteCarloVanillaGrad(theta,mdp,sampler,key,parametrization,B,H):
    #TODO implement regularizer
    batch = sample_batch(parametrization(theta),mdp,sampler,H,B,key)
    return gpomdp(batch,theta,B,H,mdp.gamma,parametrization)

def computeMonteCarloVanillaGrad(theta,mdp,sampler,key,parametrization,B,H,regularizer):
    batch = sample_batch(parametrization(theta),mdp,sampler,H,B,key,regularizer)
    return fast_gpomdp(batch,theta,B,H,mdp.gamma,parametrization)

def slowMonteCarloVanillaGrad(mdp,sampler,key,parametrization,B,H):
    #TODO implement regularizer
    return lambda p : computeSlowMonteCarloVanillaGrad(p,mdp,sampler,key,parametrization,B,H)

def monteCarloVanillaGrad(mdp,sampler,key,parametrization,B,H,regularizer=None):
    return lambda p : computeMonteCarloVanillaGrad(p,mdp,sampler,key,parametrization,B,H,regularizer)

"""Stochastic natural gradients and FIM estimation"""

def fim_sample(sa,theta):
    pg = flatten(jax.grad(lambda p : jnp.log(nn.softmax(p,axis=1))[sa[0],sa[1]])(theta))
    return jnp.outer(pg,pg)
def estimate_fim(batch,theta):
    sa_pairs = jnp.array([flatten(jnp.array([[step[0] for step in trace] for trace in batch])), flatten(jnp.array([[step[1] for step in trace] for trace in batch]))])
    return jnp.sum(jnp.array([fim_sample(sa_pairs[:,i],theta) for i in range(sa_pairs.shape[1])]),axis=0)/sa_pairs.shape[1]

def computeMonteCarloNaturalGrad(theta,mdp,sampler,key,parametrization,B,H,regularizer):
    batch = sample_batch(parametrization(theta),mdp,sampler,H,B,key,regularizer)
    _g = fast_gpomdp(batch,theta,B,H,mdp.gamma,parametrization)
    _shape = _g.shape
    fim = estimate_fim(batch,theta)
    return jnp.reshape(jla.pinv(fim)@flatten(_g),_shape) # TODO : replace with conjugate grads

def monteCarloNaturalGrad(mdp,sampler,key,parametrization,B,H,regularizer):
    return lambda p : computeMonteCarloNaturalGrad(p,mdp,sampler,key,parametrization,B,H,regularizer)
