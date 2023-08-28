import jax
from jax import numpy as jnp
from jax.numpy import linalg as jla
import jax.nn as nn

from itertools import accumulate
from tqdm import tqdm

from env.mdp import MarkovDecisionProcess, Sampler

""" Useful helper functions """
def flatten(v):
    return jnp.reshape(v,(list(accumulate(v.shape,lambda x,y:x*y))[-1],))

CLIP_THRESH = 1e3

class PolicyGradientMethod():
    def __init__(self, MDP, key, gradient_sampler, logger, progress_bar=True, clip_thresh=CLIP_THRESH) -> None:
        self.MDP : MarkovDecisionProcess = MDP
        self.key = key
        self.gradient_sampler = gradient_sampler
        self.logger = logger
        self.progress_bar = progress_bar
        self.clip_thresh = clip_thresh
        
    def train(self, nb_steps, lr ):
        theta = jax.random.uniform(self.key,(self.MDP.n,self.MDP.m))
        log = [self.logger(theta)]
        for _ in tqdm(range(nb_steps),disable=not self.progress_bar):
            _g = jnp.clip(self.gradient_sampler(theta),
                                   a_min=-self.clip_thresh,
                                   a_max=self.clip_thresh)
            _g = jax.numpy.nan_to_num(_g, copy=False, nan=0.0)
            theta += lr * _g
            log += [self.logger(theta)]
        return log, theta

""" Generic sampling routines for MDPs """
def sample_trajectory(pi,mdp,smp,H,key):
    def pick_action(pi,s,mdp):
        p = pi[s]; p /= jnp.sum(p)
        return jax.random.choice(key,jnp.arange(mdp.m), p = p)
        
    traj = []
    r_t = 0
    s_t = smp.reset()
    for _ in range(H):
        a_t = pick_action(pi,s_t,mdp)
        traj += [(s_t,a_t,r_t)]
        s_t, r_t = smp.step(a_t)
    return traj
def sample_batch(pi,mdp,smp,H,B,key):
    return [sample_trajectory(pi,mdp,smp,H,key) for _ in range(B)]

"""Exact gradient oracles"""

def vanillaGradOracle(mdp,sampler,key,parametrization,B,H):
    return jax.grad(lambda p : mdp.J(parametrization(p)))

def naturalGradOracle(mdp,sampler,key,parametrization,B,H):
    def naturalGrad(theta):
        _shape = theta.shape
        g = jax.grad(lambda p : mdp.J(parametrization(p)))(theta)    
        f_inv = jla.pinv(mdp.exact_fim_oracle(theta,lambda p:nn.softmax(p,axis=1)))
        return jnp.reshape(f_inv@flatten(g),_shape)
    return jax.jit(naturalGrad)

"""Stochastic gradients"""

def gpomdp(batch,theta,B,H,gamma):
    def g_log(theta,s,a):
        return jax.grad(lambda p : jnp.log(nn.softmax(p,axis=1))[s,a])(theta)
    def trace_grad(batch,theta,b,h):
        return jnp.sum(jnp.array([g_log(theta,e[0],e[1]) for e in batch[b][:h]]),axis=0)
    def single_sample_gpomdp(batch,theta,b,H):
        return jnp.sum(jnp.array([(gamma**h)*batch[b][h][2] \
                    *trace_grad(batch,theta,b,h) for h in range(1,H)]),axis=0)
        
    return (1/B)*jnp.sum(jnp.array([single_sample_gpomdp(batch,theta,b,H) for b in range(B)]),axis=0)


def fast_gpodmp(batch,theta,B,H,gamma,parametrization):
    s_batch = jnp.array([[e[0] for e in s] for s in batch])
    a_batch = jnp.array([[e[1] for e in s] for s in batch])
    r_batch = jnp.array([[e[2] for e in s] for s in batch])

    def g_log(s,a,theta,parametrization):
        return jax.grad(lambda p : jnp.log(parametrization(p))[s,a])(theta)
    _f = lambda s,a : g_log(s,a,theta,parametrization)


    batch_grads = jax.vmap(jax.vmap(_f))(s_batch, a_batch) ##vmap can only operate on a single axis 
    summed_grads = jnp.cumsum(batch_grads,axis=0)

    gamma_tensor = gamma**jnp.arange(H)
    gamma_tensor = jnp.repeat(jnp.repeat(jnp.repeat(
        gamma_tensor[:, jnp.newaxis, jnp.newaxis, jnp.newaxis], 
            B, axis=1),
                summed_grads.shape[2],axis=2),
                    summed_grads.shape[3],axis=3)

    reward_grads = summed_grads*jnp.repeat(jnp.repeat(
        r_batch[:, :, jnp.newaxis, jnp.newaxis], 
        summed_grads.shape[2], axis=2),
            summed_grads.shape[3],axis=3) * gamma_tensor
    return jnp.sum(reward_grads,axis=(0,1))

def computeMonteCarloVanillaGrad(theta,mdp,sampler,key,parametrization,B,H):
    batch = sample_batch(parametrization(theta),mdp,sampler,H,B,key)
    return fast_gpodmp(batch,theta,B,H,mdp.gamma,parametrization)

def monteCarloVanillaGrad(mdp,sampler,key,parametrization,B,H):
    return lambda p : computeMonteCarloVanillaGrad(p,mdp,sampler,key,parametrization,B,H)

"""Stochastic natural gradients and FIM estimation"""

def fim_sample(sa,theta):
    pg = flatten(jax.grad(lambda p : jnp.log(nn.softmax(p,axis=1))[sa[0],sa[1]])(theta))
    return jnp.outer(pg,pg)
def estimate_fim(batch,theta):
    sa_pairs = jnp.array([flatten(jnp.array([[step[0] for step in trace] for trace in batch])), flatten(jnp.array([[step[1] for step in trace] for trace in batch]))])
    return jnp.sum(jnp.array([fim_sample(sa_pairs[:,i],theta) for i in range(sa_pairs.shape[1])]),axis=0)/sa_pairs.shape[1]


def truncate_parameters(theta):
    return 

def computeMonteCarloNaturalGrad(theta,mdp,sampler,key,parametrization,B,H):
    batch = sample_batch(parametrization(theta),mdp,sampler,H,B,key)
    _g = gpomdp(batch,theta,B,H,mdp.gamma)
    _shape = _g.shape
    fim = estimate_fim(batch,theta)
    return jnp.reshape(jla.pinv(fim)@flatten(_g),_shape) # TODO : replace with conjugate grads

def monteCarloNaturalGrad(mdp,sampler,key,parametrization,B,H):
    return lambda p : computeMonteCarloNaturalGrad(p,mdp,sampler,key,parametrization,B,H)