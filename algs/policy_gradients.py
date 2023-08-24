import jax
from jax import numpy as jnp
from jax.numpy import linalg as jla
from einops import rearrange, repeat
from itertools import accumulate


""" Useful helper functions """
def flatten(v):
    return jnp.reshape(v,(list(accumulate(v.shape,lambda x,y:x*y))[-1],))

def softmax_pi(theta):
    return jnp.exp(theta)/jnp.repeat(jnp.expand_dims(jnp.sum(jnp.exp(theta),1),1),4,1)

def closed_loop_kernel(mdp,pi):
    return jnp.einsum('sap,sa->sp',mdp.P_sa,pi) # closed loop transition kernel
def state_occ_measure(mdp,P_pi):
    return (1-mdp.gamma)*(jla.inv((jnp.eye(mdp.n)-mdp.gamma*P_pi)).transpose()@mdp.init_distrib)
def occ_measure(mdp,pi):
    mu_s = state_occ_measure(mdp,closed_loop_kernel(mdp,pi))
    return pi * repeat(mu_s, 's -> s new_axis', new_axis=mdp.m)
def J_unregularized(mdp,theta):
    return jnp.dot(flatten(mdp.R),flatten(occ_measure(mdp,softmax_pi(theta))))

def exact_fim(theta,mdp):
    v = jax.jacfwd(lambda p : flatten(jnp.log(softmax_pi(p))))(theta) # computing the jacobian (step 1)
    jac = jnp.reshape(v,(mdp.n*mdp.m,mdp.n*mdp.m)) # flatten last two dimensions (step 1)
    bop = jnp.einsum('bi,bj->bji',jac,jac) # batch outer-product (step 2)
    return jnp.einsum('bij,b->ij',bop,flatten(occ_measure(mdp,softmax_pi(theta)))) # fisher information matrix (I hope) (step 3)aa