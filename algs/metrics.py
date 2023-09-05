from jax import numpy as jnp
from jax.numpy import linalg as jla
from jax.config import config; config.update("jax_enable_x64", True)

def policyReconstructionError(expertPolicy):
    def pre(policy):
        diff = policy-expertPolicy
        return jnp.sum(jla.norm(diff,ord=1,axis=1))
    return pre

def normalizedRewardError(expertReward):
    def nre(reward):
        normalize = lambda x : x/jla.norm(x)
        diff = normalize(reward)-normalize(expertReward)
        return jnp.sum(jla.norm(diff,ord=1,axis=1))
    return nre

            