from jax import numpy as jnp
from itertools import accumulate

""" Useful helper functions """
def flatten(v):
    return jnp.reshape(v,(list(accumulate(v.shape,lambda x,y:x*y))[-1],))

shannonEntropy = lambda p : -jnp.dot(jnp.log(p),p)