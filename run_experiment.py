"""Imports"""
import jax
from jax import numpy as jnp
from jax.numpy import linalg as jla
import jax.nn as nn

from jax.config import config; config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
from itertools import accumulate

from env.gridworld import Gridworld

from algs.policy_gradients import PolicyGradientMethod, vanillaGradOracle, naturalGradOracle, monteCarloVanillaGrad, Sampler, monteCarloNaturalGrad

def flatten(v):
    return jnp.reshape(v,(list(accumulate(v.shape,lambda x,y:x*y))[-1],))

key = jax.random.PRNGKey(0) 

"""Defining an MDP"""
goals = [((1,1),100)]
gridMDP = Gridworld(2,2,0.1,0.9,goals=goals,obstacles=[]) # this is our MDP
gridMDP.init_distrib =  jnp.exp(jax.random.uniform(key,(gridMDP.n,))) / \
    jnp.sum(jnp.exp(jax.random.uniform(key,(gridMDP.n,)))) # setting its init distribution


parametrization = lambda p : nn.softmax(p,axis=1)

HORIZON = 5
BATCH = 10
sampler = Sampler(gridMDP,key)

""" Picking the gradient evaluation method 
    (either through an orcale or through sampling)
    (either preconditionned with the inverse FIM or not)
"""

"""Defining the logger"""
def logger(theta):
        return {
            'theta': theta,
            'pi': nn.softmax(theta,axis=1),
            'J': gridMDP.J(nn.softmax(theta,axis=1))
        }


"""Trainig"""
alg = PolicyGradientMethod(gridMDP,key,
                           monteCarloNaturalGrad(
                               gridMDP,
                               sampler,
                               key,
                               parametrization,
                               HORIZON,BATCH)
                           ,logger)
log, theta = alg.train(20,4e-2)

"""Plotting the training curve"""
thetas=jnp.stack([e['theta'] for e in log])
pis=jnp.stack([e['pi'] for e in log])
js=jnp.stack([e['J'] for e in log])
fig, ax = plt.subplots(1,2,figsize=(10,5))
ax[0].plot(js)
ax[0].set_title('Objective function (maximizing)')
ax[1].plot(pis[:,0,:2])
ax[1].set_title('Two example policy coordinates')
plt.show()
