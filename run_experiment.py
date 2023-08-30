"""Imports"""
import jax
from jax import numpy as jnp
from jax.numpy import linalg as jla
import jax.nn as nn

from jax.config import config; config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
from itertools import accumulate

from env.gridworld import Gridworld, gridplot

from algs.policy_gradients import PolicyGradientMethod, vanillaGradOracle, naturalGradOracle, monteCarloVanillaGrad, slowMonteCarloVanillaGrad, Sampler, monteCarloNaturalGrad

def flatten(v):
    return jnp.reshape(v,(list(accumulate(v.shape,lambda x,y:x*y))[-1],))

key = jax.random.PRNGKey(0) 

"""Defining an MDP"""
R = 100
P = -300
goals = [((2,0),R),((1,0),P),((1,1),P)]
gridMDP = Gridworld(3,3,0.1,0.9,goals=goals,obstacles=[]) 
gridMDP.init_distrib =  jnp.exp(jax.random.uniform(key,(gridMDP.n,))) / \
    jnp.sum(jnp.exp(jax.random.uniform(key,(gridMDP.n,))))


parametrization = lambda p : nn.softmax(p,axis=1)

HORIZON = 10
BATCH = 30
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
                               BATCH,HORIZON)
                           ,logger)
log, theta = alg.train(40,5e-3)

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

fig, ax = plt.subplots(1,figsize=(10,10,))
gridplot(gridMDP,ax,stochastic_policy=pis[-1],goals=goals)
ax.set_title('Stochastic Vanilla PG solution')
fig.tight_layout()
plt.show()