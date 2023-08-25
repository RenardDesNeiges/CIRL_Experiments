"""Imports"""
import jax
from jax import numpy as jnp
from jax.numpy import linalg as jla
import jax.nn as nn

import matplotlib.pyplot as plt
from itertools import accumulate

from env.mdp import MarkovDecisionProcess,Sampler
from env.gridworld import Gridworld, gridplot

def flatten(v):
    return jnp.reshape(v,(list(accumulate(v.shape,lambda x,y:x*y))[-1],))

key = jax.random.PRNGKey(0) 

"""Defining an MDP"""
goals = [((2,2),100)]
gridMDP = Gridworld(3,3,0.1,0.9,goals=goals,obstacles=[]) # this is our MDP
gridMDP.init_distrib =  jnp.exp(jax.random.uniform(key,(gridMDP.n,))) / \
    jnp.sum(jnp.exp(jax.random.uniform(key,(gridMDP.n,)))) # setting its init distribution
    
    
"""Plotting the MDP"""

# fig, ax = plt.subplots(1,figsize=(10,10))
# gridplot(gridMDP,ax,goals=goals)
# ax.set_title('Gridworld reward structure')
# fig.tight_layout()
# plt.show()

vanillaGrad = jax.grad(lambda p : gridMDP.J(nn.softmax(p,axis=1)))
def naturalGrad(theta):
    _shape = theta.shape
    g = jax.grad(lambda p : gridMDP.J(nn.softmax(p,axis=1)))(theta)    
    f_inv = jla.pinv(gridMDP.exact_fim_oracle(theta,lambda p:nn.softmax(p,axis=1)))
    return jnp.reshape(f_inv@flatten(g),_shape)


def PolicyGradient(_g,mdp,eta=20, nb_steps = 10):
    def logger(theta):
        return {
            'pi': nn.softmax(theta,axis=1),
            'J': mdp.J(nn.softmax(theta,axis=1))
        }
    theta = jax.random.uniform(key,(mdp.n,mdp.m))
    log = [logger(theta)]
    for _ in range(nb_steps):
        theta += eta * _g(theta)
        log += [logger(theta)]
    return log, theta
log, theta = PolicyGradient(naturalGrad,gridMDP,5e-2)

pis=jnp.stack([e['pi'] for e in log])
js=jnp.stack([e['J'] for e in log])
fig, ax = plt.subplots(1,2,figsize=(10,5))
ax[0].plot(js)
ax[0].set_title('Objective function (maximizing)')
ax[1].plot(pis[:,0,:2])
ax[1].set_title('Two example policy coordinates')
plt.show()