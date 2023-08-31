import jax
from jax import numpy as jnp

from typing import Callable

from env.mdp import MarkovDecisionProcess
from algs.utils import flatten

def J(  m:MarkovDecisionProcess,
        p:jnp.ndarray,
        r:jnp.ndarray               = None,
        reg: Callable               = None
        )->float:
    """Computes the exact return of a policy p w.r.t to some reward r on some MDP mdp.

    Args:
        mdp (MarkovDecisionProcess): MDP that we want to compute return on.
        p (jnp.ndarray): policy of the agent on the MDP.
        r (jnp.ndarray, optional): Reward that we want the return for. Defaults to None, when none, uses the MDP's default reward.
        reg (Callable, optional): Regularizer function applied to the policy. Defaults to None, when none return is unregularized.

    Returns:
        float: the return.
    """
    
    if r is None: r = m.R
    if reg is None: reg_term = 0
    else:  reg_term = jnp.dot(jax.vmap(reg)(p),
                    m.state_occ_measure(p))
    J = jnp.dot(flatten(r),flatten(m.occ_measure(p)))
    J -= reg_term
    return  J


# TODO: defined the CIRL lagrangian