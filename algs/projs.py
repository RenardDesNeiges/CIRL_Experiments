import jax.numpy as jnp
from jax.numpy import linalg as jla
from itertools import accumulate

def euclidean_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0 
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex
    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    sh = v.shape  
    v = jnp.reshape(v,(list(accumulate(sh,lambda x,y:x*y))[-1],))
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and jnp.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = jnp.sort(v)[::-1]
    cssv = jnp.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = jnp.nonzero(u * jnp.arange(1, n+1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = float(cssv[rho] - s) / (rho+1)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return jnp.reshape(w,sh)


def euclidean_l1ball(v, s=1):
    """ Compute the Euclidean projection on a L1-ball
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. || w ||_1 <= s
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the L1-ball
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the L1-ball of radius s
    Notes
    -----
    Solves the problem by a reduction to the positive simplex case
    See also
    --------
    euclidean_proj_simplex
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    sh = v.shape  
    v = jnp.reshape(v,(list(accumulate(sh,lambda x,y:x*y))[-1],))
    n, = v.shape  # will raise ValueError if v is not 1-D
    # compute the vector of absolute values
    u = jnp.abs(v)
    # check if v is already a solution
    if u.sum() <= s:
        # L1-norm is <= s
        return jnp.reshape(v,sh)
    # v is not already a solution: optimum lies on the boundary (norm == s)
    # project *u* on the simplex
    w = euclidean_simplex(u, s=s)
    # compute the solution to the original problem on v
    w *= jnp.sign(v)
    return jnp.reshape(w,sh)

def euclidean_l2ball(v, s=1):
    """ Compute the Euclidean projection on a L2-ball
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the L1-ball
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the L1-ball of radius s
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    norm = jla.norm(v)
    if norm < s: return v
    return v*(s/norm)

def euclidian_pos_orthant(v,):
    """Projects to the positive orthant

    Args:
        v (np.ndarray/jax array): vector to be projected onto the orthant

    Returns:
        jax array: projected vector
        
    Quite a trivial operation (just returns max(0,v_i) for each coordinate v_i of v).
    """
    sh = v.shape  
    v = jnp.reshape(v,(list(accumulate(sh,lambda x,y:x*y))[-1],))
    return jnp.reshape(jnp.where(v<0, 0, v),sh)