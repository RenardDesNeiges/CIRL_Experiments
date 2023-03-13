import jax
import jax.numpy as jnp

class NaiveGDA():
    """ Naive implementation of the Gradient Descent-Ascent Algorithm,
        This basically never converges and one should almost always use 
        Extra-Grad or OptimiticGDA instead.
    """
    
    def step(f,x,y,eta,tau):
        Dx = jax.grad(f,0)(x,y) # minimization parameter grad
        Dy = jax.grad(f,1)(x,y) # maximization parameter grad
        return x - eta*Dx, y + eta*tau*Dy

    def solve(f,x,y,steps,eta,tau, step=None):
        xs = []; ys = []; fs = [];
        for _ in range(steps):
            xs += [x]; ys += [y]
            fs += [f(x,y)]
            if step:
                x, y = step(f,x,y,eta,tau)
            else:
                x, y = NaiveGDA.step(f,x,y,eta,tau)
        xs = jnp.array(xs); ys = jnp.array(ys); fs = jnp.array(fs)
        return xs,ys,fs
    

class OptimiticGDA():
    """ Implementation of the Optimistic Gradient Descent-Ascent
        Algorithm [OGDA], as presented in https://arxiv.org/abs/1901.08511
    """
    def step(f,x,y,prev_dx,prev_dy,eta,tau):
        Dx = jax.grad(f,0)(x,y) # minimization parameter grad
        Dy = jax.grad(f,1)(x,y) # maximization parameter grad
        return x - 2 * eta*Dx + eta*prev_dx, y + 2*eta*tau*Dy-eta*prev_dy, Dx, Dy

    def solve(f,x,y,steps,eta,tau,step = None):
        xs = []; ys = []; fs = []; prev_dx = 0; prev_dy = 0
        for _ in range(steps):
            xs += [x]; ys += [y]
            fs += [f(x,y)]
            if step:
                x, y, prev_dx, prev_dy = step(f,x,y,prev_dx,prev_dy,eta,tau)
            else:
                x, y, prev_dx, prev_dy = OptimiticGDA.step(f,x,y,prev_dx,prev_dy,eta,tau)
                
        xs = jnp.array(xs); ys = jnp.array(ys); fs = jnp.array(fs)
        return xs,ys,fs


class ExtraGradientGDA():
    """ Implementation of the Extra Gradient Gradient Descent-Ascent
        Algorithm [EGDA], as presented in https://arxiv.org/abs/1901.08511
    """

    def step(f,x,y,eta,tau):
        Dx = jax.grad(f,0)(x,y) # minimization parameter grad
        Dy = jax.grad(f,1)(x,y) # maximization parameter grad
        EDx = jax.grad(f,0)(x - eta*Dx, y + eta*tau*Dy)
        EDy = jax.grad(f,1)(x - eta*Dx, y + eta*tau*Dy)
        return x - eta*EDx, y + eta*tau*EDy

    def solve(f,x,y,steps,eta,tau, step = None):
        xs = []; ys = []; fs = []
        for _ in range(steps):
            xs += [x]; ys += [y]
            fs += [f(x,y)]
            if step:
                x, y = step(f,x,y,eta,tau)
            else:
                x, y = ExtraGradientGDA.step(f,x,y,eta,tau)
        xs = jnp.array(xs); ys = jnp.array(ys); fs = jnp.array(fs)
        return xs,ys,fs
