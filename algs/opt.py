import jax
from jax import numpy as jnp
from jax.numpy import linalg as jla
import jax.nn as nn

from tqdm import tqdm
from typing import Callable, Any, Dict, Tuple, List

from env.mdp import MarkovDecisionProcess
from env.sample import Sampler

"""

Generic class that provides a basic structure to the implementation of any (!)
        Iterative optimization procedure.
        Initializes the optimization parameters (returns a Dict)

"""

def defaultLogger(  params: Dict[str,Any],
                    grads:  Dict[str,Any],
                    step:   Dict[str,Any],
                    i:      int             ):
    return {
        'params'    : params,
        'grads'     : grads,
        'step'      : step,
        'iter'      : i
    }
    

class Optimizer():
    def __init__(self,  init:   Callable[[None],Dict], 
                        log:    Callable[[Dict,Dict,Dict,int],Dict], 
                        grad:   Callable[[None],Dict], 
                        proc:   Callable[[Dict],Dict],
                        proj:   Callable[[Dict],Dict]) -> None:
        """ Generic class that provides a basic structure to the implementation of any (!)
            first order optimization procedure.

        Args:
            init (Callable[[None],Dict]): Initializes the optimization parameters
            log (Callable[[Dict,Dict,Int],Dict]): Logs 
            grad (Callable[[None],Dict]): Computes the gradient
            proc (Callable[[Dict],Dict]): Processes the gradients
        """
        self.init   = init
        self.log    = log
        self.grad   = grad
        self.proc   = proc
        self.proj   = proj
        
        
    def train(self,key:jax.random.KeyArray,steps:int,pbar:bool=False)->Tuple[Dict[str,Any],List[Dict[str,Any]]]:
        """Optimize for some number of steps.

        Args:
            key (jax.random.KeyArray): jax PRNG key.
            steps (int): number of steps to optimize for.
            pbar (bool, optional):if True display a tqdm progress bar. Defaults to False.

        Returns:
            Tuple[Dict[str,Any],List[Dict[str,Any]]]: The optimized parameters and a log of training (containing the outputs of the logger function)
        """
        x   = self.proj(self.init())                   # initialize the parameters somewhere
        log = []
        for i in tqdm(range(steps),disable=(not pbar)):
            key, sk = jax.random.split(key)
            _g = self.grad(sk, x)           # compute gradients
            _s = self.proc(_g)              # process the gradients into a step
            log += [self.log(x,_g,_s,i)]    # log the parameters, gradients, step and iter-count
            x = {k:x[k]+v for k, v in _s.items()}     # take the step
            x = self.proj(x)
        return x, log