import jax
from jax import numpy as jnp
from abc import ABC, abstractmethod

from env.mdp import MarkovDecisionProcess
from env.complete import Complete
from env.gridworld import Gridworld
from algs.projs import euclidean_l2ball


class ExampleMDPs(ABC):
    @abstractmethod
    def bandit1(n:int=2,reward_class:str='L2')->MarkovDecisionProcess:
        key = jax.random.PRNGKey(1)
        mdp = Complete(1,n,0.95)
        key, sk = jax.random.split(key)
        mdp.init_distrib = jnp.exp(jax.random.uniform(sk,(mdp.n,))) / \
        jnp.sum(jnp.exp(jax.random.uniform(key,(mdp.n,))))
        assert reward_class == 'L2'
        mdp.R = euclidean_l2ball(mdp.R, 1)
        return mdp
    
    @abstractmethod
    def minimal1(reward_class:str='L2')->MarkovDecisionProcess:
        key = jax.random.PRNGKey(1)
        mdp = Complete(2,2,0.95)
        key, sk = jax.random.split(key)
        mdp.init_distrib = jnp.exp(jax.random.uniform(sk,(mdp.n,))) / \
        jnp.sum(jnp.exp(jax.random.uniform(key,(mdp.n,))))
        assert reward_class == 'L2'
        mdp.R = euclidean_l2ball(mdp.R, 1)
        return mdp
    
    @abstractmethod
    def gworld1(reward_class:str='L2')->MarkovDecisionProcess:
        key = jax.random.PRNGKey(1)
        R = 1; P = -2; 
        goals = [((2,0),R)]
        mdp = Gridworld(3,3,0.1,0.9,goals=goals,obstacles=[]) 
        key, sk = jax.random.split(key)
        mdp.init_distrib = jnp.exp(jax.random.uniform(sk,(mdp.n,))) / \
        jnp.sum(jnp.exp(jax.random.uniform(key,(mdp.n,))))
        assert reward_class == 'L2'
        mdp.R = euclidean_l2ball(mdp.R, 1)
        return mdp

    @abstractmethod
    def gworld2(reward_class:str='L2')->MarkovDecisionProcess:
        key = jax.random.PRNGKey(1)
        R = 1; P = -2; 
        R = 1; P = -2; goals = [((2,0),R),((1,0),P),((1,1),P)]
        mdp = Gridworld(3,3,0.1,0.9,goals=goals,obstacles=[]) 
        mdp.init_distrib =  jnp.exp(jax.random.uniform(key,(mdp.n,))) / \
            jnp.sum(jnp.exp(jax.random.uniform(key,(mdp.n,))))
        key, sk = jax.random.split(key)
        mdp.init_distrib = jnp.exp(jax.random.uniform(sk,(mdp.n,))) / \
        jnp.sum(jnp.exp(jax.random.uniform(key,(mdp.n,))))
        assert reward_class == 'L2'
        mdp.R = euclidean_l2ball(mdp.R, 1)
        return mdp
    