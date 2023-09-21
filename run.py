
import jax
from jax.config import config; config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import sys; sys.path.insert(1, '..')

import dill as pickle

from algs.train import IRL_Trainer, TracePlotter
from env.utils import ExampleMDPs
from env.sample import Sampler
from algs.irl import irlL1Proj
from algs.irl import initStateOnlyIRL, stochNaturalIRL

import jax.numpy as jnp
from dataclasses import dataclass

@dataclass
class trainingParameters():
    eg_plr :float = 1e-1
    eg_rlr :float = 5e-2
    mc_plr :float = 1e-1
    mc_rlr :float = 5e-2
    beta   :float = 0.1
    tmax   :float = 70
    batch  :int   = 10000
    horizon:int   = 250
    steps  :int   = 200


def run_exact_grads_irl(params,mdp,rFun,key):
    trainer = IRL_Trainer(mdp,policy_lr=2,reward_lr=2e-1,beta=0.1)
    trainer = IRL_Trainer(
                        mdp,policy_lr=params.eg_plr,
                        reward_lr=params.eg_rlr,
                        beta=params.beta,
                        init_params=initStateOnlyIRL,
                        rFun=rFun,
                        max_theta=params.tmax,
                        proj=irlL1Proj,
                        key = key)
    
    trainData = trainer.train(params.steps)
    pickle.dump(trainData, open( "logs/exact_irl.pkl", "wb" ) )
    
def run_mc_grads_irl(params, mdp,rFun,key):
    smp = Sampler(mdp,batchsize=1000,horizon=250)
    trainer = IRL_Trainer(
                        mdp,
                        policy_lr=params.mc_plr,
                        reward_lr=params.mc_rlr,
                        beta=params.beta,
                        init_params=initStateOnlyIRL,
                        sampler=smp,
                        rFun=rFun,
                        gradients=stochNaturalIRL,
                        max_theta=params.tmax,
                        proj=irlL1Proj,
                        key = key,
                        )
    trainData = trainer.train(params.steps)
    pickle.dump(trainData, open( "logs/mc_irl.pkl", "wb" ) )
    
def main():    
    key = jax.random.PRNGKey(0)
    params = trainingParameters()

    # defining a state-only-reward gridworld MDP
    mdp = ExampleMDPs.gworld1()
    _m = mdp.m
    def rFun(w):
        return jnp.repeat(w,_m,1)
    
    w_ref = jnp.expand_dims(jnp.array([1 if s==2 else 0 for  s in range(9)]),1)
    mdp.R = rFun(w_ref)
    
    print('running the exact gradient experiment')
    run_exact_grads_irl(params, mdp,rFun,key)
    print('running the stochastic gradient experiment')
    run_mc_grads_irl(params, mdp,rFun,key)

if __name__ == "__main__":
    main()