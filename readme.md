# (Constrained) Inverse Reinforcement Learning : Algorithms and Experiments

![](examples/example_IRL.png)

RL experiment environment for my master project "**Convergence and Sample Complexity of Constrained Inverse Reinforcement Learning**" and for the paper that will come out of it.

The following repository contains. 
1. An environment (`env`) module, that contains a Markov Decision Process class, samplers to generate trajectories from MDPs and specific MDP implementations (specifically a Gridworld environment and a random MDP generator). These MDPs are implemented in such a way that it is easy to compute exact gradients on them using `jax` automatic differentiation.
1. An algorithm (`alg`) module that implements policy gradient RL algorithm (both natural PG adn vanilla PG) as well as inverse reinforcement learning (IRL) and constrained IRL (CIRL). The algorithms are implemented in `jax` and aim to be as close as possible to the mathematical formulation (as they serve as experiments for a theory paper).

