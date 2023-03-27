import numpy as np
from typing import List, Tuple
from .mdp import MarkovDecisionProcess
import matplotlib
from matplotlib.patches import Rectangle

Point = Tuple[int,int]
Goal = Tuple[Point,float]


cmap = matplotlib.cm.get_cmap('Spectral')


class Gridworld(MarkovDecisionProcess):
    """Defines a Gridworld MDP
    """
    def __init__(self,
                grid_width  :int = 3, 
                grid_height :int = 3, 
                noise       :float = 0.1, 
                gamma       :float = 0.9, 
                goals       :List[Goal] = [((0,0),2.0)], # reward of 2 in point (0,0)
                obstacles   :List[Point] = [(2,2)] # forbidden to go here
                ) -> None:
        
        
        self.actions : List[Point] = [(1,0),(-1,0),(0,1),(0,-1)]
        self.grid_width : int = grid_width
        self.grid_height : int = grid_height
        self.noise : float = noise
        self.n : int = grid_width*grid_height
        self.m : int = len(self.actions)
        self.goals : List[Goal] = goals
        
        P_sa : np.ndarray = np.array(  [[[self._transition_dynamics(s,a,sp) 
                                            for s  in range(self.n)             ] 
                                            for a  in range(self.m)             ] 
                                            for sp in range(self.n)             ],dtype=np.float64)
        
        R : np.ndarray = np.array(  [[self._reward(s,a)
                                            for a  in range(self.m)             ] 
                                            for s  in range(self.n)             ],dtype=np.float64)
        
        init_distrib = np.zeros((self.n),dtype=np.float64); init_distrib[0] = 1. # TODO : allow for passing another distribution as an argument
        
        b  : np.ndarray = np.ones((len(obstacles)))*0.5
        Psi : np.ndarray = np.array(  [[[self._cost_matrix(s,a,obs) 
                                            for s  in range(self.n)             ] 
                                            for a  in range(self.m)             ] 
                                            for obs in obstacles                ],dtype=np.float64)
        
        super().__init__(self.n,self.m,gamma,P_sa,R, init_distrib=init_distrib,b=b,Psi=Psi)
        
    def _cost_matrix(self, s,a,obs):
        if self.point2state(obs) == s:
            return 2
        return 0 
        
    def neighbouring(self, p1: Point, p2: Point) -> bool:
        """Is p1 a neighbor to p2?

        Args:
            p1 (Point): point 1
            p2 (Point): point 2

        Returns:
            bool: True iff p1 in neighborhood(p2)
        """
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) <= 1
    
    def state2point(self, s:int)->Point:
        """Returns a grid-point when given a state

        Args:
            s (int): state

        Returns:
            Point: grid-point
        """
        return (s % self.grid_width, s // self.grid_width)
    
    def point2state(self, p:Point)->int:
        """Returns a state from a grid-point

        Args:
            p (Point): grid-point

        Returns:
            int: state s
        """
        return  p[0] + p[1] * self.grid_width
    
    def is_corner(self, p:Point)->bool:
        """Returns True if p is in a corner

        Args:
            p (Point): the point p on the grid

        Returns:
            bool: is_corner(p)
        """
        if  (p[0] == 0 or p[0] == self.grid_width-1) and \
            (p[1] == 0 or p[1] == self.grid_height-1) :
            return True
        return False
        
    def is_edge(self, p:Point)->bool:
        """Returns True if p is on an edge

        Args:
            p (Point): the point p on the grid

        Returns:
            bool: is_edge(p)
        """
        if  (p[0] == 0 or p[0] == self.grid_width-1) or \
            (p[1] == 0 or p[1] == self.grid_height-1) :
            return True
        return False
    
    def is_on_grid(self,p:Point,act:Point)->bool:
        """Returns False if the move act from pos p brings the agent off-grid

        Args:
            p (Point): the agent's position
            act (Point): the specified move

        Returns:
            bool: is_on_grid(p,act)
        """
        return (0 <= p[0] + act[0] < self.grid_width and
                0 <= p[1] + act[1] < self.grid_height)
    
    def grid_move(self, p:Point, act:Point)->Point:
        """Returns the arrival point when moving from p under action act

        Args:
            p (Point): the agent's position
            act (Point): the movement

        Returns:
            Point: the agent's next position
        """
        return (p[0]-act[0],p[1]-act[1])
    
    def _transition_dynamics(self, s:int, a:int, sp:int) -> float:
        """Returns the markov transition probability P(sp|s,a)

        Args:
            s (int): initial state
            a (int): action
            sp (int): final state

        Returns:
            float: P(sp|s,a)
        """
        p   : Point = self.state2point(s)
        pp  : Point = self.state2point(sp)
        act : Point = self.actions[a]
        
        if not self.neighbouring(p,pp): 
            return 0.0
        
        if self.grid_move(p,act) == (pp[0],pp[1]): # the state is the intended move
            return 1 - self.noise + self.noise/self.m
        if (p[0],p[1]) != (pp[0],pp[1]): # slip
            return self.noise/self.m
        
        # If s==sp case (implicitely specified)
        if self.is_corner(p):
            if self.is_on_grid(p,act):
                return self.noise * 2/self.m
            else:
                return 1 - self.noise + self.noise*2/self.m
        elif self.is_edge(p):
            if self.is_on_grid(p,act):
                return self.noise * 1/self.m
            else:
                return 1 - self.noise + self.noise*1/self.m
        else: # when not on the grid border, s==sp is a 0 probaility event
            return 0.0
        
    def _reward(self, s:int, a:int)->float:
        p = self.state2point(s)
        act = self.actions[a]
        if not self.is_on_grid(p,act):
            return -1.0
        for g, r in self.goals:
            if g==p:
                return r
        return 0.0
    
    def states2grid(self,states:np.ndarray)->np.ndarray:
        """Converts a state vector (n) into the grid world (width x height)
        usefull for plotting

        Args:
            states (np.ndarray): the state vector (usually a distribution, or the value)

        Returns:
            np.ndarray: the grid shaped representation 
        """
        grid : np.ndarray = np.array([[
                    states[self.point2state((x,y))] for x in range(self.grid_width)]  
                                                    for y in range(self.grid_height)], dtype=np.float64)
        return grid
    

def gridplot(gridworld:Gridworld, ax, scalar=None, policy = None, 
             stochastic_policy=None, obstacles=None, goals=None,traj=None):
    ax.set_xlim([0,gridworld.grid_width])
    ax.set_ylim([0,gridworld.grid_height])
    if scalar is not None:
        for s, value in enumerate(scalar):
            ax.add_patch(Rectangle(gridworld.state2point(s),1,1,facecolor=cmap((value-np.min(scalar))/(np.max(scalar)-np.min(scalar)))))
    if policy is not None:
        for s, value in enumerate(policy):
            point = np.array(gridworld.state2point(s)) + 0.5
            arrow = np.array(gridworld.actions[value])*0.3
            ax.arrow(point[0],point[1],arrow[0],arrow[1],head_width=0.1,color='k')
    if stochastic_policy is not None:
        for s, value in enumerate(stochastic_policy):
            point = np.array(gridworld.state2point(s)) + 0.5
            for a, prob in enumerate(value):
                arrow = np.array(gridworld.actions[a])*0.3*prob
                if prob > 1e-5:
                    ax.arrow(point[0],point[1],arrow[0],arrow[1],head_width=0.1,color='k')
    if obstacles is not None:
        for value in obstacles:
            ax.add_patch(Rectangle(value,1,1,edgecolor="r",facecolor='none',hatch='//'))
    if goals is not None:
        for value in goals:
            ax.add_patch(Rectangle(value[0],1,1,edgecolor="g",facecolor='none',hatch='//'))
    if traj is not None:
        pointx = [event[0][0]+0.5 for event in traj]
        pointy = [event[0][1]+0.5 for event in traj]
        for step, event in enumerate(traj):
            point = event[0]
            arrow = np.array(gridworld.actions[event[1]])*0.3
            ax.arrow(point[0]+0.5,point[1]+0.5,arrow[0],arrow[1],head_width=0.1,color='g')
            ax.text(point[0]+0.7,point[1]+0.7, f'{step}', fontsize=15)
        ax.plot(pointx,pointy,linewidth=2,color='b')