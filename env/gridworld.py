import numpy as np
from typing import List, Tuple
from .mdp import MarkovDecisionProcess

Point = Tuple[int,int]
Goal = Tuple[Point,float]
# TODO define a general MDP super class (abstract)
# TODO define relevant visualization functions


class MarkovDecisionProcess():
    def __init__(self,
            n  :int, 
            m :int, 
            gamma :float, 
            P_sa : np.ndarray,
            R : np.ndarray,
            ) -> None:
        self.n : int = n
        self.m : int = m
        self.gamma : float = gamma
        self.P_sa : np.ndarray = P_sa
        self.R : np.ndarray = R

    def next_state_distribution(self, s:int, a:int)->np.ndarray:
        """Given a fixed state-action pair, gives the distribution on the next state.

        Args:
            s (int): current state s
            a (int): action a

        Returns:
            np.ndarray: n-sized array containing the distribution of the random variable s'
        """
        return self.P_sa[s,a,:]


class Gridworld(MarkovDecisionProcess):
    """Defines a Gridworld MDP
    """
    def __init__(self,
                grid_width  :int = 3, 
                grid_height :int = 3, 
                noise       :float = 0.1, 
                gamma       :float = 0.9, 
                goals       :List[Goal] = [((0,0),2.0)] # reward of 2 in point (0,0)
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
        
        super().__init__(self.n,self.m,gamma,P_sa,R)
        
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