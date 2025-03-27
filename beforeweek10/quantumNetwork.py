import random
import numpy as np
import networkx as nx
from collections import deque, defaultdict
import matplotlib.pyplot as plt


class QuantumNetwork:
    def __init__(self, initialEdges, pGen, pSwap, maxAge, goalWeights, windowSize=100000000):
        self.initialEdges = initialEdges
        self.pGen = pGen
        self.pSwap = pSwap
        self.maxAge = maxAge
        self.goalWeights = goalWeights
        self.timestep = 0
        self.entanglementRate = defaultdict(lambda: deque(maxlen=windowSize))
        
        self._initializeGoalEntanglements(windowSize)
        self._initializeNodeCapacity()
        self._initializeGraph()
        self._avoidZeroDivision = 0.00001
    
    def _initializeGraph(self) -> None:
        self.G = nx.Graph()
        nodes = set()
        for edge in self.initialEdges:
            nodes.add(edge[0])
            nodes.add(edge[1])
        self.G.add_nodes_from(sorted(nodes))
        
    def _initializeGoalEntanglements(self, windowSize) -> None:
        for goal_edge, _ in self.goalWeights:
            self.entanglementRate[goal_edge] = deque(maxlen=windowSize)
    
    def _initializeNodeCapacity(self) -> None:
        self.nodeCapacity = defaultdict(int)
        for edge in self.initialEdges:
            self.nodeCapacity[edge[0]] += 1
            self.nodeCapacity[edge[1]] += 1
            
    def generateEntanglement(self, node1: int, node2: int) -> None:
        edge = tuple(sorted((node1, node2)))
        
        if (self.G.degree(edge[0]) >= self.nodeCapacity[edge[0]] or 
            self.G.degree(edge[1]) >= self.nodeCapacity[edge[1]]):
            return 
        
        if not self.G.has_edge(*edge):
            self.G.add_edge(*edge, entanglement = 0)
        else:
            self.G.edges[edge]['entanglement'] = 0
            
    def generateGlobalEntanglementsProbabalistically(self):
        for edge in self.initialEdges:
            if random.random() < self.pGen:
                self.generateEntanglement(*edge)
    
    def getState(self) -> dict:
        edge_info = {}
        for edge in self.G.edges():
            edge_info[edge] = self.G.edges[edge]['entanglement']
        return edge_info

    def ageEntanglements(self) -> None:
        for edge in list(self.G.edges()):
            self.G.edges[edge]['entanglement'] += 1
            
            if self.G.edges[edge]['entanglement'] > self.maxAge:
                self._discardEntanglement(edge)
                
    def _discardEntanglement(self, edge: tuple) -> None:
        assert self.G.has_edge(*edge), f"Edge {edge} does not exist in the graph, yet we are trying to discard it."
        if self.G.has_edge(*edge):
            self.G.remove_edge(*edge)
                
    def takeAction(self, action: tuple[list[tuple[int, int]], tuple[int, int]]) -> None:
        swaps, goal = action
        if swaps:
            for edge in swaps:
                self._discardEntanglement(edge)
        
        

    def _updateGoalEntanglementRates(self, goal) -> None:
        for goal_edge in self.entanglementRate.keys():
            self.entanglementRate[goal_edge].append(0)
        
        if goal:  
            goal_edge = tuple(sorted(goal))
            if self.entanglementRate[goal_edge]:
                self.entanglementRate[goal_edge][-1] = 1
    

    def getActions(self) -> list[tuple[list[tuple[int, int]], tuple[int, int]]]:
        possible_actions = []
        for goal_edge, _ in self.goalWeights:
            if nx.has_path(self.G, *goal_edge):
                shortest_path = nx.shortest_path(self.G, *goal_edge)
                path_edges = [tuple(sorted([shortest_path[i], shortest_path[i+1]])) for i in range(len(shortest_path)-1)]
                
                if all(self.G.has_edge(*edge) for edge in path_edges):
                    possible_actions.append((path_edges, goal_edge))
        possible_actions.append(([], None))
        return possible_actions # duplicate actions?
                
    def drawNetwork(self) -> None:
        plt.figure(figsize=(12, 8))
        plt.clf()
        G_viz = self.G.copy()
        G_viz.add_edges_from(self.initialEdges)
        
        pos = {
            0: (-1, 0.5),   
            1: (-1, -0.5),  
            2: (0, 0),      
            3: (1, 0),      
            4: (2, 0.5),   
            5: (2, -0.5)   
        }
        
        nx.draw_networkx_edges(G_viz, pos=pos, 
                            edgelist=self.initialEdges,
                            edge_color='grey',
                            style='dashed',
                            alpha=0.5)
        
        for edge in self.G.edges():
            nx.draw_networkx_edges(G_viz, pos=pos,
                                edgelist=[edge],
                                edge_color='blue',
                                width=2)
            
            # Label edges with entanglement value (age)
            entanglement_age = self.G.edges[edge]['entanglement']
            edge_x = (pos[edge[0]][0] + pos[edge[1]][0]) / 2
            edge_y = (pos[edge[0]][1] + pos[edge[1]][1]) / 2
            plt.text(edge_x, edge_y, str(entanglement_age), 
                    bbox=dict(facecolor='white', edgecolor='lightgray', alpha=1))
        
        nx.draw_networkx_nodes(G_viz, pos=pos, node_color='lightblue')
        nx.draw_networkx_labels(G_viz, pos=pos)
        plt.show()

    
    def getGoalRate(self, goal_edge) -> float:
        if goal_edge not in self.entanglementRate or sum(self.entanglementRate[goal_edge]) == 0:
            return self._avoidZeroDivision
        return sum(self.entanglementRate[goal_edge]) / len(self.entanglementRate[goal_edge])
    
    def getActionReward(self, action: tuple[list[tuple[int, int]], tuple[int, int]]) -> float:
        swaps, goal = action

        # TODO: Cleaner way to get the weights?
        if not goal:
            return 0
        goal_weight = 0
        for goal_edge, weight in self.goalWeights:
            if goal_edge == goal:
                goal_weight = weight
                break
        
        goalEntanglementRate = self.getGoalRate(goal)
        goalInstantaneousRate = self.pSwap ** len(swaps)
        
        edge_reward = (goalInstantaneousRate / goalEntanglementRate) 
        if edge_reward <= 0:
            edge_reward = self._avoidZeroDivision
        return np.log(edge_reward)
    
    def getActionEpsilonGreedyPolicy(self, Q, state, epsilon):
        possible_actions = self.getActions() 
        
        if not possible_actions:
            return ([], None) 
            
        actions_and_values = Q.getActionAndValues(state)
        if not actions_and_values:
            return random.choice(possible_actions)
        print(np.random.rand())
        if np.random.rand() < epsilon:
            return random.choice(possible_actions)
        else:
            valid_actions_and_values = [
                (action, value) for action, value in actions_and_values 
                if action in possible_actions
            ]
            if not valid_actions_and_values:
                return random.choice(possible_actions)
            return max(valid_actions_and_values, key=lambda x: x[1])[0]