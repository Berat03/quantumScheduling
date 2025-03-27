from collections import defaultdict
from itertools import product
from itertools import combinations


def getPossibleStates(initialEdges, maxAge=2):
    # Create the base state from the initial edges
    baseState = defaultdict(int)
    for edge in initialEdges:
        baseState[tuple(sorted(edge))] = -1  # -1 means the edge is not used
    
    # Convert edges to sorted tuples for consistency
    edges = [tuple(sorted(edge)) for edge in initialEdges]
    
    # Generate all possible age combinations
    possible_ages = [-1] + list(range(1, maxAge + 1))  # -1: no entanglement, > 0 == age
    all_states = []
    
    # Generate all possible combinations of ages for each edge
    for age_combination in product(possible_ages, repeat=len(edges)):
        new_state = defaultdict(int)
        for edge, age in zip(edges, age_combination):
            new_state[edge] = age
        # Convert to regular dict and make it hashable
        all_states.append(tuple(sorted(dict(new_state).items())))
    
    return all_states

def find_all_paths(start, end, available, path=None):
    if path is None:
        path = [start]
    
    if start == end:
        yield path
        return
    
    for edge in available:
        if start in edge:
            next_node = edge[0] if edge[1] == start else edge[1]
            if next_node not in path:  # Avoid cycles
                yield from find_all_paths(next_node, end, available, path + [next_node])

def getSwappingOutcomes(state, pSwap, goalEdges):
    swappingOutcomes = [(state, 1.0, {}, [])]  # No-swap case, empty dict for edges
    state_dict = dict(state)
    
    # Find paths for each goal
    goal_paths = defaultdict(list)
    for goal in goalEdges:
        available = [e for e, age in state_dict.items() if age > 0]
        all_paths = list(find_all_paths(goal[0], goal[1], available))
        for path in all_paths:
            if len(path) > 1:
                edges = [tuple(sorted([path[i], path[i+1]])) for i in range(len(path)-1)]
                if all(edge in state_dict and state_dict[edge] > 0 for edge in edges):
                    goal_paths[goal].append(edges)

    # Try each goal independently and their combinations
    for r in range(1, len(goal_paths) + 1):
        for goals in combinations(goal_paths.keys(), r):
            # For each goal, try each possible path
            for path_combination in product(*[goal_paths[goal] for goal in goals]):
                edges_per_goal = {}  # Dictionary to track which edges are used for each goal
                for goal, edges in zip(goals, path_combination):
                    edges_per_goal[goal] = edges
                
                # Get all needed edges
                needed_edges = set()
                for edges in path_combination:
                    needed_edges.update(edges)
                
                new_state = state_dict.copy()
                for edge in needed_edges:
                    new_state[edge] = -1
                    
                num_swaps = sum(len(edges) - 1 for edges in path_combination)
                new_state_tuple = tuple(sorted(new_state.items()))
                
                # Success case
                success_prob = pSwap ** num_swaps
                swappingOutcomes.append((new_state_tuple, success_prob, edges_per_goal, list(goals)))
                
                # Failure case
                fail_prob = 1 - success_prob
                swappingOutcomes.append((new_state_tuple, fail_prob, edges_per_goal, []))

    return swappingOutcomes


def ageState(state, maxAge):
    new_state = {}
    for edge, age in dict(state).items():
        if age == -1:  # If no entanglement, stays the same
            new_state[edge] = -1
        elif age >= maxAge:  # If would exceed maxAge, remove entanglement
            new_state[edge] = -1
        else:  # Otherwise increment age
            new_state[edge] = age + 1
            
    return tuple(sorted(new_state.items()))

    
def getGenerationOutcomes(state, pGen):
    state_dict = dict(state)
    empty_edges = [edge for edge, age in state_dict.items() if age == -1]
    generationOutcomes = []
    
    # Changed range to start from 0 to include no-generation case
    for r in range(0, len(empty_edges) + 1):
        for edges_to_generate in combinations(empty_edges, r):
            new_state = state_dict.copy()
            
            for edge in edges_to_generate:
                new_state[edge] = 1
                
            new_state_tuple = tuple(sorted(new_state.items()))
            
            num_generated = len(edges_to_generate)
            num_not_generated = len(empty_edges) - num_generated
            probability = (pGen ** num_generated) * ((1-pGen) ** num_not_generated)
            
            generationOutcomes.append((new_state_tuple, probability))
    
    return generationOutcomes



# states =  state1, state 2
# actions = (state, action)
# transitions = (state, action) : newState, probability)

def getTransitionProbabilities(state, pSwap, pGen, goalEdges, maxAge):
    transitions = []
    
    # First get all possible swapping outcomes
    swapping_results = getSwappingOutcomes(state, pSwap, goalEdges)
    
    # For each swapping outcome, try all possible generation combinations
    for intermediate_state, swap_prob, edges_per_goal, goals_achieved in swapping_results:
        # Age the state after swapping
        aged_state = ageState(intermediate_state, maxAge)
        
        # Get all possible generation outcomes from the aged state
        generation_results = getGenerationOutcomes(aged_state, pGen)
        
        # Combine probabilities and create final transitions
        for final_state, gen_prob in generation_results:
            total_prob = swap_prob * gen_prob
            
            action = {
                'swap_edges_per_goal': edges_per_goal,  # Now contains mapping of goal -> edges
                'goals_achieved': goals_achieved
            }
            
            transitions.append(((state, action), (final_state, total_prob)))
    
    return transitions
