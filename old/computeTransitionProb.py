from collections import defaultdict
from itertools import product
from itertools import combinations
from itertools import chain

# Global parameters (make sure these match your valueIterationPolicy.py)
pSwap = 0.8
maxAge = 2

def getPossibleStates(initialEdges, maxAge):
    """
    Generate all possible states considering all possible age combinations
    Ages can be: -1 (no entanglement), 1 to maxAge
    """
    edges = [tuple(sorted(edge)) for edge in initialEdges]
    possible_ages = [-1] + list(range(1, maxAge + 1))  # [-1, 1, 2] for maxAge=2
    
    all_states = []
    # Use product to generate all possible combinations of ages
    for age_combination in product(possible_ages, repeat=len(edges)):
        new_state = {}
        for edge, age in zip(edges, age_combination):
            new_state[edge] = age
        # Convert to sorted tuple format for consistency
        state_tuple = tuple(sorted(new_state.items()))
        all_states.append(state_tuple)
    
    return all_states

def find_paths_for_goal(state, goal):
    """Find all possible paths for a goal using available edges"""
    start, end = goal
    available_edges = [(edge, age) for edge, age in state.items() if age >= 0]
    paths = []
    
    def dfs(current, path, used):
        if current == end:
            paths.append(tuple(path))
            return
        
        for (e1, e2), age in available_edges:
            if e1 == current and (e1, e2) not in used:
                dfs(e2, path + [(e1, e2)], used | {(e1, e2)})
            elif e2 == current and (e1, e2) not in used:
                dfs(e1, path + [(e1, e2)], used | {(e1, e2)})
    
    dfs(start, [], set())
    return paths


def getSwappingOutcomes(state, goalEdges):
    """Modified to handle overlapping goals better"""
    
    swappingOutcomes = []
    available_edges = {edge for edge, age in state.items() if age >= 0}
    
    # Get all possible paths for each goal
    goal_paths = {}
    for goal in goalEdges:
        goal_paths[goal] = find_paths_for_goal(state, goal)
    
    # Try all combinations of goals
    for goal_subset in powerset(goalEdges):
        if not goal_subset:
            continue
            
        # Check if we can achieve these goals simultaneously
        paths_for_goals = {}
        edges_used = set()
        valid_combination = True
        
        for goal in goal_subset:
            valid_path = None
            for path in goal_paths[goal]:
                path_edges = set(path)
                if not (path_edges & edges_used):  # No overlap with already used edges
                    valid_path = path
                    break
            
            if valid_path:
                paths_for_goals[goal] = valid_path
                edges_used.update(valid_path)
            else:
                valid_combination = False
                break
        
        if valid_combination:
            # Add both success and failure outcomes for this combination
            add_outcomes_for_paths(state, paths_for_goals, swappingOutcomes)
    
    return swappingOutcomes

def powerset(iterable):
    """Return all possible combinations of items"""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

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
    # Convert input state to dict for manipulation
    state_dict = dict(state)
    empty_edges = [edge for edge, age in state_dict.items() if age == -1]
    generationOutcomes = []
    
    for r in range(0, len(empty_edges) + 1):
        for edges_to_generate in combinations(empty_edges, r):
            new_state = state_dict.copy()
            
            for edge in edges_to_generate:
                new_state[edge] = 1
                
            # Convert to tuple format consistently
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
    
    # 1. Get swap outcomes
    swap_outcomes = getSwappingOutcomes(state, goalEdges)
    
    # 2. For each swap outcome, age the edges and attempt generation
    for outcome_state, prob, edges_per_goal, goals in swap_outcomes:
        action = {
            'swap_edges_per_goal': edges_per_goal,
            'goals_achieved': goals
        }
        
        # Age the edges
        aged_state = {}
        empty_edges = 0
        for edge, age in dict(outcome_state).items():
            if age > 0:
                if age >= maxAge:
                    aged_state[edge] = -1
                    empty_edges += 1
                else:
                    aged_state[edge] = age + 1
            else:
                aged_state[edge] = -1
                empty_edges += 1
        
        aged_state_tuple = tuple(sorted((tuple(sorted(edge)), age) for edge, age in aged_state.items()))
        if empty_edges == 0:
            transitions.append((action, (aged_state_tuple, prob)))
        else:
            gen_outcomes = getGenerationOutcomes(aged_state_tuple, pGen)
            for next_state, gen_prob in gen_outcomes:
                transitions.append((action, (next_state, prob * gen_prob)))
    
    return transitions
