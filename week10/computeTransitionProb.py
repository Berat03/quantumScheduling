from collections import defaultdict
from itertools import product
from itertools import combinations


def getPossibleStates(initialEdges, maxAge):
    """
    Generate all possible states considering all possible age combinations
    Ages can be: -1 (no entanglement), 1 to maxAge
    """
    edges = [tuple(sorted(edge)) for edge in initialEdges]
    possible_ages = [-1] + list(range(1, maxAge + 1))  # [-1, 1, 2] for maxAge=2
    print(f"Possible ages: {possible_ages}")  # Debug print
    
    all_states = []
    # Use product to generate all possible combinations of ages
    for age_combination in product(possible_ages, repeat=len(edges)):
        new_state = {}
        for edge, age in zip(edges, age_combination):
            new_state[edge] = age
        # Convert to sorted tuple format for consistency
        state_tuple = tuple(sorted(new_state.items()))
        all_states.append(state_tuple)
    
    print(f"Generated {len(all_states)} states")  # Debug print
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
            if next_node not in path:  # Avoid cycles, are we sure?
                yield from find_all_paths(next_node, end, available, path + [next_node])

def getSwappingOutcomes(state, pSwap, goalEdges):
    # Convert input state to dict for easier manipulation
    state_dict = dict(state)
    
    goal_paths = defaultdict(list)
    for goal in goalEdges:
        available = [e for e, age in state_dict.items() if age > 0]
        all_paths = list(find_all_paths(goal[0], goal[1], available))
        for path in all_paths:
            if len(path) > 1:
                edges = [tuple(sorted([path[i], path[i+1]])) for i in range(len(path)-1)]
                if all(edge in state_dict and state_dict[edge] > 0 for edge in edges):
                    goal_paths[goal].append(edges)

    # If no possible paths, return original state with probability 1.0
    if not any(paths for paths in goal_paths.values()):
        return [(state, 1.0, {}, [])]

    swappingOutcomes = []
    
    # Collect swap possibilities
    swap_possibilities = []
    for r in range(1, len(goal_paths) + 1):
        for goals in combinations(goal_paths.keys(), r):
            for path_combination in product(*[goal_paths[goal] for goal in goals]):
                edges_per_goal = {}
                for goal, edges in zip(goals, path_combination):
                    edges_per_goal[goal] = edges
                
                needed_edges = set()
                for edges in path_combination:
                    needed_edges.update(edges)
                
                new_state = state_dict.copy()
                for edge in needed_edges:
                    new_state[edge] = -1
                    
                num_swaps = sum(len(edges) - 1 for edges in path_combination)
                # Convert to tuple format consistently
                new_state_tuple = tuple(sorted(new_state.items()))
                
                swap_possibilities.append((new_state_tuple, edges_per_goal, list(goals), num_swaps))

    # Calculate probabilities for swaps
    for new_state, edges_per_goal, goals, num_swaps in swap_possibilities:
        # Success case
        success_prob = pSwap ** num_swaps
        swappingOutcomes.append((new_state, success_prob, edges_per_goal, goals))
        
        # Failure case
        fail_prob = (1 - pSwap ** num_swaps)
        swappingOutcomes.append((state, fail_prob, edges_per_goal, []))

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
    swap_outcomes = getSwappingOutcomes(state, pSwap, goalEdges)
    
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
        if len(aged_state_tuple) > 1:
            print(f"Sample next state from transitions: {aged_state_tuple}")  # Debug print
        
        if empty_edges == 0:
            transitions.append((action, (aged_state_tuple, prob)))
        else:
            gen_outcomes = getGenerationOutcomes(aged_state_tuple, pGen)
            for next_state, gen_prob in gen_outcomes:
                transitions.append((action, (next_state, prob * gen_prob)))
    
    return transitions

def test_transition_probabilities():
    # Define a test case with 4 edges forming a network
    initial_edges = [(0, 1), (1, 2), (2, 3), (3, 4)]  # Four edges forming a path 0-1-2-3-4
    goal_edges = [(2, 4)]  # Want to connect 3 to 5
    
    # Test parameters
    pSwap = 0.8  # High probability of successful swap
    pGen = 0.3   # Lower probability of generation
    maxAge = 2
    
    initial_state = tuple(sorted({
        (0, 1): 1,
        (1, 2): 11,
        (2, 3): 1,
        (3, 4): 1
    }.items()))
    
    print("=== Test Configuration ===")
    print(f"Initial edges: {initial_edges}")
    print(f"Goal edges: {goal_edges}")
    print(f"pSwap: {pSwap}, pGen: {pGen}, maxAge: {maxAge}")
    print(f"Initial state: {dict(initial_state)}\n")
    
    transitions = getTransitionProbabilities(initial_state, pSwap, pGen, goal_edges, maxAge)
    
    print("\n=== All Possible Transitions ===")
    for i, (action, (next_state, prob)) in enumerate(transitions, 1):
        print(f"\nTransition {i}:")
        print(f"Action:")
        print(f"  Swap edges per goal: {action['swap_edges_per_goal']}")
        print(f"  Goals achieved: {action['goals_achieved']}")
        print(f"Next state: {dict(next_state)}")
        print(f"Probability: {prob:.6f}")

if __name__ == "__main__":
    test_transition_probabilities()


