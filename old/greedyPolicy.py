from computeTransitionProb import getPossibleStates, getTransitionProbabilities
import matplotlib.pyplot as plt
import math
import random

# Initialize parameters
pSwap = 0.8
pGen = 0.8
initialEdges = [(0,1), (1, 3), (2, 3), (3, 4), (4, 5), (4,6)]
goalEdges = [(0,5), (3,6)]
maxAge = 2


# Global tracking variables
total_timesteps = 0
goal_success_counts = {tuple(sorted(goal)): 0 for goal in goalEdges}
edr_history = {tuple(sorted(goal)): [] for goal in goalEdges}

def get_reward(action):
    """Calculate reward as sum of log(instant_rate/average_rate) for each achieved goal"""
    global total_timesteps, goal_success_counts, edr_history
    
    total_timesteps += 1
    
    # Update EDR history for all goals at each timestep
    for goal in goalEdges:
        goal_tuple = tuple(sorted(goal))
        if goal_tuple in action['goals_achieved']:
            goal_success_counts[goal_tuple] += 1
        current_edr = max(0.001, goal_success_counts[goal_tuple] / total_timesteps)
        edr_history[goal_tuple].append(current_edr)
    
    if not action['goals_achieved']:
        return 0
    
    total_reward = 0
    edges_per_goal = action['swap_edges_per_goal']
    
    # Calculate reward for achieved goals
    for goal in action['goals_achieved']:
        if goal in edges_per_goal:
            swaps_for_this_goal = len(edges_per_goal[goal]) - 1
            instant_rate = pSwap ** swaps_for_this_goal
            edr = goal_success_counts[goal] / total_timesteps
            
            if instant_rate > 0 and edr > 0:
                total_reward += math.log(instant_rate / edr)
    
    return total_reward

# Get all possible states and initialize
allStates = getPossibleStates(initialEdges, maxAge)
print(f"Number of initial states: {len(allStates)}")
allStates = [tuple(sorted(dict(state).items())) for state in allStates]

# Get all possible actions for each state
allActions = {}
allTransitions = {}

# Precompute transitions for all states
for state in allStates:
    transitions = getTransitionProbabilities(state, pSwap, pGen, goalEdges, maxAge)
    
    # Group actions for this state
    actions_for_state = set()
    for (action, (next_state, prob)) in transitions:
        all_edges = set()
        for edges in action['swap_edges_per_goal'].values():
            all_edges.update(tuple(sorted(edge)) for edge in edges)
        
        action_tuple = (
            tuple(sorted(all_edges)),
            tuple(sorted(tuple(sorted(g)) for g in action['goals_achieved']))
        )
        actions_for_state.add(action_tuple)
    
    allActions[state] = actions_for_state
    
    # Store transitions
    for (action, (next_state, prob)) in transitions:
        all_edges = set()
        for edges in action['swap_edges_per_goal'].values():
            all_edges.update(tuple(sorted(edge)) for edge in edges)
            
        action_tuple = (
            tuple(sorted(all_edges)),
            tuple(sorted(tuple(sorted(g)) for g in action['goals_achieved']))
        )
        key = (state, action_tuple)
        if key not in allTransitions:
            allTransitions[key] = []
        allTransitions[key].append((next_state, prob))

def get_greedy_action(state):
    """Choose action with highest immediate reward"""
    best_action = None
    best_reward = float('-inf')
    
    for action in allActions[state]:
        action_dict = {
            'swap_edges_per_goal': {},
            'goals_achieved': [tuple(sorted(g)) for g in action[1]]
        }
        
        for goal in action[1]:
            action_dict['swap_edges_per_goal'][tuple(sorted(goal))] = [tuple(sorted(e)) for e in action[0]]
        
        reward = get_reward(action_dict)
        if reward > best_reward:
            best_reward = reward
            best_action = action
    
    return best_action

def simulate_greedy_policy(initial_state, num_steps=1000):
    current_state = initial_state
    total_reward = 0
    goals_achieved = {tuple(sorted(goal)): 0 for goal in goalEdges}
    
    # Track EDR over time for plotting
    edr_history = {tuple(sorted(goal)): [] for goal in goalEdges}
    
    print(f"Starting state:")
    for edge, age in current_state:
        print(f"  {edge}(age={age})")
    
    for step in range(num_steps):
        # Get greedy action
        greedy_action = get_greedy_action(current_state)
        if not greedy_action:
            print(f"\nStep {step + 1}: No valid action found")
            continue
            
        # Print current action
        print(f"\nStep {step + 1}:")
        print(f"Action: Swap edges {greedy_action[0]}")
        print(f"Goals attempted: {greedy_action[1]}")
        
        # Calculate reward
        action_dict = {
            'swap_edges_per_goal': {},
            'goals_achieved': [tuple(sorted(g)) for g in greedy_action[1]]
        }
        for goal in greedy_action[1]:
            action_dict['swap_edges_per_goal'][tuple(sorted(goal))] = [tuple(sorted(e)) for e in greedy_action[0]]
        
        reward = get_reward(action_dict)
        total_reward += reward
        
        # Track goals achieved and update EDR history
        for goal in goalEdges:
            goal_tuple = tuple(sorted(goal))
            if goal_tuple in greedy_action[1]:
                goals_achieved[goal_tuple] += 1
            current_edr = goals_achieved[goal_tuple] / (step + 1)
            edr_history[goal_tuple].append(current_edr)
        
        # Sample next state
        transitions = allTransitions[(current_state, greedy_action)]
        rand_val = random.random()
        cumulative_prob = 0
        
        for next_state, prob in transitions:
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                current_state = next_state
                break
    
    print("\n=== Simulation Results ===")
    print(f"Total steps: {num_steps}")
    print(f"Total reward: {total_reward:.3f}")
    print(f"Average reward per step: {total_reward/num_steps:.3f}")
    print("\nGoals achieved:")
    for goal, count in goals_achieved.items():
        print(f"  {goal}: {count} times (EDR: {count/num_steps:.3f})")
    
    # Plot EDR evolution
    plt.figure(figsize=(10, 6))
    for goal in goalEdges:
        goal_tuple = tuple(sorted(goal))
        plt.plot(edr_history[goal_tuple], label=f'Goal {goal}')
    plt.xlabel('Timestep')
    plt.ylabel('EDR')
    plt.title('EDR Evolution During Greedy Policy')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)
    plt.show()
    
    return edr_history, goals_achieved, total_reward

# Run simulation
initial_state = next(state for state in allStates 
                    if all(age == -1 for _, age in state))

edr_history, goals_achieved, total_reward = simulate_greedy_policy(initial_state, num_steps=100000)