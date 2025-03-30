from computeTransitionProb import getPossibleStates, getTransitionProbabilities
import matplotlib.pyplot as plt
import math
import random

# Initialize parameters
gamma = 0.99
epsilon = 0.001

initialEdges = [(0, 1), (1, 2), (2, 3)]
goalEdges = [(0, 2), (0, 3)]
pSwap = 0.9
pGen = 0.9
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
        current_edr = goal_success_counts[goal_tuple] / total_timesteps
        edr_history[goal_tuple].append(current_edr)
    
    if not action['goals_achieved']:
        return 0
    
    total_reward = 0
    edges_per_goal = action['swap_edges_per_goal']
    # MULTIPL GOAL BONUS TODO:
    # Calculate reward for achieved goals
    for goal in action['goals_achieved']:
        if goal in edges_per_goal:
            swaps_for_this_goal = len(edges_per_goal[goal]) - 1
            instant_rate = pSwap ** swaps_for_this_goal
            edr = max(0.001, goal_success_counts[goal] / total_timesteps)
            
            if instant_rate > 0 and edr > 0:
                total_reward += instant_rate / edr #+ math.log(instant_rate / edr) # +1 to avoid negative values from log (<1)
    return total_reward

# Get all possible states and initialize value function
allStates = getPossibleStates(initialEdges, maxAge)
# Add a debug print to see what states we have
print(f"Number of initial states: {len(allStates)}")

# Verify states are in the expected format
allStates = [tuple(sorted(dict(state).items())) for state in allStates]
V = {state: 0 for state in allStates}

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

# Value iteration loop
iteration = 0
deltas = []
print("Starting value iteration...")

while True:
    prev_V = V.copy()
    delta = 0
    
    # Add at the start of each iteration
    # Validate transition probabilities sum to 1
    for state in allStates:
        old_value = V[state]
        bestActionValue = float('-inf')
        
        for action in allActions[state]:
            action_dict = {
                'swap_edges_per_goal': {},
                'goals_achieved': [tuple(sorted(g)) for g in action[1]]
            }
            
            # Set up swap_edges_per_goal
            for goal in action[1]:
                action_dict['swap_edges_per_goal'][tuple(sorted(goal))] = [tuple(sorted(e)) for e in action[0]]
            
            immediateReward = get_reward(action_dict)
            futureValue = 0
            
            # Add detailed debug printing
            print(f"\nCalculating value for action: {action}")
            print(f"Immediate reward: {immediateReward}")
            
            transitions_debug = []
            for nextState, prob in allTransitions[(state, action)]:
                next_value = prev_V[nextState]
                contribution = prob * next_value
                futureValue += contribution
                transitions_debug.append(f"  State value: {next_value:.3f}, Prob: {prob:.3f}, Contribution: {contribution:.3f}")
            
            value = immediateReward + gamma * futureValue
            
            print("Transition details:")
            for debug_line in transitions_debug:
                print(debug_line)
            print(f"Total future value: {futureValue:.3f}")
            print(f"Discounted future value: {gamma * futureValue:.3f}")
            print(f"Final value: {value:.3f}")
            
            bestActionValue = max(bestActionValue, value)
        
        V[state] = bestActionValue if bestActionValue != float('-inf') else 0
        delta = max(delta, abs(V[state] - prev_V[state]))
    if V == prev_V:
        print('########################### IDENTICAL VALUES DETECTED')
    deltas.append(delta)
    iteration += 1

    if delta < epsilon:
        print(f"\nConverged after {iteration} iterations!")
        break
    if iteration > 1000:
        print("\nWarning: Maximum iterations reached without convergence")
        break
    
# Print final statistics
print("\n=== Final EDR Statistics ===")
for goal in goalEdges:
    final_edr = goal_success_counts[goal] / total_timesteps
    print(f"Goal {goal}: EDR = {final_edr:.6f}")
    print(f"  - Successes: {goal_success_counts[goal]}")
    print(f"  - Total timesteps: {total_timesteps}")

# Plot convergence
plt.figure(figsize=(10, 6))
plt.plot(deltas, 'b-')
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Delta (log scale)')
plt.title('Value Iteration Convergence')
plt.grid(True)
plt.show()

# Plot EDR evolution
plt.figure(figsize=(10, 6))
for goal in goalEdges:
    plt.plot(edr_history[goal], label=f'Goal {goal}')
plt.xlabel('Timestep')
plt.ylabel('EDR')
plt.title('EDR Evolution Over Time')
plt.legend()
plt.grid(True)
plt.ylim(0, 1)
plt.show()

# Plot distribution of state values
plt.figure(figsize=(10, 6))
plt.hist(list(V.values()), bins=50)
plt.xlabel('State Value')
plt.ylabel('Count')
plt.title('Distribution of State Values')
plt.grid(True)
plt.show()

# print("\n=== Final State Values ===")
# sorted_states = sorted(V.items(), key=lambda x: x[1], reverse=True)

# for state, value in sorted_states:
#     # Format state for readability
#     state_str = "State: "
#     for edge, age in state:
#         state_str += f"{edge}(age={age}), "
#     state_str = state_str[:-2]  # Remove trailing comma and space
    
#     print(f"\n{state_str}")
#     print(f"Value: {value:.6f}")

# Also add a summary of value statistics
values = list(V.values())
print("\n=== Value Statistics ===")
print(f"Maximum value: {max(values):.6f}")
print(f"Minimum value: {min(values):.6f}")
print(f"Average value: {sum(values)/len(values):.6f}")
print(f"Value range: {max(values) - min(values):.6f}")

# After value iteration converges, add this code
def get_optimal_policy(state, V, allActions, allTransitions, gamma):
    best_action = None
    best_value = float('-inf')
    
    for action in allActions[state]:
        action_dict = {
            'swap_edges_per_goal': {},
            'goals_achieved': [tuple(sorted(g)) for g in action[1]]
        }
        
        for goal in action[1]:
            action_dict['swap_edges_per_goal'][tuple(sorted(goal))] = [tuple(sorted(e)) for e in action[0]]
        
        immediateReward = get_reward(action_dict)
        futureValue = 0
        
        for nextState, prob in allTransitions[(state, action)]:
            futureValue += prob * V[nextState]
        
        value = immediateReward + gamma * futureValue
        if value > best_value:
            best_value = value
            best_action = action
    
    return best_action


def compute_bellman_error(V, allStates, allActions, allTransitions, gamma):
    max_error = 0
    for state in allStates:
        best_value = float('-inf')
        
        for action in allActions[state]:
            action_dict = {
                'swap_edges_per_goal': {},
                'goals_achieved': [tuple(sorted(g)) for g in action[1]]
            }
            
            for goal in action[1]:
                action_dict['swap_edges_per_goal'][tuple(sorted(goal))] = [tuple(sorted(e)) for e in action[0]]
            
            immediateReward = get_reward(action_dict)
            futureValue = 0
            
            for nextState, prob in allTransitions[(state, action)]:
                futureValue += prob * V[nextState]
            
            value = immediateReward + gamma * futureValue
            best_value = max(best_value, value)
        
        bellman_error = abs(V[state] - best_value)
        max_error = max(max_error, bellman_error)
    
    return max_error

# # Check Bellman error after convergence
# final_error = compute_bellman_error(V, allStates, allActions, allTransitions, gamma)
# print(f"\nFinal Bellman Error: {final_error}")
# # Should be close to or less than epsilon 

# print("\n=== Sample Optimal Actions ===")
# for state in list(V.keys()): 
#     optimal_action = get_optimal_policy(state, V, allActions, allTransitions, gamma)
#     print(f"\nState: {state}")
#     print(f"Optimal action: {optimal_action}")



print("\n=== Running Policy Simulation ===")
def get_greedy_action(state, allActions):
    """Compute what a greedy policy would do - choose action with highest immediate reward"""
    best_action = None
    best_reward = float('-inf')
    
    for action in allActions[state]:
        action_dict = {
            'swap_edges_per_goal': {},
            'goals_achieved': [tuple(sorted(g)) for g in action[1]]
        }
        
        for goal in action[1]:
            action_dict['swap_edges_per_goal'][tuple(sorted(goal))] = [tuple(sorted(e)) for e in action[0]]
        
        immediate_reward = get_reward(action_dict)
        
        if immediate_reward > best_reward:
            best_reward = immediate_reward
            best_action = action
    
    return best_action, best_reward

def simulate_policy(initial_state, num_steps=1000):
    current_state = initial_state
    total_reward = 0
    goals_achieved = {tuple(sorted(goal)): 0 for goal in goalEdges}
    different_choices = 0  # Counter for different choices
    
    # Track EDR over time for plotting
    edr_history = {tuple(sorted(goal)): [] for goal in goalEdges}
    
    for step in range(num_steps):
        # Get optimal action
        optimal_action = get_optimal_policy(current_state, V, allActions, allTransitions, gamma)
        if not optimal_action:
            print(f"\nStep {step + 1}: No valid action found")
            continue
        
        # Get greedy action
        greedy_action, greedy_reward = get_greedy_action(current_state, allActions)
        
        # Compare actions
        if optimal_action != greedy_action:
            different_choices += 1
            
            # Calculate full breakdown for optimal action
            optimal_dict = {
                'swap_edges_per_goal': {},
                'goals_achieved': [tuple(sorted(g)) for g in optimal_action[1]]
            }
            for goal in optimal_action[1]:
                optimal_dict['swap_edges_per_goal'][tuple(sorted(goal))] = [tuple(sorted(e)) for e in optimal_action[0]]
            optimal_immediate = get_reward(optimal_dict)
            
            optimal_future = 0
            optimal_transitions = []
            for nextState, prob in allTransitions[(current_state, optimal_action)]:
                contribution = prob * V[nextState]
                optimal_future += contribution
                optimal_transitions.append(f"    State value: {V[nextState]:.3f}, Prob: {prob:.3f}, Contribution: {contribution:.3f}")
            
            optimal_discounted = gamma * optimal_future
            optimal_total = optimal_immediate + optimal_discounted
            
            # Calculate full breakdown for greedy action
            greedy_future = 0
            greedy_transitions = []
            for nextState, prob in allTransitions[(current_state, greedy_action)]:
                contribution = prob * V[nextState]
                greedy_future += contribution
                greedy_transitions.append(f"    State value: {V[nextState]:.3f}, Prob: {prob:.3f}, Contribution: {contribution:.3f}")
            
            greedy_discounted = gamma * greedy_future
            greedy_total = greedy_reward + greedy_discounted
            
            print(f"\nStep {step + 1}: Optimal != Greedy")
            print(f"Current state: {current_state}")
            print(f"\nOptimal action: {optimal_action}")
            print(f"  Immediate reward: {optimal_immediate:.3f}")
            print(f"  Future value breakdown:")
            for t in optimal_transitions:
                print(t)
            print(f"  Total future value: {optimal_future:.3f}")
            print(f"  Discounted future value: {optimal_discounted:.3f}")
            print(f"  Final total value: {optimal_total:.3f}")
            
            print(f"\nGreedy action: {greedy_action}")
            print(f"  Immediate reward: {greedy_reward:.3f}")
            print(f"  Future value breakdown:")
            for t in greedy_transitions:
                print(t)
            print(f"  Total future value: {greedy_future:.3f}")
            print(f"  Discounted future value: {greedy_discounted:.3f}")
            print(f"  Final total value: {greedy_total:.3f}")
            
            print(f"\nDifference in total value: {optimal_total - greedy_total:.3f}")
            
        # Calculate reward
        action_dict = {
            'swap_edges_per_goal': {},
            'goals_achieved': [tuple(sorted(g)) for g in optimal_action[1]]
        }
        for goal in optimal_action[1]:
            action_dict['swap_edges_per_goal'][tuple(sorted(goal))] = [tuple(sorted(e)) for e in optimal_action[0]]
        
        reward = get_reward(action_dict)
        total_reward += reward
        
        # Track goals achieved and update EDR history
        for goal in goalEdges:
            goal_tuple = tuple(sorted(goal))
            if goal_tuple in optimal_action[1]:
                goals_achieved[goal_tuple] += 1
            current_edr = goals_achieved[goal_tuple] / (step + 1)
            edr_history[goal_tuple].append(current_edr)
        
        # Sample next state
        transitions = allTransitions[(current_state, optimal_action)]
        rand_val = random.random()
        cumulative_prob = 0
        
        for next_state, prob in transitions:
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                current_state = next_state
                break
    
    print("\n=== Simulation Results ===")
    print(f"Total steps: {num_steps}")
    print(f"Times optimal policy differed from greedy: {different_choices}")
    print(f"Percentage of different choices: {(different_choices/num_steps)*100:.2f}%")
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
    plt.title('EDR Evolution During Policy Simulation')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)
    plt.show()
    
    return edr_history, goals_achieved, total_reward

# Run simulation
initial_state = next(state for state in allStates 
                    if all(age == -1 for _, age in state))

edr_history, goals_achieved, total_reward = simulate_policy(initial_state, num_steps=10000)