from pre_computation import getPossibleStates, getTransitionProbabilities
import matplotlib.pyplot as plt
import numpy as np
import math

# Initialize parameters
gamma = 0.9
epsilon = 0.001
pSwap = 1
pGen = 0.9
initialEdges = [(1,2), (2,3), (3,4)]
goalEdges = [(1,4)]  # Make sure all possible goals are listed here
maxAge = 2
# Add these at the top of your file with other global variables
total_timesteps = 0  # This will count ALL actions taken
goal_success_counts = {goal: 0 for goal in goalEdges}
goal_attempt_counts = {goal: 0 for goal in goalEdges}  # Track all attempts
edr_history = {goal: [] for goal in goalEdges}  # Track EDR over time for each goal

# Add after your imports
DEBUG_REWARDS = False  # Toggle for reward calculation debugging
DEBUG_VALUES = False   # Toggle for value iteration debugging

# Add these constants at the top
DELTA_THRESHOLD = 1000  # When to start detailed debugging
PRINT_TOP_VALUES = 5    # Number of highest values to track

# Add these debug counters at the start
generation_attempts = 0
generation_successes = 0
swap_attempts = 0
swap_successes = 0

def get_reward(action):
    """
    Calculate reward as sum of log(instant_rate/average_rate) for each achieved goal
    """
    global total_timesteps, goal_success_counts, edr_history
    global generation_attempts, generation_successes, swap_attempts, swap_successes
    
    total_timesteps += 1
    
    # Debug print for significant actions
    if total_timesteps % 1000 == 0:  # Print every 1000 timesteps
        print(f"\nTimestep {total_timesteps} stats:")
        print(f"pGen = {pGen}, pSwap = {pSwap}")
        for goal in goalEdges:
            current_edr = goal_success_counts[goal] / total_timesteps
            print(f"Goal {goal}:")
            print(f"  EDR: {current_edr:.4f}")
            print(f"  Successes: {goal_success_counts[goal]}")
        print(f"Generation success rate: {generation_successes/generation_attempts if generation_attempts > 0 else 0:.4f}")
        print(f"Swap success rate: {swap_successes/swap_attempts if swap_attempts > 0 else 0:.4f}")
    
    if DEBUG_REWARDS:
        print(f"\nDEBUG Reward Calculation:")
        print(f"  Action: {action}")
        print(f"  Current timesteps: {total_timesteps}")
    
    # Update EDR history for all goals at each timestep
    for goal in goalEdges:
        current_edr = goal_success_counts[goal] / total_timesteps
        edr_history[goal].append(current_edr)
        
        if DEBUG_REWARDS:
            print(f"  Goal {goal} current EDR: {current_edr:.6f}")
    
    if not action['goals_achieved']:
        if DEBUG_REWARDS:
            print("  No goals achieved, returning 0")
        return 0
    
    total_reward = 0
    edges_per_goal = action['swap_edges_per_goal']
    
    # Calculate reward for achieved goals
    for goal in action['goals_achieved']:
        if goal in edges_per_goal:
            goal_success_counts[goal] += 1
            
            swaps_for_this_goal = len(edges_per_goal[goal]) - 1
            instant_rate = pSwap ** swaps_for_this_goal
            edr = goal_success_counts[goal] / total_timesteps
            
            if DEBUG_REWARDS:
                print(f"  Goal {goal}:")
                print(f"    Swaps: {swaps_for_this_goal}")
                print(f"    Instant rate: {instant_rate:.6f}")
                print(f"    EDR: {edr:.6f}")
            
            if instant_rate > 0 and edr > 0:
                reward = math.log(instant_rate / edr)
                if DEBUG_REWARDS:
                    print(f"    Reward: {reward:.6f}")
                total_reward += reward
    
    if DEBUG_REWARDS:
        print(f"  Total reward: {total_reward:.6f}")
    return total_reward


# Get all possible states
allStates = getPossibleStates(initialEdges)

# Initialize value function
V = {state: 0 for state in allStates}

# Get all possible actions for each state
allActions = {}
allTransitions = {}

# Precompute transitions for all states
for state in allStates:
    transitions = getTransitionProbabilities(state, pSwap, pGen, goalEdges, maxAge)
    
    # Group actions for this state
    actions_for_state = set()
    for ((_, action), _) in transitions:
        # Get all edges used across all goals
        all_edges = set()
        for edges in action['swap_edges_per_goal'].values():
            all_edges.update(edges)
        
        # Convert dict to tuple for hashability
        action_tuple = (
            tuple(sorted(all_edges)),
            tuple(sorted(action['goals_achieved']))
        )
        actions_for_state.add(action_tuple)
    
    allActions[state] = actions_for_state
    
    # Store transitions
    for (_, action), (next_state, prob) in transitions:
        # Get all edges used across all goals
        all_edges = set()
        for edges in action['swap_edges_per_goal'].values():
            all_edges.update(edges)
            
        action_tuple = (
            tuple(sorted(all_edges)),
            tuple(sorted(action['goals_achieved']))
        )
        key = (state, action_tuple)
        if key not in allTransitions:
            allTransitions[key] = []
        allTransitions[key].append((next_state, prob))

# Initialize list to store deltas
deltas = []

# Add this function to help with debugging
def debug_state_values(V, prev_V, iteration):
    """Print detailed information about the largest value changes"""
    # Calculate all deltas
    all_deltas = [(state, abs(V[state] - prev_V[state])) for state in V.keys()]
    # Sort by delta
    all_deltas.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n=== Detailed State Analysis (Iteration {iteration}) ===")
    print(f"Top {PRINT_TOP_VALUES} largest value changes:")
    for state, delta in all_deltas[:PRINT_TOP_VALUES]:
        print(f"\nState: {state}")
        print(f"  Previous value: {prev_V[state]:.2f}")
        print(f"  Current value: {V[state]:.2f}")
        print(f"  Delta: {delta:.2f}")

# Add this helper function to count empty edges
def count_empty_edges(state):
    return sum(1 for _, age in dict(state).items() if age == -1)

# Value iteration loop
iteration = 0
total_timesteps = 0
print("Starting value iteration...")

while True:
    prev_V = V.copy()
    delta = 0
    total_timesteps += len(allStates)
    
    # Track maximum values for this iteration
    max_immediate_reward = float('-inf')
    max_future_value = float('-inf')
    max_total_value = float('-inf')
    state_causing_max = None
    action_causing_max = None
    
    if DEBUG_VALUES:
        print(f"\nIteration {iteration}:")
    
    for state in allStates:
        bestActionValue = float('-inf')  # Initialize here, before the action loop
        
        for action in allActions[state]:
            action_dict = {
                'swap_edges_per_goal': {},
                'goals_achieved': list(action[1])
            }
            
            # Track swap statistics
            if action[0]:  # if there are edges being used
                swap_attempts += 1
                if action[1]:  # if goals were achieved
                    swap_successes += 1
            
            # Track generation statistics for each transition
            for nextState, prob in allTransitions[(state, action)]:
                empty_edges_after = count_empty_edges(nextState)
                edges_generated = empty_edges_before - empty_edges_after
                
                if empty_edges_before > 0:  # Only count if generation was possible
                    generation_attempts += 1
                    if edges_generated > 0:
                        generation_successes += 1
                        
                print(f"\nGeneration Debug:")
                print(f"State before: {state}")
                print(f"State after: {nextState}")
                print(f"Empty edges before: {empty_edges_before}")
                print(f"Empty edges after: {empty_edges_after}")
                print(f"Edges generated: {edges_generated}")
                print(f"Probability: {prob}")
            
            for goal in action[1]:
                relevant_edges = [edge for edge in action[0] if edge in allTransitions[(state, action)][0][0]]
                action_dict['swap_edges_per_goal'][goal] = relevant_edges
            
            immediateReward = get_reward(action_dict)
            futureValue = 0
            
            for nextState, prob in allTransitions[(state, action)]:
                futureValue += prob * prev_V[nextState]
            
            value = immediateReward + gamma * futureValue
            
            if value > bestActionValue:  # Update best action value
                bestActionValue = value
            
            # Track maximum values
            if immediateReward > max_immediate_reward:
                max_immediate_reward = immediateReward
            if futureValue > max_future_value:
                max_future_value = futureValue
            if value > max_total_value:
                max_total_value = value
                state_causing_max = state
                action_causing_max = action
            
            if DEBUG_VALUES and delta > 1000:
                print(f"    Action: {action}")
                print(f"    Immediate Reward: {immediateReward:.6f}")
                print(f"    Future Value: {futureValue:.6f}")
                print(f"    Total Value: {value:.6f}")
            
        V[state] = bestActionValue
        new_delta = abs(bestActionValue - prev_V[state])
        delta = max(delta, new_delta)
    
    deltas.append(delta)
    iteration += 1
    
    # Enhanced progress printing
    if iteration % 100 == 0 or delta > DELTA_THRESHOLD:
        print(f"\nIteration {iteration}:")
        print(f"Delta: {delta:.6f}")
        print(f"Maximum values this iteration:")
        print(f"  Immediate Reward: {max_immediate_reward:.2f}")
        print(f"  Future Value: {max_future_value:.2f}")
        print(f"  Total Value: {max_total_value:.2f}")
        print(f"State causing max value: {state_causing_max}")
        print(f"Action causing max value: {action_causing_max}")
        
        if delta > DELTA_THRESHOLD:
            print("\nWARNING: Delta is growing large!")
            debug_state_values(V, prev_V, iteration)
            
            # Print value distribution statistics
            values = list(V.values())
            print("\nValue Distribution:")
            print(f"  Min: {min(values):.2f}")
            print(f"  Max: {max(values):.2f}")
            print(f"  Mean: {sum(values)/len(values):.2f}")
            print(f"  Number of states: {len(values)}")
    
    if delta < epsilon:
        print(f"\nConverged after {iteration} iterations!")
        break
    
    if iteration > 1000:
        print("\nWarning: Maximum iterations reached without convergence")
        break

# Plot convergence
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(deltas) + 1), deltas, 'b-')
plt.plot(range(1, len(deltas) + 1), deltas, 'r.')
plt.axhline(y=epsilon, color='g', linestyle='--', label='Convergence threshold')
plt.yscale('log')  # Use log scale for better visualization
plt.xlabel('Iteration')
plt.ylabel('Maximum Delta (log scale)')
plt.title('Value Iteration Convergence')
plt.legend(['Delta', 'Points', 'Convergence threshold'])
plt.grid(True)
plt.show()

# Print some statistics
print(f'Total iterations: {iteration}')
print(f'Final delta: {deltas[-1]:.6f}')
print(f'Initial delta: {deltas[0]:.6f}')

# Add this after your existing plotting code
plt.figure(figsize=(10, 6))
for goal in goalEdges:
    plt.plot(range(1, len(edr_history[goal]) + 1), 
             edr_history[goal],
             label=f'Goal {goal}')

plt.xlabel('Timestep')
plt.ylabel('EDR')
plt.title('EDR Evolution Over Time')
plt.legend()
plt.grid(True)
plt.ylim(0, 1)  # Set y-axis limits between 0 and 1
plt.show()

# Print final EDR statistics
print("\n=== Final EDR Statistics ===")
for goal in goalEdges:
    final_edr = goal_success_counts[goal] / total_timesteps  # Same denominator for all goals
    print(f"Goal {goal}: EDR = {final_edr:.6f}")
    print(f"  - Successes: {goal_success_counts[goal]}")
    print(f"  - Total timesteps: {total_timesteps}")

# At the end, print more detailed statistics
print("\n=== Final Statistics ===")
print(f"Parameters: pGen = {pGen}, pSwap = {pSwap}")

print("\nGeneration Statistics:")
print(f"Total generation attempts: {generation_attempts}")
print(f"Successful generations: {generation_successes}")
print(f"Generation success rate: {generation_successes/generation_attempts if generation_attempts > 0 else 0:.4f}")
print(f"Expected generation rate (pGen): {pGen}")

print("\nSwap Statistics:")
print(f"Total swap attempts: {swap_attempts}")
print(f"Successful swaps: {swap_successes}")
print(f"Swap success rate: {swap_successes/swap_attempts if swap_attempts > 0 else 0:.4f}")
print(f"Expected swap rate (pSwap): {pSwap}")

print("\nEDR Statistics:")
for goal in goalEdges:
    final_edr = goal_success_counts[goal] / total_timesteps
    print(f"Goal {goal}:")
    print(f"  EDR: {final_edr:.6f}")
    print(f"  Successes: {goal_success_counts[goal]}")
    print(f"  Total timesteps: {total_timesteps}")


