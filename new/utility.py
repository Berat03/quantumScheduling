import random
import matplotlib.pyplot as plt

def getPossibleActions(state, goalEdges):
    actions = []
    existing_edges = {edge for edge, age in state if age >= 0}
    
    def find_path(start, end, path=None):
        if path is None:
            path = []
        if start == end:
            return [path]
        paths = []
        for edge in existing_edges:
            if edge not in path:
                if edge[0] == start:
                    new_paths = find_path(edge[1], end, path + [edge])
                    paths.extend(new_paths)
                elif edge[1] == start:
                    new_paths = find_path(edge[0], end, path + [edge])
                    paths.extend(new_paths)
        return paths
    
    for goal in goalEdges:
        start, end = goal
        paths = find_path(start, end)
        for path in paths:
            if len(path) > 0:  # Only add if we found a valid path
                actions.append((path, goal))
    
    # If no actions found, return list with empty action
    if not actions:
        return [([], None)]
    return actions

def performAction(action, state):
    consumed_edges, goal_edge = action
    new_state = state.copy()
    
    # For each edge in consumed_edges, find and update its age to -1
    for edge_to_consume in consumed_edges:
        for i, (edge, age) in enumerate(new_state):
            if edge == edge_to_consume:
                new_state[i] = (edge, -1)
                break
    
    return new_state

def ageEntanglements(state, maxAge):
    new_state = []
    for edge, age in state:
        if age >= 0:  # If entanglement exists
            new_age = age + 1
            if new_age > maxAge:
                new_state.append((edge, -1))  # Destroy if too old
            else:
                new_state.append((edge, new_age))  # Increment age
        else:
            new_state.append((edge, age))  # Keep non-existent entanglements as is
    return new_state

def generateEntanglement(state, pGen):
    new_state = []
    for edge, age in state:
        if age < 0:  # No entanglement exists
            if random.random() < pGen: 
                new_state.append((edge, 1)) 
            else:
                new_state.append((edge, age))  
        else:
            new_state.append((edge, age))  
    return new_state


class qTable:
    def __init__(self):
        self.q_values = {}  # Dictionary to store (state, action) -> Q-value
    
    def get_state_key(self, state):
        # Sort edges to ensure consistent representation
        sorted_state = sorted(state, key=lambda x: (x[0][0], x[0][1]))
        return tuple(tuple(item) for item in sorted_state)
    
    def get_action_key(self, action):
        # Convert action (consumed_edges, goal_edge) to a hashable tuple
        consumed_edges, goal_edge = action
        # Sort consumed edges to ensure consistent representation
        sorted_edges = tuple(sorted(consumed_edges))
        return (sorted_edges, goal_edge)
    
    def get_q_value(self, state, action):
        state_key = self.get_state_key(state)
        action_key = self.get_action_key(action)
        key = (state_key, action_key)
        return self.q_values.get(key, 0.0)
    
    def set_q_value(self, state, action, value):
        state_key = self.get_state_key(state)
        action_key = self.get_action_key(action)
        key = (state_key, action_key)
        self.q_values[key] = value

def plot_simulation_results(edr_history, goal_edges, total_timesteps):
    """
    Plot the simulation results.
    
    Args:
        edr_history: Dictionary mapping goals to their EDR history
        goal_edges: List of goal edges
        total_timesteps: Total number of timesteps
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot EDR evolution
    for goal in goal_edges:
        edr_values = edr_history[goal]
        ax1.plot(range(len(edr_values)), edr_values, label=f'Goal {goal}')
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('EDR')
    ax1.set_title('EDR Evolution Over Time')
    ax1.grid(True)
    ax1.legend()
    ax1.set_ylim(0, 1)
    
    # Plot final EDRs as bar chart
    final_edrs = [edr_history[goal][-1] for goal in goal_edges]
    goal_labels = [f'Goal {goal}' for goal in goal_edges]
    ax2.bar(goal_labels, final_edrs)
    ax2.set_xlabel('Goals')
    ax2.set_ylabel('Final EDR')
    ax2.set_title('Final EDRs by Goal')
    ax2.grid(True)
    
    # Add value labels on top of bars
    for i, v in enumerate(final_edrs):
        ax2.text(i, v, f'{v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def simulate_policy(Q_table, edges, goal_edges, p_swap, p_gen, max_age, num_steps):
    """
    Simulate a policy using a Q-table.
    
    Args:
        Q_table: qTable object containing Q(s,a) values
        edges: List of network edges
        goal_edges: List of goal edges
        p_swap: Swap success probability
        p_gen: Generation success probability
        max_age: Maximum age of entanglement
        num_steps: Number of simulation steps
    """
    # Initialize state and tracking variables
    current_state = [(edge, -1) for edge in edges]  # Start with no entanglements
    goal_success_counts = {goal: 0 for goal in goal_edges}
    total_timesteps = 1
    edr_history = {goal: [] for goal in goal_edges}  # Track EDR history
    
    # Simulation loop
    for step in range(num_steps):
        # Get possible actions
        possible_actions = getPossibleActions(current_state, goal_edges)
        
        # Choose best action from Q-table
        if possible_actions:
            # Get Q-values for each possible action
            action_q_values = [(action, Q_table.get_q_value(current_state, action)) for action in possible_actions]
            # Find action with maximum Q-value
            best_action = max(action_q_values, key=lambda x: x[1])[0]
        else:
            # If no actions available, use empty action
            best_action = ([], None)
        
        # Execute action and update state
        current_state = performAction(best_action, current_state)
        current_state = ageEntanglements(current_state, max_age)
        current_state = generateEntanglement(current_state, p_gen)
        
        # Check if any goals were achieved
        consumed_edges, goal = best_action
        if goal is not None and len(consumed_edges) > 0:
            # Check if the goal was achieved
            if random.random() < p_swap ** (len(consumed_edges) - 1):
                goal_success_counts[goal] += 1
        
        total_timesteps += 1
        
        # Update EDR history
        for goal in goal_edges:
            current_edr = goal_success_counts[goal] / total_timesteps
            edr_history[goal].append(current_edr)
        
        # Print progress
        if step % 1000 == 0:
            print(f"\nStep {step}:")
            for goal in goal_edges:
                current_edr = goal_success_counts[goal] / total_timesteps
                print(f"Goal {goal}: Current EDR = {current_edr:.6f}")
    
    # Print final statistics
    print(f"\n=== Final Statistics ===")
    print(f"Parameters: pSwap={p_swap}, pGen={p_gen}, maxAge={max_age}")
    print(f"Network: edges={edges}, goals={goal_edges}")
    print("\nFinal EDRs:")
    for goal in goal_edges:
        final_edr = goal_success_counts[goal] / total_timesteps
        print(f"Goal {goal}: EDR = {final_edr:.6f}")
        print(f"  - Successes: {goal_success_counts[goal]}")
        print(f"  - Total timesteps: {total_timesteps}")
    
    # Plot results
    plot_simulation_results(edr_history, goal_edges, total_timesteps)
    
    return goal_success_counts, total_timesteps, edr_history