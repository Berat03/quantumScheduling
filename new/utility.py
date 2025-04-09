import random
import matplotlib.pyplot as plt
import math
import numpy as np

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

def jains_index(edrs):
    """Compute Jain's Fairness Index."""
    if all(edr == 0 for edr in edrs.values()):
        return 0.0
    numerator = sum(edrs.values())**2
    denominator = len(edrs) * sum(v**2 for v in edrs.values())
    return numerator / denominator if denominator > 0 else 0.0

def min_max_fairness(edrs):
    """Compute Min-Max Fairness (min / max)."""
    values = list(edrs.values())
    min_val, max_val = min(values), max(values)
    if max_val == 0:
        return 0.0
    return min_val / max_val

def simulate_policy(Q_table, edges, goal_edges, p_swap, p_gen, max_age, num_steps, plot=True):
    current_state = [(edge, -1) for edge in edges]  # Start with no entanglements
    goal_success_counts = {goal: 0 for goal in goal_edges}
    total_timesteps = 1
    edr_history = {goal: [] for goal in goal_edges}
    
    # Fairness tracking
    jain_history = []
    min_max_history = []

    for step in range(num_steps):
        possible_actions = getPossibleActions(current_state, goal_edges)
        if possible_actions:
            action_q_values = [(action, Q_table.get_q_value(current_state, action)) for action in possible_actions]
            best_action = max(action_q_values, key=lambda x: x[1])[0]
        else:
            best_action = ([], None)

        current_state = performAction(best_action, current_state)
        current_state = ageEntanglements(current_state, max_age)
        current_state = generateEntanglement(current_state, p_gen)

        consumed_edges, goal = best_action
        if goal is not None and len(consumed_edges) > 0:
            if random.random() < p_swap ** (len(consumed_edges) - 1):
                goal_success_counts[goal] += 1

        total_timesteps += 1

        # Update EDR history
        current_edrs = {}
        for goal in goal_edges:
            current_edr = goal_success_counts[goal] / total_timesteps
            edr_history[goal].append(current_edr)
            current_edrs[goal] = current_edr
        
        # Update fairness metrics
        jain_val = jains_index(current_edrs)
        min_max_val = min_max_fairness(current_edrs)
        jain_history.append(jain_val)
        min_max_history.append(min_max_val)
    # Plot
    if plot:
        fig, axs = plt.subplots(3, 1, figsize=(12, 12))

        # Plot EDRs
        for goal in goal_edges:
            axs[0].plot(edr_history[goal], label=f'Goal {goal}')
        axs[0].set_title('EDR Evolution Over Time')
        axs[0].set_xlabel('Timestep')
        axs[0].set_ylabel('EDR')
        axs[0].legend()
        axs[0].grid(True)
        axs[0].set_ylim(0, 1)

        # Plot Jain's Index
        axs[1].plot(jain_history, color='purple')
        axs[1].set_title("Jain's Fairness Index Over Time")
        axs[1].set_xlabel('Timestep')
        axs[1].set_ylabel('Jain Index')
        axs[1].grid(True)
        axs[1].set_ylim(0, 1.05)

        # Plot Min-Max Fairness
        axs[2].plot(min_max_history, color='green')
        axs[2].set_title("Min-Max Fairness Over Time")
        axs[2].set_xlabel('Timestep')
        axs[2].set_ylabel('Min / Max EDR')
        axs[2].grid(True)
        axs[2].set_ylim(0, 1.05)

        plt.tight_layout()
        plt.show()

    return goal_success_counts, total_timesteps, edr_history, jain_history, min_max_history

def validate_policy_simulation(Q_table, edges, goal_edges, p_swap, p_gen, max_age, num_steps, num_simulations, seed=27, plot=True, ):
    """
    Validate the policy by running multiple simulations and plot EDR, Jain’s Index, and Min-Max fairness.

    Returns:
        (mean_final_edr, mean_final_jain, mean_final_min_max)
    """
    all_edr_histories = []
    all_jain_histories = []
    all_min_max_histories = []

    final_edr_means = []
    final_jains = []
    final_min_max = []

    for sim in range(num_simulations):
        random.seed(seed + sim) 
        _, _, edr_history, jain_history, min_max_history = simulate_policy(
            Q_table, edges, goal_edges, p_swap, p_gen, max_age, num_steps, plot=False
        )

        all_edr_histories.append(edr_history)
        all_jain_histories.append(jain_history)
        all_min_max_histories.append(min_max_history)

        # Mean final EDR across all goals
        final_edr_per_goal = [edr_history[goal][-1] for goal in goal_edges]
        final_edr_means.append(np.mean(final_edr_per_goal))

        # Final Jain and Min-Max values
        final_jains.append(jain_history[-1])
        final_min_max.append(min_max_history[-1])

    mean_final_edr = np.mean(final_edr_means)
    mean_final_jain = np.mean(final_jains)
    mean_final_min_max = np.mean(final_min_max)

    if plot:
        # === Plot: EDR ===
        fig, ax = plt.subplots(figsize=(12, 5))
        for sim, edr_history in enumerate(all_edr_histories):
            for goal in goal_edges:
                ax.plot(range(len(edr_history[goal])), edr_history[goal], alpha=0.3, label=f"Sim {sim+1}, Goal {goal}")
        ax.axhline(mean_final_edr, color='black', linestyle='--', linewidth=2, label=f'Mean Final EDR = {mean_final_edr:.4f}')
        ax.set_title("EDR Evolution Over Multiple Simulations")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("EDR")
        ax.grid(True)
        ax.set_ylim(0, 1)
        ax.legend()
        plt.tight_layout()
        plt.show()

        # === Plot: Jain's Fairness ===
        fig, ax = plt.subplots(figsize=(12, 5))
        for sim, jain_values in enumerate(all_jain_histories):
            ax.plot(range(len(jain_values)), jain_values, alpha=0.4, label=f"Sim {sim+1}")
        ax.axhline(mean_final_jain, color='black', linestyle='--', linewidth=2, label=f'Mean Final Jain = {mean_final_jain:.4f}')
        ax.set_title("Jain's Fairness Index Over Time")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Jain’s Index")
        ax.grid(True)
        ax.set_ylim(0, 1.01)
        ax.legend()
        plt.tight_layout()
        plt.show()

        # === Plot: Min-Max Fairness ===
        fig, ax = plt.subplots(figsize=(12, 5))
        for sim, min_max_values in enumerate(all_min_max_histories):
            ax.plot(range(len(min_max_values)), min_max_values, alpha=0.4, label=f"Sim {sim+1}")
        ax.axhline(mean_final_min_max, color='black', linestyle='--', linewidth=2, label=f'Mean Final Min-Max = {mean_final_min_max:.4f}')
        ax.set_title("Min-Max Fairness Over Time")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Min / Max EDR")
        ax.grid(True)
        ax.set_ylim(0, 1.01)
        ax.legend()
        plt.tight_layout()
        plt.show()

    # Compute mean final EDR per goal across all simulations
    mean_final_edrs_by_goal = {
        goal: np.mean([edr_history[goal][-1] for edr_history in all_edr_histories])
        for goal in goal_edges
    }

    return mean_final_edrs_by_goal, mean_final_jain, mean_final_min_max

