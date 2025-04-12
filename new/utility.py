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
    current_state = [(edge, -1) for edge in edges]
    goal_success_counts = {goal: 0 for goal in goal_edges}
    total_timesteps = 1
    edr_history = {goal: [] for goal in goal_edges}

    jain_history = []
    min_max_history = []
    max_min_history = []
    throughput_history = []

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

        current_edrs = {}
        for goal in goal_edges:
            edr = goal_success_counts[goal] / total_timesteps
            edr_history[goal].append(edr)
            current_edrs[goal] = edr

        throughput = sum(current_edrs.values())
        fairness = jains_index(current_edrs)
        minmax = min_max_fairness(current_edrs)
        maxmin = min(current_edrs.values())

        throughput_history.append(throughput)
        jain_history.append(fairness)
        min_max_history.append(minmax)
        max_min_history.append(maxmin)

    if plot:
        fig, axs = plt.subplots(5, 1, figsize=(12, 20))

        for goal in goal_edges:
            axs[0].plot(edr_history[goal], label=f'Goal {goal}')
        axs[0].set_title('EDR Evolution Over Time')
        axs[0].set_xlabel('Timestep')
        axs[0].set_ylabel('EDR')
        axs[0].legend()
        axs[0].grid(True)
        axs[0].set_ylim(0, 1)

        axs[1].plot(jain_history, color='purple')
        axs[1].set_title("Jain's Fairness Index Over Time")
        axs[1].set_xlabel('Timestep')
        axs[1].set_ylabel("Jain's Index")
        axs[1].grid(True)
        axs[1].set_ylim(0, 1.05)

        axs[2].plot(min_max_history, color='green')
        axs[2].set_title("Min-Max Fairness Over Time")
        axs[2].set_xlabel('Timestep')
        axs[2].set_ylabel("Min / Max EDR")
        axs[2].grid(True)
        axs[2].set_ylim(0, 1.05)

        axs[3].plot(max_min_history, color='blue')
        axs[3].set_title("Max-Min Fairness Over Time")
        axs[3].set_xlabel('Timestep')
        axs[3].set_ylabel("Min Goal EDR")
        axs[3].grid(True)
        axs[3].set_ylim(0, 1.05)

        axs[4].plot(throughput_history, jain_history, color='darkred', alpha=0.8)
        axs[4].set_title("Pareto Curve: Throughput vs Jain's Fairness")
        axs[4].set_xlabel("Total Throughput (Sum of EDRs)")
        axs[4].set_ylabel("Jain's Index")
        axs[4].grid(True)
        axs[4].set_xlim(0, max(throughput_history) * 1.1)
        axs[4].set_ylim(0, 1.05)

        plt.tight_layout()
        plt.show()

    return goal_success_counts, total_timesteps, edr_history, jain_history, min_max_history, throughput_history, max_min_history

def validate_policy_simulation(Q_table, edges, goal_edges, p_swap, p_gen, max_age, num_steps, num_simulations, seed=27, plot=True, window=1000):
    all_edr_histories = []
    all_jain_histories = []
    all_min_max_histories = []
    all_max_min_histories = []
    all_throughput_histories = []

    final_edr_means = []
    final_jains = []
    final_min_max = []
    final_max_min = []

    for sim in range(num_simulations):
        random.seed(seed + sim)
        _, _, edr_history, jain_history, min_max_history, throughput_history, max_min_history = simulate_policy(
            Q_table, edges, goal_edges, p_swap, p_gen, max_age, num_steps, plot=False
        )

        all_edr_histories.append(edr_history)
        all_jain_histories.append(jain_history)
        all_min_max_histories.append(min_max_history)
        all_max_min_histories.append(max_min_history)
        all_throughput_histories.append(throughput_history)

        final_edr_per_goal = {
            goal: np.mean(edr_history[goal][-window:]) for goal in goal_edges
        }
        total_throughput = sum(final_edr_per_goal.values())
        final_jain = jains_index(final_edr_per_goal)
        final_minmax = min_max_fairness(final_edr_per_goal)
        final_maxmin = min(final_edr_per_goal.values())

        final_edr_means.append(total_throughput)
        final_jains.append(final_jain)
        final_min_max.append(final_minmax)
        final_max_min.append(final_maxmin)

    mean_final_edr = np.mean(final_edr_means)
    mean_final_jain = np.mean(final_jains)
    mean_final_min_max = np.mean(final_min_max)
    mean_final_max_min = np.mean(final_max_min)

    mean_final_edrs_by_goal = {
        goal: np.mean([np.mean(edr_history[goal][-window:]) for edr_history in all_edr_histories])
        for goal in goal_edges
    }

    return (
        mean_final_edrs_by_goal,
        mean_final_jain,
        mean_final_min_max,
        mean_final_max_min,
        all_throughput_histories,
        all_jain_histories
    )

def run_policy_experiments(
    train_policy_fn,
    policy_name,
    edges,
    goal_edges,
    p_swap,
    p_gen,
    max_age,
    num_runs=10,
    num_steps=30000,
    num_simulations=20,
    train_kwargs={},
    validate_kwargs={},
    plot=False
    ):
    
    final_edrs_by_goal = {goal: [] for goal in goal_edges}
    final_jains = []
    final_minmax = []
    final_maxmin = []
    all_throughput_histories = []
    all_jain_histories = []

    for seed in range(num_runs):
        print(f"\n=== {policy_name} Policy Training Run {seed + 1} ===")
        random.seed(seed)
        np.random.seed(seed)

        Q_table = train_policy_fn(
            edges=edges,
            goal_edges=goal_edges,
            p_swap=p_swap,
            p_gen=p_gen,
            max_age=max_age,
            seed=seed,
            **train_kwargs
        )

        results = validate_policy_simulation(
            Q_table=Q_table,
            edges=edges,
            goal_edges=goal_edges,
            p_swap=p_swap,
            p_gen=p_gen,
            max_age=max_age,
            num_steps=num_steps,
            num_simulations=num_simulations,
            plot=False,
            seed=seed,
            **validate_kwargs
        )

        mean_final_edrs_by_goal, mean_final_jain, mean_final_min_max, mean_final_max_min, throughput_histories, jain_histories = results

        for goal in goal_edges:
            final_edrs_by_goal[goal].append(mean_final_edrs_by_goal[goal])

        final_jains.append(mean_final_jain)
        final_minmax.append(mean_final_min_max)
        final_maxmin.append(mean_final_max_min)
        all_throughput_histories.append(throughput_histories)
        all_jain_histories.append(jain_histories)

    # === Optional Plotting ===
    if plot:
        print('PLOTTING')
        num_policies = num_runs
        throughputs = [
            sum(final_edrs_by_goal[goal][i] for goal in goal_edges)
            for i in range(num_policies)
        ]

        fig, axs = plt.subplots(1, 3, figsize=(21, 5))

        # --- Plot 1: EDR per goal ---
        for goal in goal_edges:
            axs[0].plot(final_edrs_by_goal[goal], marker='o', label=f"Goal {goal}")
        axs[0].set_title("Mean Final EDR per Goal")
        axs[0].set_xlabel("Policy Run")
        axs[0].set_ylabel("Final EDR")
        axs[0].set_ylim(0, 1)
        axs[0].legend()
        axs[0].grid(True)

        # --- Plot 2: Fairness Metrics ---
        axs[1].plot(final_jains, marker='o', label="Jain's Index", color='purple')
        axs[1].plot(final_minmax, marker='s', label='Min-Max Fairness', color='green')
        axs[1].plot(final_maxmin, marker='^', label='Max-Min Fairness', color='blue')
        axs[1].set_title("Fairness Metrics Across Policies")
        axs[1].set_xlabel("Policy Run")
        axs[1].set_ylabel("Fairness Value")
        axs[1].set_ylim(0, 1.05)
        axs[1].legend()
        axs[1].grid(True)

        # --- Plot 3: Pareto Front ---
        axs[2].scatter(throughputs, final_jains, color='darkred', alpha=0.7, s=60)
        axs[2].set_title("Pareto Front: Throughput vs Jain's Index")
        axs[2].set_xlabel("Total Throughput (Sum of EDRs)")
        axs[2].set_ylabel("Jain's Index")
        axs[2].set_xlim(0, max(throughputs) + 0.1)
        axs[2].set_ylim(0, 1.05)
        axs[2].grid(True)

        plt.suptitle(f"{policy_name} Policy Results", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    return {
        "edrs": final_edrs_by_goal,
        "jains": final_jains,
        "minmax": final_minmax,
        "maxmin": final_maxmin,
        "throughputs": all_throughput_histories,
        "jain_histories": all_jain_histories
    }
    
def compare_policies_across_param(
    policy_name,
    policy_train_fn,
    param_name,          # 'pSwap' or 'pGen'
    param_values,        # list of values to sweep
    edges,
    goal_edges,
    p_gen,
    p_swap,
    max_age,
    train_kwargs={},
    validate_kwargs={},
    plot=True,
    num_runs=5,
    num_steps=10000,
    num_simulations=10):
    assert param_name in ['pGen', 'pSwap'], "param_name must be 'pGen' or 'pSwap'"

    all_results = {}

    for param_val in param_values:
        print(f"\n=== Evaluating {policy_name} for {param_name} = {param_val} ===")

        # Choose which param to vary
        curr_p_gen = param_val if param_name == 'pGen' else p_gen
        curr_p_swap = param_val if param_name == 'pSwap' else p_swap

        results = run_policy_experiments(
            train_policy_fn=policy_train_fn,
            policy_name=f"{policy_name} ({param_name}={param_val})",
            edges=edges,
            goal_edges=goal_edges,
            p_gen=curr_p_gen,
            p_swap=curr_p_swap,
            max_age=max_age,
            num_runs=num_runs,
            num_steps=num_steps,
            num_simulations=num_simulations,
            train_kwargs=train_kwargs,
            validate_kwargs=validate_kwargs,
            plot=False  # Plot later all together
        )
        all_results[param_val] = results

    # === Plot Results Across Parameter Sweep ===
    if plot:
        fig, axs = plt.subplots(1, 4, figsize=(28, 5))

        # Plot 1: Jain's Index per run
        for val in param_values:
            jains = all_results[val]['jains']
            axs[0].plot(jains, marker='o', label=f"{param_name}={val}")
        axs[0].set_title(f"Jain's Fairness Across {param_name} Values")
        axs[0].set_xlabel("Run")
        axs[0].set_ylabel("Jain's Index")
        axs[0].legend()
        axs[0].grid(True)

        # Plot 2: Min-Max Fairness per run
        for val in param_values:
            minmax = all_results[val]['minmax']
            axs[1].plot(minmax, marker='s', label=f"{param_name}={val}")
        axs[1].set_title(f"Min-Max Fairness Across {param_name} Values")
        axs[1].set_xlabel("Run")
        axs[1].set_ylabel("Min / Max EDR")
        axs[1].legend()
        axs[1].grid(True)

        # Plot 3: Throughput vs Jain's Index (Pareto)
        for val in param_values:
            jains = all_results[val]['jains']
            throughputs = [
                sum(all_results[val]['edrs'][goal][i] for goal in goal_edges)
                for i in range(num_runs)
            ]
            axs[2].scatter(throughputs, jains, label=f"{param_name}={val}", s=60)
        axs[2].set_title(f"Pareto Curve: Throughput vs Jain's Fairness")
        axs[2].set_xlabel("Total Throughput")
        axs[2].set_ylabel("Jain's Index")
        axs[2].legend()
        axs[2].grid(True)

        # Plot 4: EDR per goal, per value
        for goal in goal_edges:
            for val in param_values:
                edrs = all_results[val]['edrs'][goal]
                axs[3].plot(edrs, marker='o', label=f"Goal {goal}, {param_name}={val}")
        axs[3].set_title("EDR per Goal")
        axs[3].set_xlabel("Run")
        axs[3].set_ylabel("Final EDR")
        axs[3].legend(fontsize=8)
        axs[3].grid(True)

        plt.tight_layout()
        plt.show()

    return all_results

