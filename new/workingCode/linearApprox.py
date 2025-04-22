import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import random
import math
import math

def plot_q_value_convergence(q_value_diffs, window=10):
    # Optional: Smooth the plot with a moving average
    smoothed = np.convolve(q_value_diffs, np.ones(window)/window, mode='valid')

    plt.figure(figsize=(10, 5))
    plt.plot(smoothed, label=f"Q-Value Change (rolling avg, window={window})")
    plt.xlabel("Training Steps")
    plt.ylabel("Average ΔQ per Update")
    plt.title("Convergence of Q-Values During Training")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def compute_reward(
    action,
    goal_success_queues,
    total_timesteps,
    pSwap,
    mode="basic",
    alpha=0.5,
    epsilon=0.001,
    success=True
):
    consumed_edges, goal_edge = action

    if goal_edge is None or not consumed_edges:
        return 0.0

    success_prob = pSwap ** (len(consumed_edges) - 1)
    edr = sum(goal_success_queues[goal_edge]) / max(1, len(goal_success_queues[goal_edge])) + epsilon
    x = success_prob / edr

    if mode == "alpha_fair":
        if alpha == 1.0:
            expected_reward = math.log(1 + x)
        else:
            expected_reward = ((1 + x) ** (1 - alpha)) / (1 - alpha)
        reward = expected_reward if success else 0.0

    elif mode == "partial":
        expected_reward = math.log(1 + x)
        reward = expected_reward if success else 0.5 * expected_reward

    else:  # "basic"
        expected_reward = math.log(1 + x)
        reward = expected_reward if success else 0.0

    goal_success_queues[goal_edge].append(1 if success else 0)
    return reward



def compare_alpha_vs_env_param(
    policy_name,
    policy_train_fn,
    param_name,
    param_values,
    alpha_values,
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
    num_simulations=10
):
    assert param_name in ['pGen', 'pSwap'], "param_name must be 'pGen' or 'pSwap'"
    all_results = {}

    for param_val in param_values:
        print(f"\n=== Running {param_name} = {param_val} ===")
        curr_p_gen = param_val if param_name == 'pGen' else p_gen
        curr_p_swap = param_val if param_name == 'pSwap' else p_swap

        alpha_results = {}

        for alpha in alpha_values:
            print(f"  ↳ Alpha = {alpha}")
            results = run_policy_experiments(
                train_policy_fn=policy_train_fn,
                policy_name=f"{policy_name} ({param_name}={param_val}, α={alpha})",
                edges=edges,
                goal_edges=goal_edges,
                p_gen=curr_p_gen,
                p_swap=curr_p_swap,
                max_age=max_age,
                num_runs=num_runs,
                num_steps=num_steps,
                num_simulations=num_simulations,
                train_kwargs={**train_kwargs, "reward_mode": "alpha_fair", "reward_alpha": alpha},
                validate_kwargs=validate_kwargs,
                plot=False
            )
            alpha_results[alpha] = results

        all_results[param_val] = alpha_results

    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(18, 6))
        color_map = plt.get_cmap("tab10")

        # Plot Jain's fairness vs alpha for each pGen/pSwap
        for i, (param_val, alpha_result) in enumerate(all_results.items()):
            x = []
            jain_y = []
            tp_y = []

            for alpha in alpha_values:
                x.append(alpha)
                jain_y.append(np.mean(alpha_result[alpha]['jains']))
                tp_y.append(np.mean([
                    sum(alpha_result[alpha]['edrs'][goal][run_i] for goal in goal_edges)
                    for run_i in range(num_runs)
                ]))

            axs[0].plot(x, jain_y, marker='o', label=f"{param_name}={param_val}", color=color_map(i))
            axs[1].plot(x, tp_y, marker='o', label=f"{param_name}={param_val}", color=color_map(i))

        axs[0].set_title(f"Jain's Fairness vs Alpha")
        axs[0].set_xlabel("Alpha")
        axs[0].set_ylabel("Jain's Index")
        axs[0].set_ylim(0, 1.05)
        axs[0].legend()
        axs[0].grid(True)

        axs[1].set_title(f"Total Throughput vs Alpha")
        axs[1].set_xlabel("Alpha")
        axs[1].set_ylabel("Total Throughput")
        axs[1].legend()
        axs[1].grid(True)

        summary_title = (
            f"{policy_name}: Alpha vs {param_name}\n"
            f"Fixed Params — pGen={p_gen}, pSwap={p_swap}, maxAge={max_age}, "
            f"num_runs={num_runs}, num_steps={num_steps}, num_sims={num_simulations}"
        )
        plt.suptitle(summary_title, fontsize=18, y=1.02)
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.show()

    return all_results




def compare_policies_across_alpha(
    policy_name,
    policy_train_fn,
    alpha_values,
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
    num_simulations=10
):
    all_results = {}

    for alpha_val in alpha_values:
        print(f"\n=== Evaluating {policy_name} for alpha = {alpha_val} ===")

        results = run_policy_experiments(
            train_policy_fn=policy_train_fn,
            policy_name=f"{policy_name} (alpha={alpha_val})",
            edges=edges,
            goal_edges=goal_edges,
            p_gen=p_gen,
            p_swap=p_swap,
            max_age=max_age,
            num_runs=num_runs,
            num_steps=num_steps,
            num_simulations=num_simulations,
            train_kwargs={**train_kwargs, "reward_mode": "alpha_fair", "reward_alpha": alpha_val},
            validate_kwargs=validate_kwargs,
            plot=False
        )
        all_results[alpha_val] = results

    if plot:
        fig, axs = plt.subplots(2, 3, figsize=(36, 12))
        axs = axs.flatten()
        color_map = plt.get_cmap("tab10")

        # --- Jain's Fairness Plot ---
        means_jain = []
        for i, alpha in enumerate(alpha_values):
            jains = all_results[alpha]['jains']
            axs[0].scatter([alpha] * len(jains), jains, color=color_map(i), alpha=0.6)
            means_jain.append(np.mean(jains))

        axs[0].plot(alpha_values, means_jain, color='black', linestyle='-', linewidth=2, label="Mean Jain's")
        axs[0].set_title("Jain's Fairness vs Alpha")
        axs[0].set_xlabel("Alpha")
        axs[0].set_ylabel("Jain's Index")
        axs[0].set_ylim(0.45, 1.05)
        axs[0].legend()
        axs[0].grid(True)

        # --- Pareto curve: Throughput vs Jain ---
        avg_throughputs = []
        avg_jains = []

        for i, alpha in enumerate(alpha_values):
            jains = all_results[alpha]['jains']
            throughputs = [sum(all_results[alpha]['edrs'][goal][run_i] for goal in goal_edges) for run_i in range(num_runs)]

            axs[1].scatter(throughputs, jains, color=color_map(i), label=f"α={alpha}", s=60, alpha=0.7)

            avg_throughputs.append(np.mean(throughputs))
            avg_jains.append(np.mean(jains))

        axs[1].plot(avg_throughputs, avg_jains, color='black', linestyle='-', linewidth=2, label="Mean Trend")
        axs[1].set_title("Pareto Curve: Throughput vs Jain's Index")
        axs[1].set_xlabel("Total Throughput")
        axs[1].set_ylabel("Jain's Index")
        axs[1].set_xlim(0, 1.05)
        axs[1].set_ylim(4.5, 1.05)
        axs[1].legend()
        axs[1].grid(True)

        # --- Final EDR per Goal ---
        for goal_i, goal in enumerate(goal_edges):
            for i, alpha in enumerate(alpha_values):
                y_vals = all_results[alpha]['edrs'][goal]
                axs[2].scatter([alpha] * len(y_vals), y_vals, label=f"{goal}" if i == 0 else "", alpha=0.6, s=30, color=color_map(goal_i))

            mean_y = [np.mean(all_results[alpha]['edrs'][goal]) for alpha in alpha_values]
            axs[2].plot(alpha_values, mean_y, linestyle='--', linewidth=2, color=color_map(goal_i))

        axs[2].set_title("Final EDR per Goal")
        axs[2].set_xlabel("Alpha")
        axs[2].set_ylabel("EDR")
        axs[2].set_ylim(0, 1)
        axs[2].legend(fontsize=9)
        axs[2].grid(True)

        # --- Expected Delivery Time per Goal ---
        for goal_i, goal in enumerate(goal_edges):
            for i, alpha in enumerate(alpha_values):
                y_vals = all_results[alpha]['edts'][goal]
                axs[3].scatter([alpha] * len(y_vals), y_vals, label=f"{goal}" if i == 0 else "", alpha=0.6, s=30, color=color_map(goal_i))

            mean_y = [np.mean(all_results[alpha]['edts'][goal]) for alpha in alpha_values]
            axs[3].plot(alpha_values, mean_y, linestyle='--', linewidth=2, color=color_map(goal_i))

        axs[3].set_title("Expected Delivery Time per Goal")
        axs[3].set_xlabel("Alpha")
        axs[3].set_ylabel("EDT (1 / EDR)")
        axs[3].set_ylim(0, 100)
        axs[3].legend(fontsize=9)
        axs[3].grid(True)

        # --- Aged-Out & Action Ratio ---
        for i, alpha in enumerate(alpha_values):
            all_aged = all_results[alpha]['aged_out_histories']
            all_actions = all_results[alpha]['action_ratio_histories']

            aged_vals = [np.mean(sim[-1000:]) for run in all_aged for sim in run]
            action_vals = [np.mean(sim[-1000:]) for run in all_actions for sim in run]

            axs[4].scatter([alpha] * len(aged_vals), aged_vals, color='orange', alpha=0.6, label="Aged-Out" if i == 0 else "")
            axs[4].scatter([alpha] * len(action_vals), action_vals, color='teal', alpha=0.6, label="Action Ratio" if i == 0 else "")

        axs[4].set_title("Aged-Out & Action Decision Ratio")
        axs[4].set_xlabel("Alpha")
        axs[4].set_ylabel("Ratio")
        axs[4].set_ylim(0, 1)
        axs[4].legend()
        axs[4].grid(True)

        axs[5].axis('off')

        summary_title = (
            f"{policy_name}: Varying Alpha Fairness\n"
            f"Fixed Params — pGen={p_gen}, pSwap={p_swap}, maxAge={max_age}, "
            f"num_runs={num_runs}, num_steps={num_steps}, num_sims={num_simulations}\n"
            f"Initial Edges={edges}, Goal Edges={goal_edges}"
        )

        plt.suptitle(summary_title, fontsize=20, y=1.05)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.show()

    return all_results


def compute_reward(
    action,
    goal_success_queues,
    total_timesteps,
    pSwap,
    mode="basic",             # "basic", "alpha_fair", or "partial"
    alpha=0.5,
    epsilon=0.001,
    success=True
):
    consumed_edges, goal_edge = action

    if goal_edge is None or not consumed_edges:
        return 0.0

    success_prob = pSwap ** (len(consumed_edges) - 1)
    edr = sum(goal_success_queues[goal_edge]) / max(1, len(goal_success_queues[goal_edge])) + epsilon

    # Relative performance ratio
    x = success_prob / edr

    # --- Choose Reward Calculation ---
    if mode == "alpha_fair":
        if alpha == 1.0:
            expected_reward = math.log(1 + x)
        else:
            expected_reward = ((1 + x) ** (1 - alpha)) / (1 - alpha)

        reward = expected_reward if success else 0.0

    elif mode == "partial":
        expected_reward = math.log(1 + x)
        if success:
            reward = expected_reward
        else:
            reward = 0.5 * expected_reward  # partial reward for attempt

    else:  # "basic"
        expected_reward = math.log(1 + x)
        reward = expected_reward if success else 0.0

    # Update queue with result
    goal_success_queues[goal_edge].append(1 if success else 0)

    return reward


def getRewardAlphaFair(action, goal_success_queues, total_timesteps, pSwap, alpha=0.5, epsilon=0.001, success=True):
    reward = 0.0
    consumed_edges, goal_edge = action

    if goal_edge is None or not consumed_edges:
        return 0.0

    success_prob = pSwap ** (len(consumed_edges) - 1)
    edr = sum(goal_success_queues[goal_edge]) / max(1, len(goal_success_queues[goal_edge])) + epsilon

    x = success_prob / edr  # relative performance for alpha fairness

    if alpha == 1.0:
        expected_reward = math.log(1 + x)  # add 1 to keep reward >= 0
    else:
        expected_reward = ((1 + x) ** (1 - alpha)) / (1 - alpha)

    if success:
        reward += expected_reward
        goal_success_queues[goal_edge].append(1)
    else:
        goal_success_queues[goal_edge].append(0)

    return reward

def getRewardAttemptConsidered(action, goal_success_queues, total_timesteps, pSwap, epsilon=0.001, success=True):
    reward = 0.0
    consumed_edges, goal_edge = action
    if goal_edge is None or not consumed_edges:
        return 0

    success_prob = pSwap ** (len(consumed_edges) - 1)
    edr = sum(goal_success_queues[goal_edge]) / max(1, len(goal_success_queues[goal_edge])) + epsilon
    expected_reward = math.log(1 + (success_prob / edr))

    if success:
        reward += expected_reward
        goal_success_queues[goal_edge].append(1)
    else:
        # Reward attempt even if failed
        attempt_bonus = 0.5 * expected_reward  # or some fixed value like 0.2
        reward += attempt_bonus
        goal_success_queues[goal_edge].append(0)

    return reward

def getReward(action, goal_success_queues, total_timesteps, pSwap, epsilon=0.001, success=True):
    reward = 0.0
    consumed_edges, goal_edge = action
    if goal_edge is None or not consumed_edges:
        return 0
    success_prob = pSwap ** (len(consumed_edges) - 1)
    edr = sum(goal_success_queues[goal_edge]) / max(1, len(goal_success_queues[goal_edge])) + epsilon
    expected_reward = math.log(1 + (success_prob / edr))
    if success:
        reward += expected_reward
        goal_success_queues[goal_edge].append(1)
    else:
        goal_success_queues[goal_edge].append(0)
    return reward

def softmax_probs(q_values, temperature=1.0):
    q_values = np.array(q_values, dtype=np.float64)

    if not np.all(np.isfinite(q_values)):
        # If any value is nan or inf, fallback to uniform
        return np.ones_like(q_values) / len(q_values)

    # Prevent division by zero or near-zero temperature
    safe_temp = max(temperature, 1e-8)
    scaled_qs = q_values / safe_temp

    # Numerical stability trick
    scaled_qs -= np.max(scaled_qs)

    exps = np.exp(scaled_qs)
    sum_exps = np.sum(exps)

    if not np.isfinite(sum_exps) or sum_exps == 0:
        # Fallback to uniform if exp results break
        return np.ones_like(q_values) / len(q_values)

    return exps / sum_exps



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

def rolling_average(data, window):
    return np.convolve(data, np.ones(window)/window, mode='valid')



def ageEntanglements(augmented_state, maxAge):
    ent_state, edr_bins = augmented_state
    new_state = []
    for edge, age in ent_state:
        if age >= 0:
            new_age = age + 1
            if new_age > maxAge:
                new_state.append((edge, -1))
            else:
                new_state.append((edge, new_age))
        else:
            new_state.append((edge, age))
    return (tuple(new_state), edr_bins)

def generateEntanglement(augmented_state, pGen):
    ent_state, edr_bins = augmented_state
    new_state = []
    for edge, age in ent_state:
        if age < 0:
            if random.random() < pGen:
                new_state.append((edge, 1))
            else:
                new_state.append((edge, age))
        else:
            new_state.append((edge, age))
    return (tuple(new_state), edr_bins)

def performAction(action, augmented_state):
    consumed_edges, goal_edge = action
    ent_state, edr_bins = augmented_state
    new_state = list(ent_state)
    for edge_to_consume in consumed_edges:
        for i, (edge, age) in enumerate(new_state):
            if edge == edge_to_consume:
                new_state[i] = (edge, -1)
                break
    return (tuple(new_state), edr_bins)

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
            if len(path) > 0:
                actions.append((path, goal))

    return actions if actions else [([], None)]

def get_augmented_state(state, edrs, goal_order=None):
    if goal_order is None:
        goal_order = sorted(edrs.keys())
    edr_vector = tuple(edrs[goal] for goal in goal_order)
    sorted_state = sorted(state, key=lambda x: (x[0][0], x[0][1]))
    return (tuple(sorted_state), edr_vector)

def featurize_state(state, goal_order):
    ent_state, edrs = state  # edrs is a tuple now
    edge_features = [age / 10.0 if age >= 0 else -1.0 for _, age in ent_state]
    edr_features = list(edrs)  # just unpack the tuple
    return np.array(edge_features + edr_features, dtype=np.float32)
class LinearQApproximator:
    def __init__(self, feature_size):
        self.weights = {}  # Dict[action_key] = weight_vector
        self.feature_size = feature_size

    def _action_key(self, action):
        consumed_edges, goal_edge = action
        return (tuple(sorted(consumed_edges)), goal_edge)

    def _init_weights(self, action_key):
        if action_key not in self.weights:
            self.weights[action_key] = np.zeros(self.feature_size)

    def get_q_value(self, features, action):
        key = self._action_key(action)
        self._init_weights(key)
        return float(np.dot(self.weights[key], features))

    def update(self, features, action, target, alpha):
        key = self._action_key(action)
        self._init_weights(key)
        prediction = np.dot(self.weights[key], features)
        error = target - prediction
        self.weights[key] += alpha * error * features
        
        
        
#########################################################################################################
### PLOTTTING #########################################################################################################
#########################################################################################################
def simulate_policy(Q_table, edges, goal_edges, p_swap, p_gen, max_age, num_steps, edr_window_size=100, plot=True):
    raw_state = [(edge, -1) for edge in edges]
    goal_success_counts = {goal: 0 for goal in goal_edges}
    recent_goal_history = {goal: [] for goal in goal_edges}

    edr_history = {goal: [] for goal in goal_edges}
    jain_history = []
    throughput_history = []
    aged_out_history = []
    action_ratio_history = []
    valid_decision_points = []
    q_values_history = []

    current_state = get_augmented_state(raw_state, {g: 0 for g in goal_edges}, goal_order=goal_edges)

    for step in range(num_steps):
        ent_state, _ = current_state
        possible_actions = getPossibleActions(ent_state, goal_edges)
        if ([], None) not in possible_actions:
            possible_actions.append(([], None))

        # --- FIXED ACTION RATIO LOGIC ---
        real_actions = [a for a in possible_actions if a != ([], None)]
        actions_available = len(real_actions) > 0
        valid_decision_points.append(1 if actions_available else 0)

        best_action = ([], None)
        best_score = -float("inf")

        features = featurize_state(current_state, goal_edges)

        for action in possible_actions:
            q_val = Q_table.get_q_value(features, action)
            if q_val > best_score:
                best_score = q_val
                best_action = action

        q_values_history.append(best_score)

        # --- FIX: Determine if action taken is one of the real actions ---
        if actions_available:
            action_taken = best_action in real_actions
            action_ratio_history.append(1.0 if action_taken else 0.0)
        else:
            action_ratio_history.append(np.nan)


        num_existing_before = sum(1 for _, age in ent_state if age >= 0)
        current_state = performAction(best_action, current_state)
        current_state = ageEntanglements(current_state, max_age)
        num_existing_after = sum(1 for _, age in current_state[0] if age >= 0)

        current_state = generateEntanglement(current_state, p_gen)
        num_generated_after = sum(1 for _, age in current_state[0] if age == 1)

        num_aged_out = num_existing_before - num_existing_after
        aged_out_ratio = num_aged_out / (num_aged_out + num_generated_after) if (num_aged_out + num_generated_after) > 0 else 0.0
        aged_out_history.append(aged_out_ratio)

        consumed_edges, goal = best_action
        for g in goal_edges:
            if g == goal and consumed_edges:
                success = random.random() < (p_swap ** (len(consumed_edges) - 1))
                if success:
                    goal_success_counts[g] += 1
                    recent_goal_history[g].append(1)
                else:
                    recent_goal_history[g].append(0)
            else:
                recent_goal_history[g].append(0)

        for g in goal_edges:
            if len(recent_goal_history[g]) > edr_window_size:
                recent_goal_history[g].pop(0)

        edrs = {g: sum(recent_goal_history[g]) / len(recent_goal_history[g]) for g in goal_edges}
        for g in goal_edges:
            edr_history[g].append(edrs[g])

        throughput = sum(edrs.values())
        fairness = jains_index(edrs)

        throughput_history.append(throughput)
        jain_history.append(fairness)

        current_state = get_augmented_state(current_state[0], edrs, goal_order=goal_edges)

    def rolling_average(data, window):
        return np.convolve(data, np.ones(window) / window, mode='valid')

    smoothed_aged_out = rolling_average(aged_out_history, 100)
    smoothed_action_ratio = rolling_average([x for x in action_ratio_history if not np.isnan(x)], 100)

    corrected_ratios = []
    window = 100
    for t in range(window, len(action_ratio_history)):
        valid_window = valid_decision_points[t - window:t]
        acted_window = action_ratio_history[t - window:t]
        if sum(valid_window) > 0:
            corrected = np.mean([a for a, v in zip(acted_window, valid_window) if v == 1 and not np.isnan(a)])
        else:
            corrected = np.nan
        corrected_ratios.append(corrected)

    if plot:
        fig, axs = plt.subplots(2, 2, figsize=(16, 10))

        for goal in goal_edges:
            axs[0, 0].plot(edr_history[goal], label=f'Goal {goal}')
        axs[0, 0].set_title('EDR Evolution Over Time')
        axs[0, 0].set_xlabel('Timestep')
        axs[0, 0].set_ylabel('EDR')
        axs[0, 0].legend()
        axs[0, 0].grid(True)
        axs[0, 0].set_ylim(0, 1)

        axs[0, 1].plot(jain_history, color='purple')
        axs[0, 1].set_title("Jain's Fairness Index Over Time")
        axs[0, 1].set_xlabel('Timestep')
        axs[0, 1].set_ylabel("Jain's Index")
        axs[0, 1].grid(True)
        axs[0, 1].set_ylim(0, 1.05)

        axs[1, 0].scatter(throughput_history, jain_history, color='darkred', alpha=0.7, s=10)
        axs[1, 0].set_title("Pareto Curve: Throughput vs Jain's Fairness")
        axs[1, 0].set_xlabel("Total Throughput")
        axs[1, 0].set_ylabel("Jain's Index")
        axs[1, 0].grid(True)
        axs[1, 0].set_xlim(0, 1)
        axs[1, 0].set_ylim(0, 1.05)

        axs[1, 1].plot(smoothed_aged_out, color='orange')
        axs[1, 1].set_title("Aged-Out Entanglement Ratio (Smoothed)")
        axs[1, 1].set_xlabel("Timestep")
        axs[1, 1].set_ylabel("Aged Out Ratio")
        axs[1, 1].grid(True)

        plt.tight_layout()

        plt.figure(figsize=(8, 4))
        plt.plot(corrected_ratios, label="Corrected Action Ratio", color='teal')
        plt.title("Corrected Action Decision Ratio (when actions possible)")
        plt.xlabel("Timestep")
        plt.ylabel("Ratio")
        plt.ylim(0, 1.05)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 4))
        plt.plot(q_values_history, label="Best Q-Value", color='slateblue')
        plt.title("Q-Value of Selected Actions Over Time")
        plt.xlabel("Timestep")
        plt.ylabel("Q-Value")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        summary_title = (
            f"Simulate Policy: Static Run\n"
            f"Fixed Params — pGen={p_gen}, pSwap={p_swap}, maxAge={max_age}, "
            f"num_steps={num_steps}\n"
            f"Initial Edges={edges}, Goal Edges={goal_edges}"
        )
        plt.suptitle(summary_title, fontsize=18, y=1.02)
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.show()

    burn_in = 2000
    trimmed_edrs = {
        goal: edr_history[goal][burn_in:] for goal in goal_edges
    }
    trimmed_jains = jain_history[burn_in:]

    mean_edrs = {goal: np.mean(trimmed_edrs[goal]) for goal in goal_edges}
    mean_jains = np.mean(trimmed_jains)

    print("\n=== Mean EDRs After Burn-In ===")
    for goal, edr_val in mean_edrs.items():
        print(f"Goal {goal}: {edr_val:.4f}")
    print(f"Mean Jain's Index (after burn-in): {mean_jains:.4f}")

    return (
        goal_success_counts,
        num_steps,
        edr_history,
        jain_history,
        throughput_history,
        smoothed_aged_out,
        smoothed_action_ratio
    )



def validate_policy_simulation(Q_table, edges, goal_edges, p_swap, p_gen, max_age, num_steps, num_simulations, seed=10, plot=True, window=1000):
    all_edr_histories = []
    all_jain_histories = []
    all_throughput_histories = []
    all_aged_out_histories = []
    all_action_ratio_histories = []

    final_edr_means = []
    final_jains = []

    for sim in range(num_simulations):
        random.seed(seed + sim)
        _, _, edr_history, jain_history, throughput_history, aged_out_history, action_ratio_history = simulate_policy(
            Q_table, edges, goal_edges, p_swap, p_gen, max_age, num_steps, plot=False
        )

        all_edr_histories.append(edr_history)
        all_jain_histories.append(jain_history)
        all_throughput_histories.append(throughput_history)
        all_aged_out_histories.append(aged_out_history)
        all_action_ratio_histories.append(action_ratio_history)

        final_edr_per_goal = {
            goal: np.mean(edr_history[goal][-window:]) for goal in goal_edges
        }
        total_throughput = sum(final_edr_per_goal.values())
        final_jain = jains_index(final_edr_per_goal)

        final_edr_means.append(total_throughput)
        final_jains.append(final_jain)

    mean_final_edrs_by_goal = {
        goal: np.mean([np.mean(edr_history[goal][-window:]) for edr_history in all_edr_histories])
        for goal in goal_edges
    }

    mean_final_edts_by_goal = {
        goal: 1.0 / (mean_final_edrs_by_goal[goal] + 1e-8)
        for goal in goal_edges
    }

    if plot:
        fig, axs = plt.subplots(2, 3, figsize=(22, 10))
        axs = axs.flatten()

        # --- Final EDR per goal across simulations ---
        for goal in goal_edges:
            edr_vals = [np.mean(edr_history[goal][-window:]) for edr_history in all_edr_histories]
            axs[0].plot(range(num_simulations), edr_vals, marker='o', label=f"Goal {goal}")

        axs[0].set_title("Final EDRs per Simulation")
        axs[0].set_xlabel("Simulation")
        axs[0].set_ylabel("Mean Final EDR (last N steps)")
        axs[0].set_ylim(0, 1)
        axs[0].grid(True)
        axs[0].legend()

        # --- Jain's Index ---
        axs[1].plot(final_jains, marker='o', color='purple')
        axs[1].set_title("Jain's Fairness Index")
        axs[1].set_ylabel("Jain's Index")
        axs[1].set_xlabel("Simulation")
        axs[1].grid(True)
        axs[1].set_ylim(0.45, 1.05)

        # --- Pareto: Throughput vs Fairness ---
        axs[2].scatter(final_edr_means, final_jains, c='darkred')
        axs[2].set_title("Pareto Curve")
        axs[2].set_xlabel("Throughput (Sum of EDRs)")
        axs[2].set_ylabel("Jain's Index")
        axs[2].grid(True)
        axs[2].set_xlim(0, max(final_edr_means) * 1.1)
        axs[2].set_ylim(0.45, 1.05)

        # --- Aged-Out Ratio ---
        avg_aged = [np.mean(sim[-window:]) for sim in all_aged_out_histories]
        axs[3].plot(avg_aged, color='orange')
        axs[3].set_title("Aged-Out Entanglement Ratio")
        axs[3].set_xlabel("Simulation")
        axs[3].set_ylabel("Ratio")
        axs[3].set_ylim(0, 1.05)
        axs[3].grid(True)

        # --- Action Decision Ratio ---
        avg_actions = [np.mean(sim[-window:]) for sim in all_action_ratio_histories]
        axs[4].plot(avg_actions, color='teal')
        axs[4].set_title("Action Decision Ratio")
        axs[4].set_xlabel("Simulation")
        axs[4].set_ylabel("Ratio")
        axs[4].set_ylim(0, 1.05)
        axs[4].grid(True)

        axs[5].axis('off')

        summary_title = (
            f"Simulate Policy: Validation Across Simulations\n"
            f"pGen={p_gen}, pSwap={p_swap}, maxAge={max_age}, num_steps={num_steps}, "
            f"num_simulations={num_simulations}\n"
            f"Initial Edges={edges}, Goal Edges={goal_edges}"
        )
        plt.suptitle(summary_title, fontsize=20, y=1.02)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.show()

    return (
        mean_final_edrs_by_goal,
        mean_final_edts_by_goal,
        np.mean(final_jains),
        all_throughput_histories,
        all_jain_histories,
        all_aged_out_histories,
        all_action_ratio_histories
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
    final_edts_by_goal = {goal: [] for goal in goal_edges}
    final_jains = []
    all_throughput_histories = []
    all_jain_histories = []
    all_aged_out_histories = []
    all_action_ratio_histories = []

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

        (
            mean_final_edrs_by_goal,
            mean_final_edts_by_goal,
            mean_final_jain,
            throughput_histories,
            jain_histories,
            aged_out_histories,
            action_ratio_histories
        ) = results

        for goal in goal_edges:
            final_edrs_by_goal[goal].append(mean_final_edrs_by_goal[goal])
            final_edts_by_goal[goal].append(mean_final_edts_by_goal[goal])

        final_jains.append(mean_final_jain)
        all_throughput_histories.append(throughput_histories)
        all_jain_histories.append(jain_histories)
        all_aged_out_histories.append(aged_out_histories)
        all_action_ratio_histories.append(action_ratio_histories)

    if plot:
        print("PLOTTING...")
        num_policies = num_runs
        throughputs = [
            sum(final_edrs_by_goal[goal][i] for goal in goal_edges)
            for i in range(num_policies)
        ]

        fig, axs = plt.subplots(2, 3, figsize=(34, 12))
        axs = axs.flatten()

        for goal in goal_edges:
            axs[0].plot(final_edrs_by_goal[goal], marker='o', label=f"Goal {goal}")
        axs[0].set_title("Mean Final EDR per Goal")
        axs[0].set_ylim(0, 1)
        axs[0].set_xlabel("Policy Run")
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(final_jains, label="Jain's Index", color='purple')
        axs[1].set_title("Fairness Metric")
        axs[1].set_ylim(0.45, 1)
        axs[1].set_xlabel("Policy Run")
        axs[1].legend()
        axs[1].grid(True)

        axs[2].scatter(throughputs, final_jains, color='darkred', s=60)
        axs[2].set_title("Pareto Curve: Throughput vs Jain's Index")
        axs[2].set_xlabel("Total Throughput")
        axs[2].set_ylabel("Jain's Index")
        axs[2].set_ylim(0.45, 1)
        axs[2].grid(True)

        for goal in goal_edges:
            axs[3].plot(final_edts_by_goal[goal], label=f"Goal {goal}", linestyle='--')
        axs[3].set_title("Expected Delivery Time per Goal")
        axs[3].set_xlabel("Policy Run")
        axs[3].legend()
        axs[3].grid(True)

        avg_aged_out = []
        for t in range(num_steps):
            vals = [[sim[t] for sim in hist if t < len(sim)] for hist in all_aged_out_histories]
            flat = [v for sublist in vals for v in sublist]
            avg_aged_out.append(np.mean(flat) if flat else 0)

        axs[4].plot(avg_aged_out, color='orange')
        axs[4].set_title("Avg. Aged-Out Entanglement Ratio Over Time")
        axs[4].set_xlabel("Timestep")
        axs[4].set_ylim(0, 1)
        axs[4].set_ylabel("Ratio")
        axs[4].grid(True)

        avg_action_ratio = []
        for t in range(num_steps):
            vals = [[sim[t] for sim in hist if t < len(sim)] for hist in all_action_ratio_histories]
            flat = [v for sublist in vals for v in sublist]
            avg_action_ratio.append(np.mean(flat) if flat else 0)

        axs[5].plot(avg_action_ratio, color='teal')
        axs[5].set_title("Avg. Action Decision Ratio Over Time")
        axs[5].set_xlabel("Timestep")
        axs[5].set_ylabel("Ratio")
        axs[5].grid(True)

        plt.suptitle(
            f"Simulate Policy: {policy_name} (Avg. over {num_runs} runs)\n"
            f"Fixed Params — pGen={p_gen}, pSwap={p_swap}, maxAge={max_age}, "
            f"num_steps={num_steps}, num_sims={num_simulations}\n"
            f"Edges={edges}, Goals={goal_edges}",
            fontsize=20, y=1.02
        )
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.show()

    return {
        "edrs": final_edrs_by_goal,
        "edts": final_edts_by_goal,
        "jains": final_jains,
        "throughputs": all_throughput_histories,
        "jain_histories": all_jain_histories,
        "aged_out_histories": all_aged_out_histories,
        "action_ratio_histories": all_action_ratio_histories
    }


def compare_policies_across_param(
    policy_name,
    policy_train_fn,
    param_name,
    param_values,
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
    num_simulations=10
):
    assert param_name in ['pGen', 'pSwap'], "param_name must be 'pGen' or 'pSwap'"
    all_results = {}

    for param_val in param_values:
        print(f"\n=== Evaluating {policy_name} for {param_name} = {param_val} ===")
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
            plot=False
        )
        all_results[param_val] = results

    if plot:
        fig, axs = plt.subplots(2, 3, figsize=(36, 12))
        axs = axs.flatten()
        color_map = plt.get_cmap("tab10")

        # --- Jain's Fairness Plot ---
        means_jain = []
        for i, val in enumerate(param_values):
            jains = all_results[val]['jains']
            axs[0].scatter([val] * len(jains), jains, color=color_map(i), alpha=0.6)
            means_jain.append(np.mean(jains))

        axs[0].plot(param_values, means_jain, color='black', linestyle='-', linewidth=2, label="Mean Jain's")
        axs[0].set_title("Jain's Fairness")
        axs[0].set_xlabel(param_name)
        axs[0].set_ylabel("Jain's Index")
        axs[0].set_ylim(0.5, 1.05) # Minimum of Jain's metric is 0.5... 1/n 
        axs[0].legend()
        axs[0].grid(True)

        # --- Pareto curve: Throughput vs Jain ---
        avg_throughputs = []
        avg_jains = []

        for i, val in enumerate(param_values):
            jains = all_results[val]['jains']
            throughputs = [sum(all_results[val]['edrs'][goal][run_i] for goal in goal_edges) for run_i in range(num_runs)]

            axs[1].scatter(throughputs, jains, color=color_map(i), label=f"{param_name}={val}", s=60, alpha=0.7)

            avg_throughputs.append(np.mean(throughputs))
            avg_jains.append(np.mean(jains))

        axs[1].plot(avg_throughputs, avg_jains, color='black', linestyle='-', linewidth=2, label="Mean Trend")
        axs[1].set_title("Pareto Curve: Throughput vs Jain's Index")
        axs[1].set_xlabel("Total Throughput")
        axs[1].set_ylabel("Jain's Index")
        axs[1].set_xlim(0, 1.05)
        axs[1].set_ylim(0.45, 1.05)
        axs[1].legend()
        axs[1].grid(True)

        # --- Final EDR per Goal ---
        for goal_i, goal in enumerate(goal_edges):
            for i, val in enumerate(param_values):
                y_vals = all_results[val]['edrs'][goal]
                axs[2].scatter([val] * len(y_vals), y_vals, label=f"{goal}" if i == 0 else "", alpha=0.6, s=30, color=color_map(goal_i))

            mean_y = [np.mean(all_results[val]['edrs'][goal]) for val in param_values]
            axs[2].plot(param_values, mean_y, linestyle='--', linewidth=2, color=color_map(goal_i))

        axs[2].set_title("Final EDR per Goal")
        axs[2].set_xlabel(param_name)
        axs[2].set_ylabel("EDR")
        axs[2].set_ylim(0, 1)
        axs[2].legend(fontsize=9)
        axs[2].grid(True)

        # --- Expected Delivery Time per Goal ---
        for goal_i, goal in enumerate(goal_edges):
            for i, val in enumerate(param_values):
                y_vals = all_results[val]['edts'][goal]
                axs[3].scatter([val] * len(y_vals), y_vals, label=f"{goal}" if i == 0 else "", alpha=0.6, s=30, color=color_map(goal_i))

            mean_y = [np.mean(all_results[val]['edts'][goal]) for val in param_values]
            axs[3].plot(param_values, mean_y, linestyle='--', linewidth=2, color=color_map(goal_i))

        axs[3].set_title("Expected Delivery Time per Goal")
        axs[3].set_xlabel(param_name)
        axs[3].set_ylabel("EDT (1 / EDR)")
        axs[3].set_ylim(0, 100)
        axs[3].legend(fontsize=9)
        axs[3].grid(True)

        # --- Aged-Out & Action Ratio ---
        for i, val in enumerate(param_values):
            all_aged = all_results[val]['aged_out_histories']
            all_actions = all_results[val]['action_ratio_histories']

            aged_vals = [np.mean(sim[-1000:]) for run in all_aged for sim in run]
            action_vals = [np.mean(sim[-1000:]) for run in all_actions for sim in run]

            axs[4].scatter([val] * len(aged_vals), aged_vals, color='orange', alpha=0.6, label="Aged-Out" if i == 0 else "")
            axs[4].scatter([val] * len(action_vals), action_vals, color='teal', alpha=0.6, label="Action Ratio" if i == 0 else "")

        axs[4].set_title("Aged-Out & Action Decision Ratio")
        axs[4].set_xlabel(param_name)
        axs[4].set_ylabel("Ratio")
        axs[4].set_ylim(0, 1)
        axs[4].legend()
        axs[4].grid(True)

        axs[5].axis('off')

        summary_title = (
            f"{policy_name}: Varying {param_name}\n"
            f"Fixed Params — pGen={p_gen}, pSwap={p_swap}, maxAge={max_age}, "
            f"num_runs={num_runs}, num_steps={num_steps}, num_sims={num_simulations}\n"
            f"Initial Edges={edges}, Goal Edges={goal_edges}"
        )

        plt.suptitle(summary_title, fontsize=20, y=1.05)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.show()

    return all_results
