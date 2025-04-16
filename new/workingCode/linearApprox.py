import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import random
import math

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
        
        
        
###################################
### PLOTTTING
###################################
def simulate_policy(Q_table, edges, goal_edges, p_swap, p_gen, max_age, num_steps, edr_window_size=100, plot=True):
    raw_state = [(edge, -1) for edge in edges]
    goal_success_counts = {goal: 0 for goal in goal_edges}
    recent_goal_history = {goal: [] for goal in goal_edges}

    edr_history = {goal: [] for goal in goal_edges}
    jain_history = []
    min_max_history = []
    max_min_history = []
    throughput_history = []
    aged_out_history = []
    action_ratio_history = []

    current_state = get_augmented_state(raw_state, {g: 0 for g in goal_edges}, goal_order=goal_edges)

    for step in range(num_steps):
        ent_state, _ = current_state
        possible_actions = getPossibleActions(ent_state, goal_edges)
        actions_available = len(possible_actions) > 0 and possible_actions != [([], None)]

        best_action = ([], None)
        best_score = -float("inf")

        features = featurize_state(current_state, goal_edges)

        for action in possible_actions:
            q_val = Q_table.get_q_value(features, action)
            if q_val > best_score:
                best_score = q_val
                best_action = action

        action_taken = best_action != ([], None)
        action_ratio_history.append(1.0 if action_taken else 0.0)

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
        minmax = min_max_fairness(edrs)
        maxmin = min(edrs.values())

        throughput_history.append(throughput)
        jain_history.append(fairness)
        min_max_history.append(minmax)
        max_min_history.append(maxmin)

        current_state = get_augmented_state(current_state[0], edrs, goal_order=goal_edges)



    def rolling_average(data, window):
        return np.convolve(data, np.ones(window) / window, mode='valid')

    smoothed_aged_out = rolling_average(aged_out_history, 100)
    smoothed_action_ratio = rolling_average(action_ratio_history, 100)

    if plot:
        fig, axs = plt.subplots(7, 1, figsize=(12, 28))

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

        axs[5].plot(smoothed_aged_out, color='orange')
        axs[5].set_title("Aged-Out Entanglement Ratio (Smoothed)")
        axs[5].set_xlabel("Timestep")
        axs[5].set_ylabel("Aged Out Ratio")
        axs[5].grid(True)

        axs[6].plot(smoothed_action_ratio, color='teal')
        axs[6].set_title("Action Taken When Available (Smoothed)")
        axs[6].set_xlabel("Timestep")
        axs[6].set_ylabel("Action Decision Ratio")
        axs[6].grid(True)

        plt.tight_layout()
        plt.show()

    return (
        goal_success_counts,
        num_steps,
        edr_history,
        jain_history,
        min_max_history,
        throughput_history,
        max_min_history,
        smoothed_aged_out,
        smoothed_action_ratio
    )
