import random
import matplotlib.pyplot as plt
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


def getEpsilonGreedyAction(state, Q, epsilon, goalEdges):
    possible_actions = getPossibleActions(state, goalEdges)
    if not possible_actions:  # If no actions available
        return ([], None)  # Return empty action tuple
    if random.random() < epsilon:
        return random.choice(possible_actions)  # Fixed: use possible_actions instead of actions
    else:
        # Get the action with highest Q-value
        return max(possible_actions, key=lambda a: Q.get_q_value(state, a))

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
        print(f"Setting Q-value {value} for key: {key}")  # Debug print
        self.q_values[key] = value

def getReward(action, goal_success_counts, total_timesteps, pSwap):
    consumed_edges, goal = action
    if goal is None or len(consumed_edges) == 0:
        return 0
    
    # Calculate instant rate based on path length
    start, end = goal
    num_edges = abs(end - start)
    instant_rate = pSwap ** (num_edges - 1)
    
    # Calculate EDR with small constant to avoid division by zero
    errorTerm = 0.001
    edr = goal_success_counts[goal] / max(1, total_timesteps) + errorTerm
    
    # Return reward based on instant rate and EDR
    if instant_rate > 0 and edr > 0:
        return instant_rate / edr
    return 0

def plot_learning_metrics(states, actions, rewards, Q, edr_history, total_steps):
    # Create figure with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Plot EDR over time
    for goal in edr_history:
        axs[0, 0].plot(range(total_steps), edr_history[goal], label=f'Goal {goal}')
    axs[0, 0].set_title('Entanglement Distribution Rate (EDR)')
    axs[0, 0].set_xlabel('Steps')
    axs[0, 0].set_ylabel('EDR (successes/step)')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # 2. Plot Q-value changes
    q_values = []
    for i in range(len(states)-1):
        if i < len(actions) and actions[i]:
            q_values.append(Q.get_q_value(states[i], actions[i]))
    axs[0, 1].plot(range(len(q_values)), q_values)
    axs[0, 1].set_title('Q-value Changes')
    axs[0, 1].set_xlabel('Steps')
    axs[0, 1].set_ylabel('Q-value')
    axs[0, 1].grid(True)
    
    # 3. Plot reward history
    axs[1, 0].plot(range(len(rewards)), rewards)
    axs[1, 0].set_title('Reward History')
    axs[1, 0].set_xlabel('Steps')
    axs[1, 0].set_ylabel('Reward')
    axs[1, 0].grid(True)
    
    # 4. Plot action selection distribution
    action_counts = {}
    for action_list in actions:
        if action_list:
            action = action_list[0]
            key = str(action)  # Convert to string for counting
            action_counts[key] = action_counts.get(key, 0) + 1
    
    axs[1, 1].bar(action_counts.keys(), action_counts.values())
    axs[1, 1].set_title('Action Selection Distribution')
    axs[1, 1].set_xlabel('Actions')
    axs[1, 1].set_ylabel('Count')
    axs[1, 1].tick_params(axis='x', rotation=45)
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

###### N-Step SARSA ######
# Parameters:
initialEdges = [(0,1),(1,2), (2, 3)]
currentState = [(edge, -1) for edge in initialEdges]
goalEdges = [(0, 2), (0, 3), (1,3)]
totalSteps = 1000000
nLookahead = 3
epsilon = 0.1
maxAge = 3
pSwap = 0.8
pGen = 0.8
gamma = 0.95
alpha = 0.1

def run_n_step_sarsa(initialEdges, goalEdges, totalSteps, nLookahead, epsilon, gamma, alpha, pGen, pSwap, maxAge):
    Q = qTable()
    total_timesteps = 0
    EDRS = {goal: 0 for goal in goalEdges}
    edr_history = {goal: [] for goal in goalEdges}  # Track EDR at each step
    
    # Convert initialEdges to proper state format (edge, age)
    initial_state = [(edge, -1) for edge in initialEdges]  # Start with no entanglements
    states = [initial_state]
    actions = [([], None)]  # Always include do nothing action first, as the intial state is always empty too
    rewards = []
    
    current_state = initial_state
    
    print("\nStarting SARSA run...")
    print("Initial state:", current_state)
    print("Goal edges:", goalEdges)
    
    for t in range(totalSteps):
        print(f"\nStep {t}:")
        
        # Take action A_t, observe R_{t+1} and S_{t+1}
        action = actions[-1]
        # Take the action
        current_state = performAction(action, current_state)
        current_state = ageEntanglements(current_state, maxAge)
        current_state = generateEntanglement(current_state, pGen)
        
        # Observe the reward (and update EDRS)
        reward = getReward(action, EDRS, t+1, pSwap)  # Pass EDRS and current timestep
        rewards.append(reward)
        # Update rate parameters before the action is taken 
        total_timesteps += 1
        _, goal = action
        if goal is not None:
            EDRS[goal] += 1
        
        # Update EDR history
        for g in goalEdges:
            edr_history[g].append(EDRS[g] / (t + 1))
        
        states.append(current_state)
        
        # Select A_{t+1} using epsilon-greedy policy
        action = getEpsilonGreedyAction(current_state, Q, epsilon, goalEdges)
        actions.append(action)
        
        # 1-step SARSA update
        if t > 0:  # We need at least one previous state to update
            # Get current state, action, and reward
            s = states[t-1]
            a = action
            r = rewards[t-1]
            
            # Get next state and action
            s_prime = current_state
            a_prime = action
            
            # Get current Q-value and next Q-value
            current_q = Q.get_q_value(s, a)
            next_q = Q.get_q_value(s_prime, a_prime)
            
            # Apply 1-step SARSA update
            new_q = current_q + alpha * (r + gamma * next_q - current_q)
            Q.set_q_value(s, a, new_q)
            print(f"Q-value update: {current_q} -> {new_q}")
    
    print("\nFinal EDRS:", EDRS)
    print("Total rewards:", sum(rewards))
    print("Number of Q-value updates:", len(Q.q_values))
    
    # Plot learning metrics
    plot_learning_metrics(states, actions, rewards, Q, edr_history, totalSteps)
    
    return current_state, Q, EDRS


run_n_step_sarsa(initialEdges, goalEdges, totalSteps, nLookahead, epsilon, gamma, alpha, pGen, pSwap, maxAge)
    
    
    
    
    
