import random
import matplotlib.pyplot as plt
import numpy as np
from utility import qTable, getPossibleActions, performAction, ageEntanglements, generateEntanglement

def getEpsilonGreedyAction(state, Q, epsilon, goalEdges):
    possible_actions = getPossibleActions(state, goalEdges)
    if not possible_actions:  # If no actions available
        return ([], None)  # Return empty action tuple
    if random.random() < epsilon:
        return random.choice(possible_actions)  # Fixed: use possible_actions instead of actions
    else:
        # Get the action with highest Q-value
        return max(possible_actions, key=lambda a: Q.get_q_value(state, a))


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


###### N-Step SARSA ######
# Parameters:
# === CONFIGURATION ===

edges = [(0, 1), (1,2), (2, 3), (3, 4)]
goalEdges = [(0, 4), (2, 4)]
pSwap = 0.8
pGen = 0.8
maxAge = 2


totalSteps = 300000
nLookahead = 5
epsilon = 0.1
gamma = 0.95
alpha = 0.1
currentState, Q, EDRS = run_n_step_sarsa(edges, goalEdges, totalSteps, nLookahead, epsilon, gamma, alpha, pGen, pSwap, maxAge)

from utility import simulate_policy
goal_success_counts, total_timesteps, edr_history = simulate_policy(
    Q_table=Q,  # Your trained qTable object
    edges=edges,
    goal_edges=goalEdges,
    p_swap=pSwap,
    p_gen=pGen,
    max_age=maxAge,
    num_steps=10000
)