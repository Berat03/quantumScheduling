from pre_computation import getPossibleStates, getTransitionProbabilities


gamma = 0.9
epsilon = 0.001

V = {} # {state: value} # state = {(1,2): 1, (2,3): 2} ... hashable_state = tuple(sorted(state.items()))
allStates = [] # [state] precomputed
allActions = {} # {state: [actions]} precomputed
allTransitions = {} # {(state, action): [(next_state, prob), ...]}


def getActionValue(state, action, V):
    # Immediate reward for this state-action pair
    immediateReward = get_reward(action)
    
    # Expected future value
    futureValue = 0
    for nextState, transitionProbability in transitions[(state, action)]:
        futureValue += transitionProbability * V[nextState]
        
    # Bellman equation
    return immediateReward + gamma * futureValue


while True: 
    prev_V = V.copy()
    delta = 0
    
    for state in allPossibleStates:
        bestActionValue = float('-inf')
        
        for action in allActions[state]:
            
            value = getActionValue(state, action, V) # Uses V(S`) and transition probabilities
            bestActionValue = max(bestActionValue, value)
            
            
        V[state] = bestActionValue # Update the value of the state
        
        delta = max(delta, abs(bestActionValue - prev_V[state]))
        
    if delta < epsilon: # Convergence
        break
    