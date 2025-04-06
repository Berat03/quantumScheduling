# TESTING 

def test_getPossibleActions():
    # Test 1: Basic case with all edges existing
    state1 = [((0,1), 1), ((1,2), 1), ((2,3), 1)]
    goals1 = [(1,3), (0,2), (0,3)]
    result1 = getPossibleActions(state1, goals1)
    print("Test 1 - All edges exist:")
    print(f"State: {state1}")
    print(f"Goals: {goals1}")
    print(f"Result: {result1}")
    print("Expected: Should find paths for all goals")
    print("---")

    # Test 2: Some edges don't exist (age < 0)
    state2 = [((0,1), -1), ((1,2), 1), ((2,3), 1)]
    goals2 = [(1,3), (0,2), (0,3)]
    result2 = getPossibleActions(state2, goals2)
    print("Test 2 - Some edges don't exist:")
    print(f"State: {state2}")
    print(f"Goals: {goals2}")
    print(f"Result: {result2}")
    print("Expected: Should only find path for (1,3)")
    print("---")

    # Test 3: Complex graph with multiple paths
    state3 = [((0,1), 1), ((1,2), 1), ((2,3), 1), ((0,2), 1)]
    goals3 = [(0,3)]
    result3 = getPossibleActions(state3, goals3)
    print("Test 3 - Multiple paths possible:")
    print(f"State: {state3}")
    print(f"Goals: {goals3}")
    print(f"Result: {result3}")
    print("Expected: Should find two paths to (0,3)")
    print("---")

    # Test 4: No possible paths
    state4 = [((0,1), -1), ((1,2), -1), ((2,3), -1)]
    goals4 = [(0,3)]
    result4 = getPossibleActions(state4, goals4)
    print("Test 4 - No possible paths:")
    print(f"State: {state4}")
    print(f"Goals: {goals4}")
    print(f"Result: {result4}")
    print("Expected: Should return empty list")
    print("---")

# Run the tests
test_getPossibleActions()


def test_q_learning():
    pGen = 1
    pSwap = 0.8
    maxAge = 2
    n_steps = 3
    
    # Initialize state and Q-table
    state = [(edge, 1) for edge in initialEdges]  # Start with all edges entangled
    Q = qTable()
    
    print("Starting Q-learning test...")
    print("Initial state:", state)
    
    for step in range(n_steps):
        print(f"\nStep {step + 1}:")
        
        # Get possible actions
        actions = getPossibleActions(state, goalEdges)
        print("Possible actions:", actions)
        
        if not actions:
            print("No actions possible, generating new entanglements...")
            state = generateEntanglement(state, pGen)
            print("New state:", state)
            continue
        
        # Choose first action for demonstration
        action = actions[0]
        
        # Get current Q-value
        print(f"Current state: {state}")
        print("Current action:", action)
        current_q = Q.get_q_value(state, action)
        print("Current Q-value:", current_q)
        
        # Perform action and get next state
        next_state = performAction(action, state)
        print("State after action:", next_state)
        
        # Calculate reward (simplified for testing)
        reward = len(action[0])  # Reward based on number of edges consumed
        print("Reward:", reward)
        
        # Update Q-value by simply adding the reward
        new_q = current_q + reward
        Q.set_q_value(state, action, new_q)
        print("Updated Q-value:", new_q)
        
        # Age entanglements
        state = ageEntanglements(next_state, maxAge)
        print("State after aging:", state)
        
        # Try to generate new entanglements
        state = generateEntanglement(state, pGen)
        print("Final state for this step:", state)

# Run the test
test_q_learning()
