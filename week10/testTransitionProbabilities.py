from pre_computation import getTransitionProbabilities, getPossibleStates
from collections import defaultdict

def analyze_transitions(state, pSwap, pGen, goalEdges, maxAge=2):
    """Analyze transitions for a given state and probability parameters"""
    transitions = getTransitionProbabilities(state, pSwap, pGen, goalEdges, maxAge)
    
    print(f"\n=== Analysis for pSwap={pSwap}, pGen={pGen} ===")
    
    # Track probabilities
    total_prob = 0
    success_probs = defaultdict(float)  # {num_goals: prob}
    gen_probs = defaultdict(float)  # {num_generated: prob}
    
    print("\nDetailed Transitions:")
    for ((orig_state, action), (next_state, prob)) in transitions:
        total_prob += prob
        
        # Count successful goals
        num_goals = len(action['goals_achieved'])
        success_probs[num_goals] += prob
        
        # Count generations (only positive generations)
        empty_edges = sum(1 for _, age in dict(next_state).items() if age == 1)
        gen_probs[empty_edges] += prob
        
        print(f"\nProb: {prob:.4f}")
        print(f"Goals: {action['goals_achieved']}")
        if action['swap_edges_per_goal']:
            print(f"Swap paths: {action['swap_edges_per_goal']}")
        print(f"Next state: {dict(next_state)}")
    
    print("\nProbability Analysis:")
    print(f"Total probability: {total_prob:.6f} (should be 1.0)")
    
    print("\nGoal Achievement Distribution:")
    for goals, prob in sorted(success_probs.items()):
        print(f"{goals} goals: {prob:.4f}")
    print(f"Sum of goal probs: {sum(success_probs.values()):.4f} (should be 1.0)")
    
    print("\nGeneration Distribution:")
    for num_gen, prob in sorted(gen_probs.items()):
        print(f"{num_gen} new edges: {prob:.4f}")
    print(f"Sum of gen probs: {sum(gen_probs.values()):.4f} (should be 1.0)")
    
    return total_prob, success_probs, gen_probs

def main():
    # Test setup
    initialEdges = [(1,2), (2,3), (3,4)]
    goalEdges = [(1,4)]
    maxAge = 2
    
    # Get a test state where all edges have entanglement
    test_state = tuple(sorted({(1,2): 1, (2,3): 1, (3,4): 1}.items()))
    
    # Test both scenarios
    print("\nTesting high generation probability...")
    high_gen = analyze_transitions(test_state, pSwap=1.0, pGen=0.9, goalEdges=goalEdges)
    
    print("\nTesting low generation probability...")
    low_gen = analyze_transitions(test_state, pSwap=1.0, pGen=0.1, goalEdges=goalEdges)
    
    # Compare results
    print("\n=== Comparison ===")
    print("With pSwap=1.0, we expect:")
    print("1. All swap attempts should succeed")
    print("2. Generation probabilities should differ significantly:")
    print("   - pGen=0.9 should show higher probability of generating edges")
    print("   - pGen=0.1 should show lower probability of generating edges")

if __name__ == "__main__":
    main()
