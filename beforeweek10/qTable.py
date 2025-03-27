from collections import defaultdict
from typing import Dict, List, Tuple, Optional

class QTable:
    def __init__(self, default_q_value: float = 0.0):
        self.q_table = defaultdict(lambda: defaultdict(lambda: default_q_value))
    
    def get_q_value(self, state: Dict, action: Tuple) -> float:
        # Convert state and action to hashable format
        state_key = tuple(sorted(state.items()))
        action_key = (tuple(sorted(action[0])), action[1])
        return self.q_table[state_key][action_key]
    
    def update_q_value(self, state: Dict, action: Tuple, new_value: float) -> None:
        state_key = tuple(sorted(state.items()))
        action_key = (tuple(sorted(action[0])), action[1])
        self.q_table[state_key][action_key] = new_value
    
    def getActionAndValues(self, state: Dict) -> List[Tuple[Tuple, float]]:
        state_key = tuple(sorted(state.items()))
        return [(
            (list(action_key[0]), action_key[1]), # Convert action back to original format
            value
        ) for action_key, value in self.q_table[state_key].items()]