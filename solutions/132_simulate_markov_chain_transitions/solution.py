import numpy as np
def simulate_markov_chain(transition_matrix, initial_state, num_steps):
    states = [initial_state]
    for i in range(num_steps):
        next_state = np.random.choice(len(transition_matrix), p=transition_matrix[states[-1]])
        states.append(next_state)
    return states