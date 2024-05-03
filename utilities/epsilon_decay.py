import numpy as np

def linear_decay(initial_epsilon, final_epsilon, current_epoch, total_epochs):
    if initial_epsilon == final_epsilon:
        return initial_epsilon
    elif total_epochs == 1:
        return final_epsilon
    else:
        rate_of_change = (final_epsilon - initial_epsilon) / (total_epochs-1)
        current_epsilon = np.round((initial_epsilon - rate_of_change) + (rate_of_change * current_epoch),3)
        
        if current_epsilon > initial_epsilon or current_epsilon < final_epsilon:
            raise ValueError(f'Epsilon value ({current_epsilon}) out of valid range ({initial_epsilon}:{final_epsilon})')
    
        return current_epsilon 