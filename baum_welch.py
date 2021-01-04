import numpy as np
import math
import sys
from utils import forward, backward, mylog, build_emission_matrix, build_transition_matrix, print_model_params_header, print_model_params


# A         #C          #T         #G
States = {
    'A': 0,
    'C': 1,
    'T': 2,
    'G': 3
}



def count_emissions(s, f, b, emission_count, num_of_states):
    for i in range(0, len(s)):
        for j in range(num_of_states):
            f_cell = np.exp(f[j, i])
            b_cell = np.exp(b[j, i])

            emission_count[j, States[s[i]]] += f_cell * b_cell
    return emission_count


def baum_welch(s, transitions, emissions, epsilon):
    print_model_params_header('B')
    f, likelihood = forward(s, transitions, emissions)
    b = backward(s, transitions, emissions)

    num_of_states = len(emissions)
    emission_count = np.zeros((num_of_states, 4), dtype=float)

    emission_count = count_emissions(s, f, b, emission_count, num_of_states)






    pass
