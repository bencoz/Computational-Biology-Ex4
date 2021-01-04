import numpy as np
import math
import sys
from utils import states, mylog, build_emission_matrix, build_transition_matrix, print_model_params_header, \
    print_model_params


def count_emissions(s, f, b, emission_count, num_of_states):
    for i in range(0, len(s)):
        for j in range(num_of_states):
            f_cell = np.exp(f[j, i])
            b_cell = np.exp(b[j, i])

            emission_count[j, states[s[i]]] += f_cell * b_cell
    return emission_count


def baum_welch(s, transitions, emissions, epsilon):
    reach_epsilon = False
    previous_score = -math.inf
    print_model_params_header('B')
    while not reach_epsilon:

        f, score = forward(s, transitions, emissions)
        b = backward(s, transitions, emissions)

        print_model_params(transitions, emissions, score)

        num_of_states = len(emissions)
        emission_count = np.zeros((num_of_states, 4), dtype=float)

        emission_count = count_emissions(s, f, b, emission_count, num_of_states)

        if math.fabs(score - previous_score) < epsilon:
            reach_epsilon = True
        previous_score = score


def forward(s, transitions, emissions):
    s_length = len(s)  # n.Rows
    num_of_states = len(emissions)  # k.Columns

    f = np.zeros((num_of_states, s_length), dtype=float)

    # initialize f[0, i]
    # Regular
    # f[0, 0] = 1
    f[0, 0] = math.log(1)
    for i in range(1, num_of_states):
        # Regular
        # f[i, 0] = 0
        f[i, 0] = mylog(0)

    for i in range(1, len(s)):
        for j in range(0, num_of_states):
            emission = emissions[j].get(s[i])
            a_max = sys.float_info.min
            a_l = []
            for l in range(0, num_of_states):
                curr = f[l, i - 1] + mylog(transitions[l, j])
                if curr > a_max:
                    a_max = curr
                a_l.append(curr)

                # Regular
                # f[j, i] += f[l, i - 1] * transitions[l, j] * emission

            f[j, i] = 0
            for l in range(0, num_of_states):
                b_l = a_l[l] - a_max
                f[j, i] += math.exp(b_l)

            f[j, i] = mylog(f[j, i]) + a_max + mylog(emission)

    likelihood = 0
    for i in range(0, num_of_states):
        curr = f[i, len(s) - 1]
        if curr > -math.inf:
            likelihood += curr
    # print(f"forward likelihood is: {likelihood}")
    return f, likelihood


def backward(s, transitions, emissions):
    s_length = len(s)  # n.Rows
    num_of_states = len(emissions)  # k.Columns

    b = np.zeros((num_of_states, s_length), dtype=float)

    # initialize b[n, i]
    for i in range(0, num_of_states):
        b[i, len(s) - 1] = math.log(1)  # The most left column

    for i in reversed(range(0, len(s) - 1)):
        for j in range(0, num_of_states):
            a_max = sys.float_info.min
            a_l = []
            for l in range(0, num_of_states):
                emission = emissions[l].get(
                    s[i + 1])  # emission inserted into the "l" for because he is being dependent on l

                curr = b[l, i + 1] + mylog(transitions[j, l]) + mylog(emission)
                if curr > a_max:
                    a_max = curr
                a_l.append(curr)

                # Regular
                # b[j, i] += b[l, i + 1] * transition * emission

            b[j, i] = 0
            for l in range(0, num_of_states):
                b_l = a_l[l] - a_max
                b[j, i] += math.exp(b_l)

            b[j, i] = mylog(b[j, i]) + a_max

    return b
