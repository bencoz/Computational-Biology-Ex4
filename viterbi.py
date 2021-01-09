import numpy as np
import math
from utils import mylog, build_emission_matrix, build_transition_matrix, print_model_params_header, print_model_params, \
    extract_model_params


def count_transitions_and_emissions(annotated_sequence):
    prev_state = 0
    T_IG, T_GI, E_IA, E_IT, E_IC, E_GA, E_GT, E_GC = 0, 0, 0, 0, 0, 0, 0, 0
    sum_transition_from_intergenic = 0
    sum_transition_from_gene = 0
    sum_emission_gene = 0
    sum_emission_int = 0
    for item in annotated_sequence:
        base = item[0]
        state_num = item[1]
        if state_num == 0:
            sum_emission_int += 1
            if base == 'A':
                E_IA += 1
            elif base == 'T':
                E_IT += 1
            elif base == 'C':
                E_IC += 1
        elif 2 <= state_num <= 4:
            sum_emission_gene += 1
            if base == 'A':
                E_GA += 1
            elif base == 'T':
                E_GT += 1
            elif base == 'C':
                E_GC += 1

        if prev_state == 0:
            sum_transition_from_intergenic += 1
            if state_num == 1:
                T_IG += 1

        if prev_state == 4:
            sum_transition_from_gene += 1
            if state_num == 5:
                T_GI += 1

        prev_state = state_num

    # To prevent dividing by zero
    if sum_transition_from_intergenic == 0:
        sum_transition_from_intergenic = 1
    if sum_transition_from_gene == 0:
        sum_transition_from_gene = 1
    if sum_emission_int == 0:
        sum_emission_int = 1
    if sum_emission_gene == 0:
        sum_emission_gene = 1

    return (T_IG / sum_transition_from_intergenic), (T_GI / sum_transition_from_gene), (E_IA / sum_emission_int), (
            E_IT / sum_emission_int), (E_IC / sum_emission_int), (E_GA / sum_emission_gene), (
                   E_GT / sum_emission_gene), (E_GC / sum_emission_gene)


def viterbi_training(s, transitions, emissions, epsilon):
    reach_epsilon = False
    previous_score = -math.inf
    print_model_params_header('V')
    print_one_time = True
    while not reach_epsilon:
        annotated_sequence, score = viterbi(s, transitions, emissions)
        if print_one_time:
            print_model_params(transitions, emissions, score)
            print_one_time = False

        T_IG, T_GI, E_IA, E_IT, E_IC, E_GA, E_GT, E_GC = count_transitions_and_emissions(annotated_sequence)

        # update matrices
        transitions = build_transition_matrix(T_IG, T_GI)
        emissions = build_emission_matrix(E_IA, E_IT, E_IC, E_GA, E_GT, E_GC)

        if math.fabs(score - previous_score) < epsilon:
            reach_epsilon = True
        previous_score = score

    if not print_one_time:
        print_model_params(transitions, emissions, score)
    return score, extract_model_params(transitions, emissions)


def viterbi(s, transitions, emissions):
    s_length = len(s)  # n.Rows
    num_of_states = len(emissions)  # k.Columns

    v = np.zeros((num_of_states, s_length), dtype=object)
    v[0, 0] = (math.log(1), -1)  # the tuple is to know from what i value in the previous column the maximum was chosen.
    # initialize v[0, j]
    for i in range(1, num_of_states):
        v[i, 0] = (mylog(emissions[0].get(s[0])), -1)  # there is no previous because this is the most left column.

    for i in range(1, len(s)):
        for j in range(0, num_of_states):
            curr_max = -math.inf
            max_prev_state_index = -1
            emission = emissions[j].get(s[i])
            for l in range(0, num_of_states):
                score = mylog(emission) + float(v[l, i - 1][0]) + mylog(transitions[l, j])

                if score > curr_max:
                    curr_max = score
                    max_prev_state_index = l
            v[j, i] = (curr_max, max_prev_state_index)

    last_column_max = -math.inf
    result = []
    # Find the max in the last column
    prev_index = -1
    for idx in range(num_of_states):
        if v[idx, len(s) - 1][0] > last_column_max:
            last_column_max = v[idx, len(s) - 1][0]
            prev_index = idx

    # Reconstructing
    for k in reversed(range(0, len(s))):
        result.append((
            s[k], prev_index, v[prev_index, k][0]
        ))
        prev_index = v[prev_index, k][1]

    result.reverse()

    return result, last_column_max
