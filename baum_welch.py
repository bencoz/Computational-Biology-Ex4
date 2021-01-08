import numpy as np
import math
import sys
from utils import mylog, build_emission_matrix, build_transition_matrix, print_model_params_header, \
    print_model_params, myexp, extract_model_params


def compute_expected_transitions_counts(s, transitions, emissions, f, b):
    # E[N_jl] = T_jl * sum_i_of(F[i,j]*E_lx_i+1*B[i+1,l])
    # we normalize at the end so no division of sum_j_of(F[n,j])
    T_IG = transitions[0, 1]
    T_II = transitions[0, 0]
    T_GI = transitions[4, 5]
    T_GG = transitions[4, 2]
    sum_intergenic_to_gene, sum_intergenic_to_intergenic = 0, 0
    sum_gene_to_intergenic, sum_gene_to_gene = 0, 0
    for i in range(0, len(s) - 1):
        sum_intergenic_to_gene += myexp(f[i, 0] + b[i + 1, 1]) * emissions[1][s[i + 1]]
        sum_intergenic_to_intergenic += myexp(f[i, 0] + b[i + 1, 0]) * emissions[0][s[i + 1]]

        sum_intergenic_to_gene += myexp(f[i, 5] + b[i + 1, 1]) * emissions[1][s[i + 1]]
        sum_intergenic_to_intergenic += myexp(f[i, 5] + b[i + 1, 0]) * emissions[0][s[i + 1]]

        sum_gene_to_intergenic += myexp(f[i, 4] + b[i + 1, 5]) * emissions[5][s[i + 1]]
        sum_gene_to_gene += myexp(f[i, 4] + b[i + 1, 2]) * emissions[2][s[i + 1]]

    N_IG = T_IG * sum_intergenic_to_gene
    N_II = T_II * sum_intergenic_to_intergenic
    N_GI = T_GI * sum_gene_to_intergenic
    N_GG = T_GG * sum_gene_to_gene

    return (N_IG / (N_IG + N_II)), (N_GI / (N_GI + N_GG))


def compute_expected_emissions_counts(s, f, b):
    # E[N_jo] = sum_i|Xi=o_of(F[i,j]*B[i,j])
    # we normalize at the end so no division of sum_j_of(F[n,j])
    sum_intergenic_a, sum_intergenic_t, sum_intergenic_c, sum_intergenic_g = 0, 0, 0, 0
    sum_gene_a, sum_gene_t, sum_gene_c, sum_gene_g = 0, 0, 0, 0
    for i in range(0, len(s)):
        if s[i] == 'A':
            sum_intergenic_a += myexp(f[i, 0] + b[i, 0])
            sum_gene_a += myexp(f[i, 2] + b[i, 2]) + myexp(f[i, 3] + b[i, 3]) + myexp(f[i, 4] + b[i, 4])
        elif s[i] == 'T':
            sum_intergenic_t += myexp(f[i, 0] + b[i, 0])
            sum_gene_t += myexp(f[i, 2] + b[i, 2]) + myexp(f[i, 3] + b[i, 3]) + myexp(f[i, 4] + b[i, 4])
        elif s[i] == 'C':
            sum_intergenic_c += myexp(f[i, 0] + b[i, 0])
            sum_gene_c += myexp(f[i, 2] + b[i, 2]) + myexp(f[i, 3] + b[i, 3]) + myexp(f[i, 4] + b[i, 4])
        else:  # s[i] == 'G'
            sum_intergenic_g += myexp(f[i, 0] + b[i, 0])
            sum_gene_g += myexp(f[i, 2] + b[i, 2]) + myexp(f[i, 3] + b[i, 3]) + myexp(f[i, 4] + b[i, 4])

    sum_gene = sum_gene_a + sum_gene_t + sum_gene_c + sum_gene_g
    sum_intergenic = sum_intergenic_a + sum_intergenic_t + sum_intergenic_c + sum_intergenic_g
    return (sum_intergenic_a / sum_intergenic), (sum_intergenic_t / sum_intergenic), (
            sum_intergenic_c / sum_intergenic), (sum_gene_a / sum_gene), (sum_gene_t / sum_gene), (
                   sum_gene_c / sum_gene)


def baum_welch(s, transitions, emissions, epsilon):
    reach_epsilon = False
    previous_score = -math.inf
    print_model_params_header('B')
    while not reach_epsilon:

        f, score = forward(s, transitions, emissions)
        b = backward(s, transitions, emissions)

        print_model_params(transitions, emissions, score)

        T_IG, T_GI = compute_expected_transitions_counts(s, transitions, emissions, f, b)
        E_IA, E_IT, E_IC, E_GA, E_GT, E_GC = compute_expected_emissions_counts(s, f, b)

        # update matrices
        transitions = build_transition_matrix(T_IG, T_GI)
        emissions = build_emission_matrix(E_IA, E_IT, E_IC, E_GA, E_GT, E_GC)

        if math.fabs(score - previous_score) < epsilon:
            reach_epsilon = True
        previous_score = score

    return score, extract_model_params(transitions, emissions)


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
                f[j, i] += myexp(b_l)

            f[j, i] = mylog(f[j, i]) + a_max + mylog(emission)

    likelihood = 0
    for i in range(0, num_of_states):
        curr = f[i, len(s) - 1]
        if curr > -math.inf:
            likelihood += myexp(curr)
    # print(f"forward likelihood is: {likelihood}")
    return f.T, mylog(likelihood)


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
                b[j, i] += myexp(b_l)

            b[j, i] = mylog(b[j, i]) + a_max

    return b.T
