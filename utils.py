import math
import numpy as np
import sys

def mylog(x):
    try:
        res = math.log(x)
        return res
    except ValueError:
        return -math.inf


def build_transition_matrix(T_IG, T_GI):
    return np.array([
        # S0        #S1     #S2         #S3     #S4     #S5
        [1 - T_IG, T_IG, 0.0, 0.0, 0.0, 0.0],  # InterGen(S0)
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # A(S1)
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # Codon1(S2)
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # Codon2(S3)
        [0.0, 0.0, 1 - T_GI, 0.0, 0.0, T_GI],  # Codon3(S4)
        [1 - T_IG, T_IG, 0.0, 0.0, 0.0, 0.0]  # T(S5)
    ])


def build_emission_matrix(E_IA, E_IT, E_IC, E_GA, E_GT, E_GC):
    return np.array([
        # A         #C          #T         #G
        {'A': E_IA, 'C': E_IC, 'T': E_IT, 'G': 1 - (E_IA + E_IC + E_IT)},  # InterGen(S0)
        {'A': 1.0, 'C': 0.0, 'T': 0.0, 'G': 0.0},  # A(S1)
        {'A': E_GA, 'C': E_GC, 'T': E_GT, 'G': 1 - (E_GA + E_GC + E_GT)},  # Codon1(S2)
        {'A': E_GA, 'C': E_GC, 'T': E_GT, 'G': 1 - (E_GA + E_GC + E_GT)},  # Codon2(S3)
        {'A': E_GA, 'C': E_GC, 'T': E_GT, 'G': 1 - (E_GA + E_GC + E_GT)},  # Codon3(S4)
        {'A': 0.0, 'C': 0.0, 'T': 1.0, 'G': 0.0},  # T(S5)
    ])


def print_model_params_header(algorithm):
    if algorithm == 'V':
        print("|\tT_IG\tT_GI\tE_IA\tE_IT\tE_IC\tE_GA\tE_GT\tE_GC\t\tscore (Viterbi score)")
    elif algorithm == 'B':
        print("|\tT_IG\tT_GI\tE_IA\tE_IT\tE_IC\tE_GA\tE_GT\tE_GC\t\tscore (log likelihood)")


def print_model_params(transition, emission, score):
    T_IG = format(transition[0, 1], '.2f')
    T_GI = format(transition[4, 5], '.2f')
    E_IA = format(emission[0]['A'], '.2f')
    E_IT = format(emission[0]['T'], '.2f')
    E_IC = format(emission[0]['C'], '.2f')
    E_GA = format(emission[2]['A'], '.2f')
    E_GT = format(emission[2]['T'], '.2f')
    E_GC = format(emission[2]['C'], '.2f')
    score_str = format(score, '.4f')
    print(f"|\t{T_IG}\t{T_GI}\t{E_IA}\t{E_IT}\t{E_IC}\t{E_GA}\t{E_GT}\t{E_GC}\t\t{score_str}\t|")

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
    #print(f"forward likelihood is: {likelihood}")
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
                emission = emissions[l].get(s[i + 1])  # emission inserted into the "l" for because he is being dependent on l

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