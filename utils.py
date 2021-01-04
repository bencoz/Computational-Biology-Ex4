import math
import numpy as np
import sys

states = {
    'A': 0,
    'C': 1,
    'T': 2,
    'G': 3
}


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

