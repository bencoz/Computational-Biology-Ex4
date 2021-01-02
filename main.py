import sys
import numpy as np
from viterbi import viterbi_training
from .baum_welch import baum_welch


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


def run_algorithm(algorithm, sequence, parameters, epsilon):
    transition = build_transition_matrix(*parameters[:2])
    emission = build_emission_matrix(*parameters[2:])

    if algorithm == 'V':
        # viterbi_training(sequence, transition, emission, epsilon)
        pass
    elif algorithm == 'B':
        # baum_welch(sequence, transition, emission, epsilon)
        pass


if __name__ == "__main__":
    # Input validation
    algorithms = ['V', 'B']
    arguments_num = len(sys.argv)
    if arguments_num != 11:
        raise Exception("Arguments number is not valid:")
    sequence = str(sys.argv[1])
    algorithm = str(sys.argv[2])
    if algorithm not in algorithms:
        raise Exception("algorithm name is not valid (V/B are valid)")
    parameters = []
    for i in range(3, arguments_num):
        parameters.append(float(sys.argv[i]))

    epsilon = 1e-5
    print(" ".join(sequence))
    print('-' * len(sequence) * 2)
    run_algorithm(algorithm, sequence, parameters, epsilon)
    print('\nDone.')
