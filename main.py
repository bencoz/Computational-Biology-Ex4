import sys
from viterbi import viterbi_training
from baum_welch import baum_welch
import random
import numpy as np, numpy.random
from utils import build_transition_matrix, build_emission_matrix


def create_random_probabilities():
    # Random probabilities
    # return T_IG, T_GI, E_IA, E_IT, E_IC, E_GA, E_GT, E_GC
    E_I = np.random.dirichlet(np.ones(4), size=1)[0]
    E_G = np.random.dirichlet(np.ones(4), size=1)[0]
    return [random.random(), random.random(), E_I[0], E_I[1], E_I[2], E_G[0], E_G[1], E_G[2]]


def run_algorithm(algorithm, sequence, parameters, epsilon):
    transition = build_transition_matrix(*parameters[:2])
    emission = build_emission_matrix(*parameters[2:])

    if algorithm == 'V':
        return viterbi_training(sequence, transition, emission, epsilon)
    elif algorithm == 'B':
        return baum_welch(sequence, transition, emission, epsilon)


def main(s=None, a=None):
    # Input validation
    algorithms = ['V', 'B']
    parameters = []
    arguments_num = len(sys.argv)
    if arguments_num == 1:
        algorithm = a
        sequence = s
        parameters = create_random_probabilities()
    else:
        if arguments_num == 3:
            parameters = create_random_probabilities()
        elif arguments_num != 11:
            raise Exception("Arguments number is not valid:")
        sequence = str(sys.argv[1])
        algorithm = str(sys.argv[2])
    if algorithm not in algorithms:
        raise Exception("algorithm name is not valid (V/B are valid)")

    for i in range(3, arguments_num):
        parameters.append(float(sys.argv[i]))

    epsilon = 1e-5
    print(" ".join(sequence))
    print('-' * len(sequence) * 2)
    score, final_params = run_algorithm(algorithm, sequence, parameters, epsilon)
    print('\nDone.')
    print(f"Score is: {format(score, '.2f')} and params are: {final_params}\n")


"This is for running from command line"
if __name__ == "__main__":
    main()

"This is only for experiments and should be commented out for running from command line "
# for i in range(0, 10):
#     s = "AAATTTTATTACGTTTAGTAGAAGAGAAAGGTAAACATGATGGTTCAGTGGTGCTAGATGAACAAACAATTATAAAATAAAATGAAGTATTTGTATAGAA"
#     algorithm = "V"
#     main(s, algorithm)
