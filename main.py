import sys
from viterbi import viterbi_training
from baum_welch import baum_welch
import random
import numpy as np, numpy.random
import math
import datetime
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
    print(f"Score is: {format(score, '.4f')} and params are: {final_params}\n")
    return score, final_params


"This is for running from command line"
if __name__ == "__main__":
    main()

"This is only for experiments and should be commented out for running from command line "
# best_score = -math.inf
# best_final_params = None
# start = datetime.datetime.now()
# start_time = start.strftime("%H:%M:%S.%f")
# print("Parameter inference – start time is", start_time)
# n_total_iterations = 1000
# for i in range(0, n_total_iterations):
#     s = "CGCACACGTCCTTGAGGGCAGTTTTTTTGTCGCCCCCACGATTTTTCTCGGCCGCAGTTCCCGTTTTTTTTTGTTTTTTTTGTTGGCCTCTGGTTTTCTACGAGGCCGGGGAGAGGCCGGGGCGGCAGATTTTCTTGTTTTTCAGGATTGCTGGTTTGCTCAGTGTTTTTCTTCTTTGTTTGGCTGTGCCGGAAGAGATG"
#     algorithm = "B"
#     score, params = main(s, algorithm)
#     if score > best_score:
#         best_score = score
#         best_final_params = params
#     if (i + 1) % 100 == 0:
#         print(f"Iteration [{i + 1}/{n_total_iterations}] had passed")
#
# end = datetime.datetime.now()
# end_time = end.strftime("%H:%M:%S.%f")
# print("Parameter inference end time is", end_time)
# print("Parameter inference time was", str(end-start))
# print(f"Final Best score is: {format(best_score, '.4f')}\nBest final params are: {best_final_params}\n")
