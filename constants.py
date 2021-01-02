import numpy as np

transition_matrix = np.array([
    # S0    #S1    #S2    #S3      #S4     #S5
    [0.95, 0.05, 0.0, 0.0, 0.0, 0.0],  # InterGen(S0)
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # A(S1)
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # Codon1(S2)
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # Codon2(S3)
    [0.0, 0.0, 0.8, 0.0, 0.0, 0.2],  # Codon3(S4)
    [0.95, 0.05, 0.0, 0.0, 0.0, 0.0]  # T(S5)
])

emission_matrix = np.array([
     #A        #C       #T        #G
    {'A': 0.3, 'C': 0.2, 'T': 0.3, 'G': 0.2},  # InterGen(S0)
    {'A': 1.0, 'C': 0.0, 'T': 0.0, 'G': 0.0},  # A(S1)
    {'A': 0.0, 'C': 0.4, 'T': 0.2, 'G': 0.4},  # Codon1(S2)
    {'A': 0.0, 'C': 0.4, 'T': 0.2, 'G': 0.4},  # Codon2(S3)
    {'A': 0.0, 'C': 0.4, 'T': 0.2, 'G': 0.4},  # Codon3(S4)
    {'A': 0.0, 'C': 0.0, 'T': 1.0, 'G': 0.0},  # T(S5)
])

emission_matrix2 = np.array([
     #A         #C        #T         #G
    {'A': 0.3,  'C': 0.2, 'T': 0.3,  'G': 0.2},  # InterGen(S0)
    {'A': 1.0,  'C': 0.0, 'T': 0.0,  'G': 0.0},  # A(S1)
    {'A': 0.05, 'C': 0.4, 'T': 0.15, 'G': 0.4},  # Codon1(S2)
    {'A': 0.05, 'C': 0.4, 'T': 0.15, 'G': 0.4},  # Codon2(S3)
    {'A': 0.05, 'C': 0.4, 'T': 0.15, 'G': 0.4},  # Codon3(S4)
    {'A': 0.0,  'C': 0.0, 'T': 1.0,  'G': 0.0},  # T(S5)
])

sequence = "CCATCGCACTCCGATGTGGCCGGTGCTCACGTTGCCT"
