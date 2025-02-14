

import numpy as np

d = np.arange(0, 256)
r = np.arange(0, 256)

x = np.zeros([256, 256])


def hamming_distance(str1, str2):
    if len(str1) != len(str2):
        raise ValueError("Strings must be of equal length")
    
    distance = 0
    for c1, c2 in zip(str1, str2):
        if c1 != c2:
            distance += 1
    return distance

# Example usage
binary_str1 = "101101"
binary_str2 = "100100"
print(hamming_distance(binary_str1, binary_str2))  # Output would be 2


for r in d:
    for c in d:
        rb = format(r, '#010b')[2:] 
        cb = format(c, '#010b')[2:]     
        x[r, c] = hamming_distance(rb, cb)



import pandas as pd

# Convert to DataFrame
df = pd.DataFrame(x)


np.savetxt('hamming_matrix.csv', x, delimiter=',', fmt='%d')


# visit this link for a nice visual of this matrix: https://docs.google.com/spreadsheets/u/1/d/11sq_Do77Juzc0UfIx5hqXUzkdIf7-V3gIP1qcp03wnE/edit?usp=drive_web&ouid=117246686464735992968