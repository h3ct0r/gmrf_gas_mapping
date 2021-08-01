import csv
import matplotlib.pyplot as plt
import numpy as np
import math

def max_value(inputlist):
    return max([sublist[-1] for sublist in inputlist])

filename = '/tmp/GMRF.txt'
delimiter = ' '  # import a file with space delimiters
data = []
# quoting=csv.QUOTE_NONNUMERIC to automatically converto to float
for row in csv.reader(open(filename), delimiter=delimiter, quoting=csv.QUOTE_NONNUMERIC):
    data.append(row)

N = max_value(data)

n_x_cells = int(data[1][1])
n_y_cells = int(N/n_x_cells)

print("n_x_cells:", n_x_cells, "n_y_cells:", n_y_cells)

for i in range(len(data)):
    a = data[i][0]
    b = data[i][1]
    if a < b:
        plt.plot([np.mod(a, n_x_cells), np.mod(b,n_x_cells)], [math.floor(a/n_x_cells), math.floor(b/n_x_cells)], c=[0, 0, 1], linewidth=2)

plt.show()