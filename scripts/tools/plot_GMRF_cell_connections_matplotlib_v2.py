import csv
import matplotlib.pyplot as plt
import numpy as np
import math

def max_value(inputlist):
    return max([sublist[-1] for sublist in inputlist])

filename = '/tmp/GMRF_v2.txt'
delimiter = ' '  # import a file with space delimiters
data = []

# quoting=csv.QUOTE_NONNUMERIC to automatically converto to float
for row in csv.reader(open(filename), delimiter=delimiter, quoting=csv.QUOTE_NONNUMERIC):
    data.append(row)

connection_size = len(data)
print("connection_size:", connection_size)

x = []
y = []

for i in range(connection_size):
    #a = (data[i][0], data[i][1])
    #b = (data[i][2], data[i][3])

    x.append(data[i][0])
    x.append(data[i][2])
    y.append(data[i][1])
    y.append(data[i][3])

    if i % 1000 == 0:
        print("{:.2f}%".format((i / float(connection_size)) * 100))

plt.plot(x, y, c=[0, 0, 1], linewidth=2)
plt.show()