import numpy as np
import random
from timeit import timeit
import matplotlib.pyplot as plt

'''
Comments only for first task, similar for others
'''

N = 2000  # maximum number of elements

v = [i for i in range(N)]  # generating array
random.shuffle(v)  # random shuffle for array
m = np.matrix([[random.random() for j in range(N)] for i in range(N)])  # generating martix


# constant time function
def foo1(v):
    return 42


experimental_time1 = []

for i in range(N + 1):
    tmp = v[:i]  # choosing first i elements of array
    experimental_time1 += [timeit(lambda: foo1(tmp), number=5)]  # measure experimental time

theoretical_time1 = [np.mean(experimental_time1) for n in range(N + 1)]  # calculate theoretical time

# create plot
plt.title('constant function', fontsize=20)
plt.xlabel('Number of elements')
plt.ylabel('Running time')
plt.plot(experimental_time1, color='red', label='experimental')
plt.plot(theoretical_time1, color='green', linewidth=4, label='theoretical')
plt.legend(loc='upper left')
plt.show()


# linear time function (sum)
def foo2(v):
    return sum(v)


experimental_time2 = []

for i in range(N + 1):
    tmp = v[:i]
    experimental_time2 += [timeit(lambda: foo2(tmp), number=5)]

# theoretical_time2 = [experimental_time2[-1] / N * n for n in range(N + 1)]
#
# plt.title('sum of elements', fontsize=20)
# plt.xlabel('Number of elements')
# plt.ylabel('Running time')
# plt.plot(experimental_time2, color='red', label='experimental')
# plt.plot(theoretical_time2, color='green', linewidth=4, label='theoretical')
# plt.legend(loc='upper left')
# plt.show()
#
#
# # linear time function (prod)
# def foo3(v):
#     return np.prod(v)
#
#
# experimental_time3 = []
#
# for i in range(N + 1):
#     tmp = v[:i]
#     experimental_time3 += [timeit(lambda: foo3(tmp), number=5)]
#
# theoretical_time3 = [experimental_time3[-1] / N * n for n in range(N + 1)]
#
# plt.title('prod of elements', fontsize=20)
# plt.xlabel('Number of elements')
# plt.ylabel('Running time')
# plt.plot(experimental_time3, color='red', label='experimental')
# plt.plot(theoretical_time3, color='green', linewidth=4, label='theoretical')
# plt.legend(loc='upper left')
# plt.show()
#
#
# # linear time function (polynom)
# def foo4(v):
#     return np.poly1d(v, 1.5)
#
#
# experimental_time4 = []
#
# for i in range(N + 1):
#     tmp = v[:i]
#     experimental_time4 += [timeit(lambda: foo4(tmp), number=5)]
#
# theoretical_time4 = [experimental_time4[-1] / N * n for n in range(N + 1)]
#
# plt.title('polynom', fontsize=20)
# plt.xlabel('Number of elements')
# plt.ylabel('Running time')
# plt.plot(experimental_time4, color='red', label='experimental')
# plt.plot(theoretical_time4, color='green', linewidth=4, label='theoretical')
# plt.legend(loc='upper left')
# plt.show()


# linear time function (Horner)
def foo5(v):
    return np.polynomial.polynomial.polyval(v, 1.5)


experimental_time5 = []

for i in range(N + 1):
    tmp = v[:i]
    experimental_time5 += [timeit(lambda: foo5(tmp), number=5)]

theoretical_time5 = [experimental_time5[-1] / N * n for n in range(N + 1)]

plt.title('Horner', fontsize=20)
plt.xlabel('Number of elements')
plt.ylabel('Running time')
plt.plot(experimental_time5, color='red', label='experimental')
plt.plot(theoretical_time5, color='green', linewidth=4, label='theoretical')
plt.legend(loc='upper left')
plt.show()


# O(n**2) bubble sort
def foo6(v):
    for passesLeft in range(len(v) - 1, 0, -1):
        for index in range(passesLeft):
            if v[index] < v[index + 1]:
                v[index], v[index + 1] = v[index + 1], v[index]
    return v


experimental_time6 = []

for i in range(N + 1):
    tmp = v[:i]
    experimental_time6 += [timeit(lambda: foo6(tmp), number=5)]

theoretical_time6 = [experimental_time6[-1] / (N ** 2) * n * n for n in range(N + 1)]

plt.title('bubble sort', fontsize=20)
plt.xlabel('Number of elements')
plt.ylabel('Running time')
plt.plot(experimental_time6, color='red', label='experimental')
plt.plot(theoretical_time6, color='green', linewidth=4, label='theoretical')
plt.legend(loc='upper left')
plt.show()


# O(nlogn) quick sort
def foo7(v):
    return np.sort(v, kind='quicksort')


experimental_time7 = []

for i in range(N + 1):
    tmp = v[:i]
    experimental_time7 += [timeit(lambda: foo7(tmp), number=5)]

theoretical_time7 = [experimental_time7[-1] / N / np.log(N) * n * np.log(n) for n in range(N + 1)]

plt.title('quick sort', fontsize=20)
plt.xlabel('Number of elements')
plt.ylabel('Running time')
plt.plot(experimental_time7, color='red', label='experimental')
plt.plot(theoretical_time7, color='green', linewidth=4, label='theoretical')
plt.legend(loc='upper left')
plt.show()


# O(nlogn) quick sort
def foo8(v):
    v.sort()


experimental_time8 = []

for i in range(N + 1):
    tmp = v[:i]
    experimental_time8 += [timeit(lambda: foo8(tmp), number=5)]

theoretical_time8 = [experimental_time8[-1] / N / np.log(N) * n * np.log(n) for n in range(N + 1)]

plt.title('tim sort', fontsize=20)
plt.xlabel('Number of elements')
plt.ylabel('Running time')
plt.plot(experimental_time8, color='red', label='experimental')
plt.plot(theoretical_time8, color='green', linewidth=4, label='theoretical')
plt.legend(loc='upper left')
plt.show()


# O(n**3) matrix prod
def foo9(m):
    return m * m


experimental_time9 = []

for i in range(N + 1):
    tmp = m[:i, :i]
    experimental_time9 += [timeit(lambda: foo9(tmp), number=5)]

theoretical_time9 = [experimental_time9[-1] / (N ** 3) * (n ** 3) for n in range(N + 1)]

plt.title('matrix prod', fontsize=20)
plt.xlabel('Number of elements')
plt.ylabel('Running time')
plt.plot(experimental_time9, color='red', label='experimental')
plt.plot(theoretical_time9, color='green', linewidth=4, label='theoretical')
plt.legend(loc='upper left')
plt.show()
