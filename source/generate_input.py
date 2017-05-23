from __future__ import print_function
import numpy
import time
import sys


def rand_matrix(n):
    A = 100.0*numpy.random.randn(n, n)
    row_sums = numpy.linalg.norm(A, 1)
    # make sure A is diagonally dominant
    A = A + row_sums*numpy.identity(n)
    return numpy.around(A)


def rand_result(n):
    b = 100.0*numpy.random.randn(n)
    return numpy.around(b)


def write_rand_problem(n, name="input"):
    A = rand_matrix(n)
    A.tofile(name + "_A.bin")
    b = rand_result(n)
    b.tofile(name + "_b.bin")
    print("A=")
    print(A)
    print("b=")
    print(b)


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_input.py <n>")
        return
    n = int(sys.argv[1])
    write_rand_problem(n)


if __name__ == '__main__':
    main()
