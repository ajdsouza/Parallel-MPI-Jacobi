from __future__ import print_function
import numpy
import time
import sys


def solve(n):
    A = numpy.random.randn(n, n)
    b = numpy.random.randn(n)
    start = time.time()
    x = numpy.linalg.solve(A, b)
    end = time.time()
    print("Runtime for n = %d: %d" % (n, end - start))


def jacobi(A, b, max_iter=1024, termination_l2=1.0e-12):
    # get size
    n = len(A[0])

    # initialize variables
    start = time.time()
    x = numpy.zeros(n)
    D = numpy.diag(A)
    R = A - numpy.diagflat(D)

    # use jacobi till maxiteration or termination criteria
    for i in range(0, max_iter):
        x = (b - numpy.dot(R, x))/D
        if (numpy.linalg.norm(numpy.dot(A, x) - b) < termination_l2):
            break
    end = time.time()
    l2 = numpy.linalg.norm(numpy.dot(A, x) - b)
    print("Jacobi: n = %d: time %fs, l2 %e" % (n, end - start, l2))
    return x


def jacobi_test(n):
    A = 100.0*numpy.random.randn(n, n)
    b = numpy.random.randn(n)
    row_sums = numpy.linalg.norm(A, 1)
    # make sure A is diagonally dominant
    A = A + row_sums*numpy.identity(n)
    #print(A)
    start = time.time()
    x = jacobi(A, b)
    end = time.time()
    #print(x)
    print("Total Runtime for n = %d: %d" % (n, end - start))


def write_rand_problem(n, name="input"):
    A = rand_matrix(n)
    A.tofile(name + "_A.bin")
    b = rand_result(n)
    b.tofile(name + "_b.bin")
    x = jacobi(A, b)
    print(A)
    print(b)
    print(numpy.dot(A, b))


def check_result(file_A, file_b, file_x):
    # read input
    b = numpy.fromfile(file_b)
    print(b)
    n = b.size
    A = numpy.fromfile(file_A)
    if A.size != n*n:
        print("File dimensions do not match")
        return
    A = A.reshape((n, n))
    print(A)
    x = numpy.fromfile(file_x)
    if x.size != n:
        print("File dimensions do not match")
        return
    # check result
    if n < 256:
        real_x = numpy.linalg.solve(A, b)
    else:
        real_x = jacobi(A, b)

    if not numpy.allclose(x, real_x, 1e-10):
        print("Error: output is not close to input")
    else:
        print("Output is correct!")


def main():
    if len(sys.argv) < 4:
        print("Usage: python check_output.py <input_A> <input_b> <output_x>")
        return
    check_result(sys.argv[1], sys.argv[2], sys.argv[3])

if __name__ == '__main__':
    main()
