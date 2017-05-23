/**
 * @file    mpi_jacobi.h
 * @brief   Implements MPI functions for distributing vectors and matrixes,
 *          parallel distributed matrix-vector multiplication and Jacobi's
 *          method.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 *
 */

/*********************************************************************
 *                  !!  DO NOT CHANGE THIS FILE  !!                  *
 *********************************************************************/

#ifndef MPI_JACOBI_H
#define MPI_JACOBI_H

#include <mpi.h>

/**
 * @brief   Equally distributes a vector stored on processor (0,0) onto
 *          processors (i,0) [i.e., the first column of the processor grid].
 *
 * Block distributes the input vector of size `n` from process (0,0) to
 * processes (i,0), i.e., the first column of the 2D grid communicator.
 *
 * Let `q` be the number of processes in the first dimension of the given
 * grid communicator.
 *
 * Elements are equally distributed, such that ceil(n/q) elements go to the `n
 * mod q` first processes and floor(n/q) elements go to the remaining
 * processes.
 *
 * Example for 3x3 processor grid:
 *
 * Input:
 *    ________________________________________
 *   |   (0,0)    |             |   (0,2)     |
 *   |   v[1:n]   |             |             |
 *   |------------|-------------|-------------|
 *   |            |             |             |
 *   |            |             |             |
 *   |------------|-------------|-------------|
 *   |   (2,0)    |             |   (2,2)     |
 *   |            |             |             |
 *   |------------|-------------|-------------|
 *
 *
 * Output:
 *    ________________________________________
 *   |            |             |             |
 *   | v[1:n/3]   |             |             |
 *   |------------|-------------|-------------|
 *   |            |             |             |
 *   | v[n/3:2n/3]|             |             |
 *   |------------|-------------|-------------|
 *   |            |             |             |
 *   | v[2n/3:n]  |             |             |
 *   |------------|-------------|-------------|
 *
 * @param n             The size of the input vector.
 * @param input_vector  The input vector of length `n`, only on processor (0,0).
 * @param local_vector  The local output vector of size floor(n/q) or
 *                      ceil(n/q), where `q` is the number of processors in the
 *                      first dimension of the 2d grid communicator. This
 *                      has to be allocated on the processors (i,0) according
 *                      to their block distirbuted size.
 * @param comm          A 2d cartesian grid communicator.
 */
void distribute_vector(const int n, double* input_vector, double** local_vector,
                       MPI_Comm comm);

/**
 * @brief Reverts the operation of `distribute_vector()`.
 *
 * Gathers the vector `local_vector`, which is distributed among the first
 * column of processes (i,0), onto the process with rank (0,0).
 *
 * Example for 3x3 processor grid:
 *
 * Input:
 *    ________________________________________
 *   |            |             |             |
 *   | v[1:n/3]   |             |             |
 *   |------------|-------------|-------------|
 *   |            |             |             |
 *   | v[n/3:2n/3]|             |             |
 *   |------------|-------------|-------------|
 *   |            |             |             |
 *   | v[2n/3:n]  |             |             |
 *   |------------|-------------|-------------|
 *
 * Output:
 *    ________________________________________
 *   |            |             |             |
 *   |   v[1:n]   |             |             |
 *   |------------|-------------|-------------|
 *   |            |             |             |
 *   |            |             |             |
 *   |------------|-------------|-------------|
 *   |            |             |             |
 *   |            |             |             |
 *   |------------|-------------|-------------|
 *
 * @param n             The total size of the output vector.
 * @param local_vector  The local input vector of size floor(n/q) or
 *                      ceil(n/q), where `q` is the number of processors in the
 *                      first dimension of the 2d grid communicator.
 * @param output_vector The output vector of length `n`, only on processor (0,0).
 * @param comm          A 2d cartesian grid communicator.
 */
void gather_vector(const int n, double* local_vector, double* output_vector,
                   MPI_Comm comm);

/**
 * @brief   Equally distributes a matrix stored on processor (0,0) onto the
 *          whole grid (i,j).
 *
 * Block distributes the input matrix of size n-by-n, stored in row-major
 * format onto a 2d communicator grid of size q-by-q with a total of
 * p = q*q processes.
 *
 * The matrix is decomposed into blocks of sizes n1-by-n2, where both `n1` and
 * `n2` can be either ceil(n/q) or floor(n/q). The `n` rows of the matrix are
 * block decomposed among the `q` rows of the 2d grid. The first `n mod q`
 * processes will have ceil(n/q) rows, whereas the remaining proceses will have
 * floor(n/q) rows. The same applies for the distribution of columns.
 *
 * Input:
 *    ________________________________________
 *   |   A[1:n,   |             |             |
 *   |     1:n]   |             |             |
 *   |------------|-------------|-------------|
 *   |            |             |             |
 *   |            |             |             |
 *   |------------|-------------|-------------|
 *   |            |             |             |
 *   |            |             |             |
 *   |------------|-------------|-------------|
 *
 *
 * Output:
 *    Note: the example simplifies the indexing, leaving out a necessary `-1`
 *          for the end of each index range. Don't duplcate elements across
 *          boundaries! Note that if `n` is not divisable by `q`, the block
 *          distribution will have ceil(n/q) or floor(n/q) rows/columns in
 *          each cell. The following illustrates this point:
 *
 *   |- ceil(n/q)-|    ....     |- floor(n/q)-|
 *    ________________________________________  _
 *   | A[1:n/3,   | A[1:n/3,    | A[1:n/3,    | |  ceil(n/q)
 *   |   1:n/3]   |   n/3:2n/3] |   2n/3:n]   | |
 *   |------------|-------------|-------------| -
 *   |A[n/3:2n/3, |    ....     |     ...     | ...
 *   |  1:n/3]    |             |             |
 *   |------------|-------------|-------------| -
 *   | A[2n/3:n,  |    ....     |  A[2n/3:n,  | |  floor(n/q)
 *   |   1:n/3]   |             |    2n/3:n]  | |
 *   |------------|-------------|-------------| -
 *
 * @param n             The size of the input dimensions.
 * @param input_matrix  The input matrix of size n-by-n, stored on processor
 *                      (0,0). This is an invalid pointer on all other processes.
 * @param local_matrix  The local output matrix of size n1-by-n2, where both n1
 *                      and n2 are given by the block distribution among rows
 *                      and columns. The output has to be allocated in this
 *                      process.
 * @param comm          A 2d cartesian grid communicator of size q-by-q (i.e.,
 *                      a perfect square).
 */
void distribute_matrix(const int n, double* input_matrix, double** local_matrix,
                       MPI_Comm comm);



/**
 * @brief   Given a vector distributed among the first column,
 *          this function transposes it to be distributed among rows.
 *
 * Given a vector that is distirbuted among the first column (i,0) of processors,
 * this function will "transpose" the vector, such that it is block decomposed
 * by row of processors.
 *
 * Example:
 *   Input:
 *    ________________________________________
 *   |            |             |             |
 *   | v[1:n/3]   |             |             |
 *   |------------|-------------|-------------|
 *   |            |             |             |
 *   | v[n/3:2n/3]|             |             |
 *   |------------|-------------|-------------|
 *   |            |             |             |
 *   | v[2n/3:n]  |             |             |
 *   |------------|-------------|-------------|
 *
 *   Output:
 *    ________________________________________
 *   |            |             |             |
 *   | v[1:n/3]   | v[n/3:2n/3] | v[2n/3:n]   |
 *   |------------|-------------|-------------|
 *   |            |             |             |
 *   | v[1:n/3]   | v[n/3:2n/3] | v[2n/3:n]   |
 *   |------------|-------------|-------------|
 *   |            |             |             |
 *   | v[1:n/3]   | v[n/3:2n/3] | v[2n/3:n]   |
 *   |------------|-------------|-------------|
 *
 *  To accomplish this, first, each proccessor (i,0) sends it's local vector
 *  to the diagonal processor (i,i). Then the diagonal processor (i,i)
 *  broadcasts the message among it's column using a column sub-communicator.
 *
 * @param n             The total size of the input vector.
 * @param col_vector    The local vector as distributed among the first column
 *                      of processors. Has size ceil(n/q) or floor(n/q) on
 *                      processors (i,0). This is an invalid pointer (DO NOT
 *                      ACCESS) on processors (i,j) with j != 0.
 * @param row_vector    (Output) The local vector, block distributed among the
 *                      rows (see above example).
 * @param comm          A 2d cartesian grid communicator of size q-by-q (i.e.,
 *                      a perfect square).
 */
void transpose_bcast_vector(const int n, double* col_vector, double* row_vector,
                            MPI_Comm comm);


/**
 * @brief   Calculates y = A*x for a distributed n-by-n matrix A and
 *          distributed, size `n` vectors x and y on a q-by-q processor grid.
 *
 * The matrix A is distirbuted on the q-by-q grid as explained for the
 * `distirbute_matrix()` function.
 *
 * The vectors are distirbuted on the q-by-q grid as explained for the
 * `distribute_vector()` function. I.e., the vectors x and y are block
 * distributed accross the first column of the processor grid. The pointers are
 * invalid for other processors.
 *
 * The matrix multiplication is solved by first transposing the input vector
 * `x` onto all processors as described for the `transpose_bcast_vector()`
 * function, then locally multiplying the row decomposed vector by the local
 * matrix. Then, the resulting local vectors are summed by using MPI_Reduce
 * along rows by using row sub-communicators onto processors of the first
 * column, which yields the result `y`.
 *
 * @param n             The size of the input dimensions.
 * @param local_A       The distributed matrix A.
 * @param local_x       The distirbuted input vector x, distirbuted among the
 *                      first column of the processor grid (i,0).
 * @param local_y       The distributed output vector y, distributed among the
 *                      first column of the processor grid (i,0).
 * @param comm          A 2d cartesian grid communicator of size q-by-q (i.e.,
 *                      a perfect square).
 */
void distributed_matrix_vector_mult(const int n, double* local_A,
                                    double* local_x, double* local_y,
                                    MPI_Comm comm);


/**
 * @brief   Solves A*x = b for `x` using Jacobi's method, where A, b, and x are
 *          distirbuted among a q-by-q processor grid.
 *
 * The steps for the Jacobi methods are roughly as follows (pseudo-code):
 *
 * @code
 *      D = diag(A)     // block distribute to first column (i,0)
 *      R = A - D       // copy A and set diagonal to zero
 *      x = [0,...,0]   // init x to zero, block distributed on first column
 *
 *      for (iter in 1:max_iter):
 *          w = R*x         // using distributed_matrix_vector_mult()
 *          x = (b - P)/D   // purely local on first column, no communication necessary!
 *          w = A*x         // using distributed_matrix_vector_mult()
 *          l2 = ||b - w||  // calculate L2-norm in a distributed fashion
 *          if l2 <= l2_termination:
 *              return      // exit if termination criteria is met, make sure
 *                          // all processor know that they should exit
 *                          // (-> MPI_Allreduce)
 * @endcode
 *
 * @param n             The size of the input dimensions.
 * @param local_A       The distributed matrix A.
 * @param local_b       The distirbuted input vector b, distirbuted among the
 *                      first column of the processor grid (i,0).
 * @param local_x       The distributed output vector x, distributed among the
 *                      first column of the processor grid (i,0).
 * @param comm          A 2d cartesian grid communicator of size q-by-q (i.e.,
 *                      a perfect square).
 * @param max_iter          The maximum number of iterations to run.
 * @param l2_termination    The termination criteria for the L2-norm of
 *                          ||Ax - b||. Terminates as soon as the total L2-norm
 *                          is smaller or equal to this.
 */
void distributed_jacobi(const int n, double* local_A, double* local_b,
                        double* local_x, MPI_Comm comm, int max_iter = 100,
                        double l2_termination = 1e-10);



/*********************************************************************
 *                Pre-implemented Wrapper Functions:                 *
 *********************************************************************/

/**
 * @brief   Performs the matrix vector product: y = A*x in parallel using MPI.
 *
 * This wraps the distributed_matrix_vector_mult() function, in that it
 * first distributes the data, then performs the distributed matrix vector
 * multiplication and then gathers the data back to the processor with rank 0.
 *
 * @param n     The size of the dimensions.
 * @param A     A n-by-n matrix represented in row-major order.
 * @param x     The input vector of length n.
 * @param y     The output vector of length n.
 * @param comm          A 2d cartesian grid communicator of size q-by-q (i.e.,
 *                      a perfect square).
 */
void mpi_matrix_vector_mult(const int n, double* A,
                            double* x, double* y, MPI_Comm comm);

/**
 * @brief   Performs Jacobi's method for solving A*x=b for x in parallel using
 *          MPI.
 *
 * This wraps the distributed_jacobi() function, in that it first distributes
 * all the data, then executes the distributed jacobi function, and finally
 * gathers the results back to the processor with rank 0.
 *
 * @param n                 The size of the input.
 * @param A                 The input matrix `A` of size n-by-n.
 * @param b                 The input vector `b` of size n.
 * @param x                 The output vector `x` of size n.
 * @param comm              A 2d cartesian grid communicator of size q-by-q
 *                          (i.e., a perfect square).
 * @param max_iter          The maximum number of iterations to run.
 * @param l2_termination    The termination criteria for the L2-norm of
 *                          ||Ax - b||. Terminates as soon as the total L2-norm
 *                          is smaller or equal to this.
 */
void mpi_jacobi(const int n, double* A, double* b, double* x,
                MPI_Comm comm,
                int max_iter = 100, double l2_termination = 1e-10);
#endif // MPI_JACOBI_H
