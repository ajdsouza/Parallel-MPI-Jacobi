/**
 * @file    jacobi.h
 * @brief   Implements matrix vector multiplication and Jacobi's method.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 *
 */

/*********************************************************************
 *                  !!  DO NOT CHANGE THIS FILE  !!                  *
 *********************************************************************/

#ifndef JACOBI_H
#define JACOBI_H

/**
 * @brief   Performs the matrix vector product: y = A*x
 *
 * @param n     The size of the dimensions.
 * @param A     A n-by-n matrix represented in row-major order.
 * @param x     The input vector of length n.
 * @param y     The output vector of length n.
 */
void matrix_vector_mult(const int n, const double* A,
                        const double* x, double* y);


/**
 * @brief   Performs the matrix vector product: y = A*x
 *
 * @param n     The size of the first dimension.
 * @param m     The size of the second dimension.
 * @param A     A n-by-m matrix represented in row-major order.
 * @param x     The input vector of length m.
 * @param y     The output vector of length n.
 */
void matrix_vector_mult(const int n, const int m, const double* A,
                        const double* x, double* y);


/**
 * @brief   Performs Jacobi's method for solving A*x=b for x.
 *
 * @param n                 The size of the input.
 * @param A                 The input matrix `A` of size n-by-n.
 * @param b                 The input vector `b` of size n.
 * @param x                 The output vector `x` of size n.
 * @param max_iter          The maximum number of iterations to run.
 * @param l2_termination    The termination criteria for the L2-norm of
 *                          ||Ax - b||. Terminates as soon as the total L2-norm
 *                          is smaller or equal to this.
 */
void jacobi(const int n, double* A, double* b, double* x,
            int max_iter = 100, double l2_termination = 1e-10);

#endif // JACOBI_H
