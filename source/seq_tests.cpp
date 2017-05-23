/**
 * @file    seq_tests.cpp
 * @ingroup group
 * @brief   GTest Unit Tests for the sequential code.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */
/*
 * Add your own test cases here. We will test your final submission using
 * a more extensive tests suite. Make sure your code works for many different
 * input cases.
 */

#include <gtest/gtest.h>

#include "jacobi.h"

/*
 * Simple example of using the GTest Unit Testing Framework:
 * More information:
 * https://code.google.com/p/googletest/wiki/V1_7_Documentation
 */
/*
TEST(SimpleTest, SimpleExample) {
    // test for equal numbers
    EXPECT_EQ(1, 1);
    // test for close doubles
    EXPECT_NEAR(1., 1.000005, 1e-5);
    // test for not equal
    EXPECT_NE(0, 1);
    // add extra output for failing tests:
    EXPECT_EQ("hello", "world") << "strings are not equal!";
}
*/

// test sequential matrix vector multiplication
TEST(SequentialTest, MatrixVectorMult1) {
    // simple 4 by 4 input matrix
    double A[4*4] = {10., -1., 2., 0.,
                           -1., 11., -1., 3.,
                           2., -1., 10., -1.,
                           0.0, 3., -1., 8.};
    double x[4] =  {6., 25., -11., 15.};
    double y[4];
    double expected_y[4] = {13.,  325., -138.,  206.};
    int n = 4;

    // testing sequential matrix multiplication
    matrix_vector_mult(n, &A[0], &x[0], &y[0]);

    // checking if all values are correct (up to some error value)
    for (int i = 0; i < n; ++i)
    {
        EXPECT_NEAR(expected_y[i], y[i], 1e-10) << " element y[" << i << "] is wrong";
    }
}

// test sequential Jacobi method
TEST(SequentialTest, Jacobi1) {
    // simple 4 by 4 input matrix
    double A[4*4] = {10., -1., 2., 0.,
                           -1., 11., -1., 3.,
                           2., -1., 10., -1.,
                           0.0, 3., -1., 8.};
    double b[4] =  {6., 25., -11., 15.};
    double x[4];
    double expected_x[4] = {1.0,  2.0, -1.0, 1.0};
    int n = 4;

    // testing sequential matrix multiplication
    jacobi(n, A, b, x);

    // checking if all values are correct (up to some error value)
    for (int i = 0; i < n; ++i)
    {
        EXPECT_NEAR(expected_x[i], x[i], 1e-10) << " element y[" << i << "] is wrong";
    }
}
