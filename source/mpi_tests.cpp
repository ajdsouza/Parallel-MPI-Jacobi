/**
 * @file    mpi_tests.cpp
 * @ingroup group
 * @brief   GTest Unit Tests for the parallel MPI code.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */
/*
 * Add your own test cases here. We will test your final submission using
 * a more extensive tests suite. Make sure your code works for many different
 * input cases.
 *
 * Note:
 * The google test framework is configured, such that
 * only errors from the processor with rank = 0 are shown.
 */

#include <mpi.h>
#include <gtest/gtest.h>

#include <math.h>
#include "jacobi.h"
#include "mpi_jacobi.h"
#include "utils.h"
#include "io.h"

/**
 * @brief Creates and returns the square 2d grid communicator for MPI_COMM_WORLD
 */
void get_grid_comm(MPI_Comm* grid_comm)
{
    // get comm size and rank
    int rank, p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int q = (int)sqrt(p);
    ASSERT_EQ(q*q, p) << "Number of processors must be a perfect square.";

    // split into grid communicator
    int dims[2] = {q, q};
    int periods[2] = {0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, grid_comm);
}


TEST(MpiTest, MatrixVectorMult1)
{
    // simple 4 by 4 input matrix
    double A[4*4] = {10., -1., 2., 0.,
                           -1., 11., -1., 3.,
                           2., -1., 10., -1.,
                           0.0, 3., -1., 8.};
    double x[4] =  {6., 25., -11., 15.};
    double y[4];
    double expected_y[4] = {13.,  325., -138.,  206.};
    int n = 4;

    // get grid communicator
    MPI_Comm grid_comm;
    get_grid_comm(&grid_comm);

    // testing sequential matrix multiplication
    mpi_matrix_vector_mult(n, A, x, y, grid_comm);

    // checking if all values are correct (up to some error value)
    for (int i = 0; i < n; ++i)
    {
        EXPECT_NEAR(expected_y[i], y[i], 1e-10) << " element y[" << i << "] is wrong";
    }
}



// test parallel MPI matrix vector multiplication
TEST(MpiTest, MatrixVectorMult2)
{
    // simple 4 by 4 input matrix
    double A[3*3] = {10., -1.,0, 
                     2., 0.,0,
                     6,0,8};
    double x[3] =  {1., 2., 3.};
    double y[3];
    double expected_y[3] = {8., 2, 30.};
    int n = 3;

    // get grid communicator
    MPI_Comm grid_comm;
    get_grid_comm(&grid_comm);

    // testing sequential matrix multiplication
    mpi_matrix_vector_mult(n, A, x, y, grid_comm);

    // checking if all values are correct (up to some error value)
    for (int i = 0; i < n; ++i)
    {
      //std::cout<<"expected_y["<<i<<"]="<<expected_y[i]<<",y["<<i<<"]="<<y[i]<<std::endl;
        EXPECT_NEAR(expected_y[i], y[i], 1e-10) << " element y[" << i << "] is wrong";
    }
}


// test parallel MPI matrix vector multiplication
TEST(MpiTest, Jacobi1)
{
    // simple 4 by 4 input matrix
    double A[4*4] = {10., -1., 2., 0.,
                           -1., 11., -1., 3.,
                           2., -1., 10., -1.,
                           0.0, 3., -1., 8.};
    double b[4] =  {6., 25., -11., 15.};
    double x[4];
    double expected_x[4] = {1.0,  2.0, -1.0, 1.0};
    int n = 4;

    // get grid communicator
    MPI_Comm grid_comm;
    get_grid_comm(&grid_comm);

    // testing sequential matrix multiplication
    mpi_jacobi(n, A, b, x, grid_comm);

    // checking if all values are correct (up to some error value)
    for (int i = 0; i < n; ++i)
    {
        EXPECT_NEAR(expected_x[i], x[i], 1e-5) << " element y[" << i << "] is wrong";
    }
}



// test parallel MPI matrix vector multiplication
TEST(MpiTest, Jacobi2)
{
    // simple 3 by 3 input matrix
    double A[3*3] = { 483.,   -6., -186.,
  28. , 729. , -88.,
  38. , -104.,  413.};

    double b[3] =  {-30. , 65., -31. };
    double x[3];
    double expected_x[3] = {-0.0787494 ,0.0866353 ,-0.0459987 };
    int n = 3;

    // get grid communicator
    MPI_Comm grid_comm;
    get_grid_comm(&grid_comm);

    // testing sequential matrix multiplication
    mpi_jacobi(n, A, b, x, grid_comm);

    // checking if all values are correct (up to some error value)
    for (int i = 0; i < n; ++i)
    {
        EXPECT_NEAR(expected_x[i], x[i], 1e-5) << " element y[" << i << "] is wrong";
    }
}


// test parallel MPI matrix vector multiplication
TEST(MpiTest, Jacobi3)
{
    // simple 10 by 10 input matrix
    double A[10*10] = 
  { 1229.,    38.,    58.,  -132.,  -177.,   -48.,  -145.,   -47.,   -92.,   -59.,
    74.,  1478.,   114.,   -35.,    49.,   -73.,   153.,  -120.,   115.,   269.,
    88.,    87.,  1460.,  -159.,    68.,    84.,    31.,    84.,   254.,   100.,
  -121.,   115.,   -77.,  1088.,   142.,    34.,    49.,   -85.,     8.,   173.,
  -137.,    57.,    32.,   -77.,  1247.,  -172.,   -54.,    38.,   -11.,   -44.,
   169.,   107.,   -74.,    31.,    -9.,  1449.,  -211.,    95.,   140.,  -228.,
   -31.,   132.,   -14.,   270.,    97.,   227.,  1389.,    77.,    17.,   -57.,
  -144.,    43.,    22.,  -239.,   -34.,    41.,    76.,  1516.,   -21.,   124.,
  -101.,   -16.,    89.,    27.,    64.,   238.,  -101.,   158.,  1263.,    53.,
    -81.,   -50.,   -20.,   119.,  -169.,   -34.,   150.,   105.,    48.,  1539. };

    double b[10] =  {  18.,   -2.,  -42.,  103.,   94., -209.,   98.,  -74.,  -58.,   19.};

    double x[10];
    double expected_x[10] = { 0.0375472 ,-0.0194894 ,-0.0139968 ,0.0880206 ,0.0719718 ,-0.134131 ,0.0749806 ,-0.029865 ,-0.0129856 ,0.00677537 };

    int n = 10;

    // get grid communicator
    MPI_Comm grid_comm;
    get_grid_comm(&grid_comm);

    // testing sequential matrix multiplication
    mpi_jacobi(n, A, b, x, grid_comm);

    // checking if all values are correct (up to some error value)
    for (int i = 0; i < n; ++i)
    {
        EXPECT_NEAR(expected_x[i], x[i], 1e-5) << " element y[" << i << "] is wrong";
    }
}





//
//  Test the parallel code and compare the results with the sequential code.
//
TEST(MpiTest, JacobiCrossTest1)
{
    // test random matrixes, test parallel code with sequential solutions
    std::vector<double> A;
    std::vector<double> b;
    std::vector<double> mpi_x;

    // get grid communicator
    MPI_Comm grid_comm;
    get_grid_comm(&grid_comm);
    int rank;
    MPI_Comm_rank(grid_comm, &rank);

    int n = 36;
    // initialize data only on rank 0
    if (rank == 0)
    {
        A = diag_dom_rand(n);
        b = randn(n, 100.0, 50.0);
    }

    // getting sequential results
    std::vector<double> x;
    if (rank == 0)
    {
        x.resize(n);
        jacobi(n, &A[0], &b[0], &x[0]);
    }

    // parallel jacobi
    if (rank == 0)
        mpi_x.resize(n);
    mpi_jacobi(n, &A[0], &b[0], &mpi_x[0], grid_comm);

    if (rank == 0)
    {
        // checking if all values are correct (up to some error value)
        for (int i = 0; i < n; ++i)
        {
            EXPECT_NEAR(x[i], mpi_x[i], 1e-8) << " MPI solution x[" << i << "] differs from sequential result";
        }
    }
}



TEST(MpiTest, JacobiCrossTest2)
{

   // simple 3 by 3 input matrix
    double A[3*3] = { 483.,   -6., -186.,
  28. , 729. , -88.,
  38. , -104.,  413.};

    double b[3] =  {-30. , 65., -31. };
    double x[3];
    double mpi_x[3];
    double expected_x[3] = {-0.0787494 ,0.0866353 ,-0.0459987 };
    int n = 3;

    // get grid communicator
    MPI_Comm grid_comm;
    get_grid_comm(&grid_comm);
    int rank;
    MPI_Comm_rank(grid_comm, &rank);

    if (rank == 0)
    {
        jacobi(n, &A[0], &b[0], &x[0]);
    }

    // parallel jacobi
    mpi_jacobi(n, &A[0], &b[0], &mpi_x[0], grid_comm);

    if (rank == 0)
    {
        // checking if all values are correct (up to some error value)
        for (int i = 0; i < n; ++i)
        {
            EXPECT_NEAR(x[i], mpi_x[i], 1e-8) << " MPI solution x[" << i << "] differs from sequential result";
            EXPECT_NEAR(expected_x[i], x[i], 1e-5) << " Serial solution x[" << i << "] is wrong";
        }

        //for (int i = 0; i < n; ++i)
        //{
	//  std::cout<<"mpi_tests:JacobiCrossTest2:serial x["<<i<<"]="<<x[i]<<",--expected_x["<<i<<"]="<<expected_x[i]<<",--mpi_x["<<i<<"]="<<mpi_x[i]<<std::endl;
        //}
    }
}


TEST(MpiTest, JacobiCrossTest3)
{
    // test random matrixes, test parallel code with sequential solutions
    std::vector<double> A;
    std::vector<double> b;
    std::vector<double> mpi_x;

    // get grid communicator
    MPI_Comm grid_comm;
    get_grid_comm(&grid_comm);
    int rank;
    MPI_Comm_rank(grid_comm, &rank);

    int n = 100;
    // initialize data only on rank 0
    if (rank == 0)
    {
        A = diag_dom_rand(n);
        b = randn(n, 100.0, 50.0);
    }

    // getting sequential results
    std::vector<double> x;
    if (rank == 0)
    {
        x.resize(n);
        jacobi(n, &A[0], &b[0], &x[0]);
    }

    // parallel jacobi
    if (rank == 0)
        mpi_x.resize(n);
    mpi_jacobi(n, &A[0], &b[0], &mpi_x[0], grid_comm);

    if (rank == 0)
    {
        // checking if all values are correct (up to some error value)
        for (int i = 0; i < n; ++i)
        {
            EXPECT_NEAR(x[i], mpi_x[i], 1e-8) << " MPI solution x[" << i << "] differs from sequential result";
        }
    }
}
