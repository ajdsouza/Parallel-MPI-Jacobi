/**
 * @file    main.cpp
 * @brief   Implements the main function for the 'jacobi' executable.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */

/*********************************************************************
 *                  !!  DO NOT CHANGE THIS FILE  !!                  *
 *********************************************************************/

#include <mpi.h>

#include <stdexcept>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <string>
#include <sstream>

#include "jacobi.h"
#include "mpi_jacobi.h"
#include "utils.h"
#include "io.h"

void print_usage()
{
    std::cerr << "Usage: ./jacobi <input_A> <input_b> <output_x>" << std::endl;
    std::cerr << "                  Reads the input A and b from the given binary files and" << std::endl;
    std::cerr << "                  writes the result to the given file in binary format." << std::endl;
    std::cerr << "       ./jacobi -n <n> [-d <difficulty>]" << std::endl;
    std::cerr << "                  Creates random input of size <n> (A has size n-by-n) of" << std::endl;
    std::cerr << "                  the given difficulty, a value between 0.0 (easiest) and 1.0" << std::endl;
    std::cerr << "                  (optional, default = 0.5)." << std::endl;
}

int main(int argc, char *argv[])
{
   // set up MPI
   MPI_Init(&argc, &argv);

   // get communicator size
   MPI_Comm comm = MPI_COMM_WORLD;
   int p;
   MPI_Comm_size(comm, &p);

   // Ax = b
   std::vector<double> A;
   std::vector<double> b;
   std::vector<double> x;


   /**********************************
    *  Parse command line arguments  *
    **********************************/
   bool write_output = false;
   std::string outfile_name;
   int n;
   bool rand_input = false;
   double difficulty = 0.5;
   std::string fileA;
   std::string fileB;
   if (argc < 3)
   {
      print_usage();
      exit(EXIT_FAILURE);
   }
   if (std::string(argv[1]) == "-n")
   {
      // randomly generate input
      n = atoi(argv[2]);
      if (!(n > 0))
      {
         print_usage();
         exit(EXIT_FAILURE);
      }
      if (argc == 5)
      {
         if (std::string(argv[3]) != "-d")
         {
            print_usage();
            exit(EXIT_FAILURE);
         }
         std::istringstream iss(argv[4]);
         iss >> difficulty;
      }
      rand_input = true;
   }
   else
   {
      if (argc != 4)
      {
         print_usage();
         exit(EXIT_FAILURE);
      }

      // get output filename
      outfile_name = std::string(argv[3]);
      write_output = true;
   }


   // start timer
   //   we omit the file loading and argument parsing from the runtime
   //   timings, we measure the time needed by the processor with rank 0
   struct timespec t_start, t_end;
   if (p > 1)
   {
      // get the dimensions
      int q = (int)sqrt(p);
      if (p != q*q)
      {
         throw std::runtime_error("The number of MPI processes must be a perfect square");
      }
      // create 2D cartesian grid for the processors (enable reordering)
      MPI_Comm grid_comm;
      int dims[2] = {q, q};
      int periods[2] = {0, 0};
      MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid_comm);
      // get the rank of process with coordinates (0,0)
      int rank00;
      int myrank;
      int coords[2] = {0, 0};
      MPI_Cart_rank(grid_comm, coords, &rank00);
      MPI_Comm_rank(grid_comm, &myrank);

      // process (0,0) loads input
      if (myrank == rank00)
      {
         if (rand_input) {
            // generate random input of the given size
            A = diag_dom_rand(n, difficulty);
            b = randn(n);
         } else {
            // read input from file
            A = read_binary_file<double>(argv[1]);
            b = read_binary_file<double>(argv[2]);
            n = b.size();
            if (A.size() != n*(size_t)n)
            {
               throw std::runtime_error("The input dimensions are not matching");
            }
         }
      }
      MPI_Bcast(&n, 1, MPI_INT, rank00, grid_comm);

      // start timer
      clock_gettime(CLOCK_MONOTONIC,  &t_start);

      // allocate output and run the parallel jacobi implementation
      if (myrank == rank00)
         x = std::vector<double>(n);
      mpi_jacobi(n, &A[0], &b[0], &x[0], grid_comm);

      if (myrank == rank00)
      {
         // get time
         clock_gettime(CLOCK_MONOTONIC,  &t_end);
         // time in seconds
         double time_secs = (t_end.tv_sec - t_start.tv_sec)
            + (double) (t_end.tv_nsec - t_start.tv_nsec) * 1e-9;
         // output time
         std::cerr << time_secs << std::endl;
         // write output
         if (write_output)
         {
            write_binary_file(outfile_name.c_str(), x);
         }
      }
   }
   else
   {
      std::cerr << "[WARNING]: Running the sequential solver. Start with mpirun to execute the parallel version." << std::endl;

      // get input
      if (rand_input) {
         // generate random input of the given size
         A = diag_dom_rand(n, difficulty);
         b = randn(n);
      } else {
         // read input from file
         A = read_binary_file<double>(argv[1]);
         b = read_binary_file<double>(argv[2]);
         n = b.size();
         if (A.size() != n*(size_t)n)
         {
            throw std::runtime_error("The input dimensions are not matching");
         }
      }

      clock_gettime(CLOCK_MONOTONIC,  &t_start);
      // sequential jacobi
      x = std::vector<double>(n);
      jacobi(n, &A[0], &b[0], &x[0]);
      // get time
      clock_gettime(CLOCK_MONOTONIC,  &t_end);
      // time in seconds
      double time_secs = (t_end.tv_sec - t_start.tv_sec)
         + (double) (t_end.tv_nsec - t_start.tv_nsec) * 1e-9;
      // output time
      std::cerr << time_secs << std::endl;
      // write output
      if (write_output)
      {
         write_binary_file(outfile_name.c_str(), x);
      }
   }

   // finalize MPI
   MPI_Finalize();
   return 0;
}
