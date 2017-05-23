/**
 * @file    io.h
 * @brief   Implements common IO and input generation functions.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */

/*********************************************************************
 *                  !!  DO NOT CHANGE THIS FILE  !!                  *
 *********************************************************************/

#ifndef IO_H
#define IO_H

#include <fstream>
#include <iterator>
#include <vector>
#include <string>
#include <iostream>
#include <stdexcept>
#include <limits>
#include <cmath>

/**
 * @brief   Reads a file into a vector in in binary mode.
 *
 * @tparam T        The value type of the vector.
 * @param filename  The filename of the input file.
 *
 * @return  A vector containing the data read from file.
 */
template<typename T>
std::vector<T> read_binary_file(const char* filename)
{
    // get file size
    std::ifstream in(filename, std::ios::binary | std::ios::ate);
    if (!(in.good() && in.is_open()))
        throw std::runtime_error(std::string("Couldn't open file ") + filename);
    std::size_t nbytes = in.tellg();
    in.close();
    // open again, this time at the beginning
    std::ifstream infile(filename, std::ios::binary);

    // create vector of the correct size
    std::size_t n = nbytes / sizeof(T);
    std::vector<T> result(n);
    // read file into vector:
    infile.read((char*) &result[0], nbytes);
    return result;
}

/**
 * @brief   Writes binary data from a vector to a file.
 *
 * @param filename  The filename of the output file.
 * @param data      Vector of templated type `T` to be written to the file as
 *                  bytes.
 */
template<typename T>
void write_binary_file(const char* filename, const std::vector<T>& data)
{
    // open file in binary mode
    std::ofstream out(filename, std::ios::binary);
    if (!(out.good() && out.is_open()))
        throw std::runtime_error(std::string("Couldn't open file ") + filename);
    // write data to file
    out.write(reinterpret_cast<const char*>(&data[0]), sizeof(T)*data.size());
    out.close();
}

/// Gaussian normal distributed random noise
/// @source: http://en.wikipedia.org/wiki/Box-Muller_transform
double rnorm(double mu, double sigma)
{
    const double epsilon = std::numeric_limits<double>::min();
    const double two_pi = 2.0*3.14159265358979323846;

    static double z0, z1;
    static bool generate;
    generate = !generate;

    if (!generate)
        return z1 * sigma + mu;

    double u1, u2;
    do
    {
        u1 = rand() * (1.0 / RAND_MAX);
        u2 = rand() * (1.0 / RAND_MAX);
    }
    while ( u1 <= epsilon );

    z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
    z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
    return z0 * sigma + mu;
}


// generates a random matrix/vector of total size n using the gaussian normal
// distribution
std::vector<double> randn(int n, double mu = 0.0, double sigma = 1.0)
{
    std::vector<double> x(n);
    for (int i = 0; i < n; ++i)
    {
        x[i] = rnorm(mu, sigma);
    }
    return x;
}

// generates a random diagonally dominant matrix
std::vector<double> diag_dom_rand(int n, double difficulty = .5)
{
    double sigma = 100.0;
    double mu = difficulty * sigma;
    std::vector<double> A = randn(n*n, mu, sigma);
    std::vector<double> abssums(n);
    // get max row absolute sum
    double abssum_max = 0.0;
    for (int i = 0; i < n; ++i)
    {
        double abssum = 0.0;
        for (int j = 0; j < n; ++j)
        {
            abssum += std::fabs(A[i*n+j]);
        }
        if (abssum > abssum_max)
            abssum_max = abssum;
    }
    // add abssum max to abs of diagonal
    for (int i = 0; i < n; ++i)
    {
        A[i*n+i] = std::fabs(A[i*n+i]) + abssum_max;
    }
    return A;
}

#endif // IO_H
