/**
 * @file    jacobi.cpp
 * @brief   Implements matrix vector multiplication and Jacobi's method.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */
#include "jacobi.h"

/*
 * TODO: Implement your solutions here
 */

// my implementation:
#include <iostream>
#include <math.h>
#include <vector>
#include <stdlib.h>

// Calculates y = A*x for a square n-by-n matrix A, and n-dimensional vectors x
// and y
void matrix_vector_mult(const int n, const double* A, const double* x, double* y)
{
  // TODO
  matrix_vector_mult(n,n,A,x,y);
}

// calculates y = a*x for a n-by-m matrix a, a m-dimensional vector x
// and a n-dimensional vector y
void matrix_vector_mult(const int n, const int m, const double* a, const double* x, double* y)
{
  //TODO
  for (int i = 0;i<n;i++){
    y[i]=0;
    for (int j=0;j<m;j++){
      y[i] = y[i] + a[n*i+j]*x[j];  
    }
  }

}



// implements the sequential jacobi method
void jacobi(const int n, double* A, double* b, double* x, int max_iter, double l2_termination)
{
  // TODO
  //std::cout<<"DEBUG:jacobi.cpp::jacobi::Begin"<<std::endl;

  // remove the diagonal elements into r
  //double D[n];
  //double R[n*n];
  double  *D = (double *)malloc(n*sizeof(double));
  double  *y = (double *)malloc(n*sizeof(double));

  for (int i = 0;i<n;i++){
    for (int j = 0;j<n;j++){
      //std::cout<<"A("<<i<<","<<j<<")="<<A[i*n+j];
      if ( i==j ) {
	D[i]=A[i*n+j];
	//R[i*n+j]=0.0;
      }
      //else
//	R[i*n+j]=A[i*n+j];
    }
  }

  double l2norm = l2_termination+1;
  int iter = 0;

      // y = Rx
      //double y[n];
      matrix_vector_mult(n,A,x,y);

  while ( (iter < max_iter) && (l2norm > l2_termination) )
    {

      // 1/D (b-Rx)
      for (int i = 0;i<n;i++){
	x[i]=(b[i]-(y[i]-(D[i]*x[i])))/D[i];
      }
 
      // Ax
      matrix_vector_mult(n,A,x,y);

      // ||Ax-B||
      l2norm=0;
      for (int i = 0;i<n;i++){
	l2norm = l2norm + pow(b[i]-y[i],2);
      }

      l2norm = sqrt(l2norm);
      iter++;

      //  std::cout<<"DEBUG:jacobi.cpp::jacobi::in Iteration "<<iter<<std::endl;
    }

   free(D);
   free(y);

  // debug results
  // std::cout<<"DEBUG:jacobi::jacobi:Results x = [";
  //for (int i = 0;i<n;i++){
  //  std::cout<<x[i]<<" ,";
  //}
  //std::cout<<" ]"<<std::endl;


}
