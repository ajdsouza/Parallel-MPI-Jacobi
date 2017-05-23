/**
 * @file    mpi_jacobi.cpp
 * @brief   Implements MPI functions for distributing vectors and matrixes,
 *          parallel distributed matrix-vector multiplication and Jacobi's
 *          method.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */

#include "mpi_jacobi.h"
#include "jacobi.h"
#include "utils.h"

#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <vector>
#include <iostream>

/*
 * TODO: Implement your solutions here
 */


void distribute_vector(const int n, double* input_vector, double** local_vector, MPI_Comm comm)
{
  // TODO
  //std::cout<<"DEBUG:mpi_jacobi::distribute_vector:begin with n="<<n<<std::endl;

  // rank in grid
  int myrank;
  int mycoords[2];
  MPI_Comm_rank(comm,&myrank);
  // i,j coord in grid
  MPI_Cart_coords(comm,myrank,2,mycoords);

  // rank of root 0,0 matrix in grid
  int rank00;
  int coords00[2]={0,0};
  MPI_Cart_rank(comm,coords00,&rank00);

  // recieved input to distribute
  //if ( myrank == rank00 ) {
  //for (int i=0;i<n;i++){
      //std::cout<<"DEBUG:mpi_jacobi::distribute_vector:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),recieved input_vector["<<i<<"]="<<input_vector[i]<<std::endl;
  //}
  //}

  // create a comm group for each column
  MPI_Comm commcol;
  int cdims[2] = {1,0};
  MPI_Cart_sub(comm, cdims, &commcol);

  //std::cout<<"DEBUG:mpi_jacobi::distribute_vector:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),created cart sub for rank="<<myrank<<std::endl;

  // distribute vector among first column processors only 
  if ( mycoords[1]==0 ){

    //std::cout<<"DEBUG:mpi_jacobi::distribute_vector:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),rank="<<myrank<<std::endl;

    // rank in the column
    int colsize;
    int mycolrank;
    MPI_Comm_size(commcol,&colsize);
    MPI_Comm_rank(commcol,&mycolrank);

    //std::cout<<"DEBUG:mpi_jacobi::distribute_vector:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),n="<<n<<",colrank="<<mycolrank<<",colsize="<<colsize<<std::endl;

    // floor and ceil values
    int fnp = (int)floor(((double)n)/colsize);
    int cnp = (int)ceil( ((double)n)/colsize);

    //std::cout<<"DEBUG:mpi_jacobi::distribute_vector:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<",cnp="<<cnp<<",fnp="<<fnp<<std::endl;

    // allocate memory for rcv buffer
    int rcvsize;
    if ( mycolrank < ( n % colsize ) ) 
      rcvsize = cnp;
    else
      rcvsize = fnp;

    //std::cout<<"DEBUG:mpi_jacobi::distribute_vector:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),Allocation memory for local vector="<<rcvsize<<std::endl;

    double *temp_vector;
    temp_vector=(double *)malloc(rcvsize*sizeof(double));
    *local_vector = &temp_vector[0];

    // rank of root 0,0 matrix in the col comm to be used in scatterv
    int rankcol00;
    MPI_Cart_rank(commcol,coords00,&rankcol00);

    //std::cout<<"DEBUG:mpi_jacobi::distribute_vector:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),the col rank of (0,,0) is "<<rankcol00<<std::endl;

    //std::cout<<"DEBUG:mpi_jacobi::distribute_vector:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),Allocation memory for count and displ array for scatterv="<<colsize<<std::endl;

    int *disp = (int *)malloc(colsize*sizeof(int));
    int *ncount = (int *)malloc(colsize*sizeof(int));

    // for cart 0,0 processor whcih is sender
    // prepare the count and disp arrays for scatterv
    if ( mycolrank == rankcol00 ){

      for (int i=0;i<colsize;i++) {

	if ( i < ( n % colsize ) ) 
	  ncount[i]=cnp;
	else
	  ncount[i]=fnp;

	if ( i > 0 )
	  disp[i]=disp[i-1]+ncount[i-1];
	else
	  disp[i]=0;

	//std::cout<<"DEBUG:mpi_jacobi::distribute_vector:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"), allocated for col="<<i<<", disp="<<disp[i]<<",count="<<ncount[i]<<std::endl;

      }
    }

    //std::cout<<"DEBUG:mpi_jacobi::distribute_vector:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),colrank="<<mycolrank<<",scatterv, with rank00="<<rankcol00<<std::endl;

 
    // scatterv
    MPI_Scatterv(input_vector,ncount,disp,
		 MPI_DOUBLE,temp_vector,rcvsize,
		 MPI_DOUBLE,
		 rankcol00, commcol);

    //std::cout<<"DEBUG:mpi_jacobi::distribute_vector:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),Completed scatterv, with rank00="<<rankcol00<<std::endl;

    // print the rcv buffer
    //double *ptr = *local_vector;

    //for (int i=0;i<rcvsize;i++){
      //std::cout<<"DEBUG:mpi_jacobi::distribute_vector:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),recieved from scatterv local_vector["<<i<<"]="<<ptr[i]<<std::endl;
    //}

    free(ncount);
    free(disp);

  }
  
  //std::cout<<"DEBUG:mpi_jacobi::distribute_vector:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),completed"<<std::endl;

  // test
  //double *rr=NULL;
  //double *row_vector=NULL;
  //gather_vector(n,*local_vector,rr,comm);
  //transpose_bcast_vector(n,*local_vector,row_vector,comm);

}





// gather the local vector distributed among (i,0) to the processor (0,0)
void gather_vector(const int n, double* local_vector, double* output_vector, MPI_Comm comm)
{
  // TODO

  //std::cout<<"DEBUG:mpi_jacobi::gather_vector:begin with n="<<n<<std::endl;

  // rank in grid
  int myrank;
  MPI_Comm_rank(comm,&myrank);

  // i,j coord in grid
  int mycoords[2];
  MPI_Cart_coords(comm,myrank,2,mycoords);

  // rank of root 0,0 matrix in grid
  int rank00;
  int coords00[2]={0,0};
  MPI_Cart_rank(comm,coords00,&rank00);

  // create a comm group for each column
  MPI_Comm commcol;
  int cdims[2] = {1,0};
  MPI_Cart_sub(comm, cdims, &commcol);

  //std::cout<<"DEBUG:mpi_jacobi::gather_vector:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),created cart sub for rank="<<myrank<<std::endl;

  // gather vector among first column processors only 
  if ( mycoords[1]==0 ){

    //std::cout<<"DEBUG:mpi_jacobi::gather_vector:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),rank="<<myrank<<std::endl;

    // rank in the column
    int colsize;
    int mycolrank;
    MPI_Comm_size(commcol,&colsize);
    MPI_Comm_rank(commcol,&mycolrank);

    //std::cout<<"DEBUG:mpi_jacobi::gather_vector:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),colrank="<<mycolrank<<",colsize="<<colsize<<std::endl;

    // floor and ceil values
    int fnp = (int)floor(((double)n)/colsize);
    int cnp = (int)ceil( ((double)n)/colsize);

    // send buffer size at each processor
    int sendsize;
    if ( mycolrank < ( n % colsize ) ) 
      sendsize = cnp;
    else
      sendsize = fnp;


    // rank of root 0,0 matrix in the col comm to be used in scatterv
    int rankcol00;
    MPI_Cart_rank(commcol,coords00,&rankcol00);

    //std::cout<<"DEBUG:mpi_jacobi::gather_vector:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),the col rank of (0,,0) is "<<rankcol00<<std::endl;

    //std::cout<<"DEBUG:mpi_jacobi::gather_vector:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),Allocation memory for count and displ array for scatterv="<<colsize<<std::endl;

    int *disp = (int *)malloc(colsize*sizeof(int));
    int *ncount = (int *)malloc(colsize*sizeof(int));

    // for cart 0,0 processor whcih is sender
    // prepare the count and disp arrays for scatterv
    if ( mycolrank == rankcol00 ){

      //std::cout<<"DEBUG:mpi_jacobi::gather_vector:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),Allocation memory for local vector n ="<<n<<std::endl;

      // memory is already allocated on rank 00 processor
      //output_vector=(double *)malloc(n*sizeof(double));

      for (int i=0;i<colsize;i++) {

	if ( i < ( n % colsize ) ) 
	  ncount[i]=cnp;
	else
	  ncount[i]=fnp;

	if ( i > 0 )
	  disp[i]=disp[i-1]+ncount[i-1];
	else
	  disp[i]=0;

	//std::cout<<"DEBUG:mpi_jacobi::gather_vector:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"), allocated for col="<<i<<", disp="<<disp[i]<<",count="<<ncount[i]<<std::endl;

      }
    }

    //std::cout<<"DEBUG:mpi_jacobi::gather_vector:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),colrank="<<mycolrank<<",gatherv, with rank00="<<rankcol00<<std::endl;

    // scatterv
    MPI_Gatherv(local_vector,
		sendsize,
		MPI_DOUBLE, 
		output_vector,
		ncount,disp,
		MPI_DOUBLE,
		rankcol00, 
		commcol);

    //std::cout<<"DEBUG:mpi_jacobi::gather_vector:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),Completed gatherv, with rank00="<<rankcol00<<std::endl;

    // print the rcv buffer
    //double *ptr = local_vector;

    //for (int i=0;i<sendsize;i++){
      //std::cout<<"DEBUG:mpi_jacobi::gather_vector:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),sent through gather local_vector["<<i<<"]="<<ptr[i]<<std::endl;
    //}

    //ptr = output_vector;

    //if ( mycolrank == rankcol00 ) {
    //  for (int i=0;i<n;i++){
	//std::cout<<"DEBUG:mpi_jacobi::gather_vector:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),recieved from gather output_vector["<<i<<"]="<<ptr[i]<<std::endl;
    //}
    //}

    free(ncount);
    free(disp);

  }

  //std::cout<<"DEBUG:mpi_jacobi::gather_vector:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),completed"<<std::endl;


}





void distribute_matrix(const int n, double* input_matrix, double** local_matrix, MPI_Comm comm)
{
  // TODO

  //std::cout<<"DEBUG:mpi_jacobi::distribute_matrix:begin with n="<<n<<std::endl;
 
  int *disp=NULL;
  int *ncount=NULL;
  double *row_block_matrix=NULL;
  int blockrows=0;

  // rank in grid
  int myrank;
  int mycoords[2];
  MPI_Comm_rank(comm,&myrank);

  // i,j coord in grid
  MPI_Cart_coords(comm,myrank,2,mycoords);
 
  // rank of root 0,0 matrix in grid
  int rank00;
  int coords00[2]={0,0};

  MPI_Cart_rank(comm,coords00,&rank00);

  // rcvd input - DEBUG
  //if ( myrank == rank00 ) {
  //for (int i=0;i<pow(n,2);i++){
      //std::cout<<"DEBUG:mpi_jacobi::distribute_matrix:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),recevied input input_matrix["<<i<<"]="<<input_matrix[i]<<std::endl;
  //}
  //}

  // 1.  Distribute the rows from processor 0.0 to all processors in the first column
  //
  // create a comm group for each column
  MPI_Comm commcol;
  int cdims[2] = {1,0};
  MPI_Cart_sub(comm, cdims, &commcol);

  //std::cout<<"DEBUG:mpi_jacobi::distribute_matrix:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),created column cart sub for rank="<<myrank<<std::endl;

  // rank in the column
  int colsize;
  int mycolrank;
  MPI_Comm_size(commcol,&colsize);
  MPI_Comm_rank(commcol,&mycolrank);

  // floor and ceil values
  int fnp = (int)floor(((double)n)/colsize);
  int cnp = (int)ceil( ((double)n)/colsize);

  // allocate memory for rcv buffer
  if ( mycolrank < ( n % colsize )) 
    blockrows = cnp;
  else
    blockrows = fnp;

  // first distribute among first column processors only 
  if ( mycoords[1]==0 ){

    //std::cout<<"DEBUG:mpi_jacobi::distribute_matrix:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),rank="<<myrank<<",colrank="<<mycolrank<<",colsize="<<colsize<<std::endl;

    int rcvsize = blockrows*n;

    //std::cout<<"DEBUG:mpi_jacobi::distribute_matrix:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),Allocation memory for row_block_matrix="<<rcvsize<<std::endl;

    row_block_matrix=(double *)malloc(rcvsize*sizeof(double));

    // rank of root 0,0 matrix in the col comm to be used in scatterv
    int rankcol00;
    int colcoords00 = 0;
    MPI_Cart_rank(commcol,&colcoords00,&rankcol00);

    //std::cout<<"DEBUG:mpi_jacobi::distribute_matrix:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),the col rank of (0,0) is "<<rankcol00<<std::endl;

    //std::cout<<"DEBUG:mpi_jacobi::distribute_matrix:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),Allocation memory for count and displ array for scatterv="<<colsize<<std::endl;

    //disp = (int *)malloc(colsize*sizeof(int));
    //ncount = (int *)malloc(colsize*sizeof(int));

    // for cart 0,0 processor whcih is sender
    // prepare the count and disp arrays for scatterv
    if ( mycolrank == rankcol00 ){

    disp = (int *)malloc(colsize*sizeof(int));
    ncount = (int *)malloc(colsize*sizeof(int));

      for (int i=0;i<colsize;i++) {

	if ( i < ( n % colsize ) ) 
	  ncount[i]=n*cnp;
	else
	  ncount[i]=n*fnp;

	if ( i > 0 )
	  disp[i]=disp[i-1]+ncount[i-1];
	else
	  disp[i]=0;

	//std::cout<<"DEBUG:mpi_jacobi::distribute_matrix:n="<<n<<"(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"), allocated for col="<<i<<", disp="<<disp[i]<<",count="<<ncount[i]<<std::endl;

      }

    }

    //std::cout<<"DEBUG:mpi_jacobi::distribute_matrix:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),colrank="<<mycolrank<<",scatterv, with rank00="<<rankcol00<<std::endl;

    // scatterv
    MPI_Scatterv(input_matrix,ncount,disp,
		 MPI_DOUBLE, row_block_matrix,rcvsize,
		 MPI_DOUBLE,
		 rankcol00, commcol);

    //std::cout<<"DEBUG:mpi_jacobi::distribute_matrix:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),Completed scatterv, with rank00="<<rankcol00<<std::endl;

    // print the rcv buffer
    //double *ptr = row_block_matrix;

    //if ( mycoords[0]==1) {
      //for (int i=0;i<rcvsize;i++){
	//std::cout<<"DEBUG:mpi_jacobi::distribute_matrix::n="<<n<<"(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),recieved from scatterv row_block_matrix["<<i<<"]="<<ptr[i]<<std::endl;
	//}
      //}

if ( mycolrank == rankcol00 ) {
    free(ncount);
    free(disp);
}

  }

  //std::cout<<"DEBUG:mpi_jacobi::distribute_matrix:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),Completed distributing the rows in processor 0,0 to processors in first column "<<std::endl;




  // 2. distribute the columns from first processor in each row to al row processors
  //
  MPI_Barrier(comm);

  // 1.  Distribute the rows in the first column
  //
  // create a comm group for each row
  MPI_Comm commrow;
  int rdims[2] = {0,1};
  MPI_Cart_sub(comm, rdims, &commrow);

  //std::cout<<"DEBUG:mpi_jacobi::distribute_matrix:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),created row cart sub for rank="<<myrank<<std::endl;

  // rank in the row
  int rowsize;
  int myrowrank;
  MPI_Comm_size(commrow,&rowsize);
  MPI_Comm_rank(commrow,&myrowrank);

  //std::cout<<"DEBUG:mpi_jacobi::distribute_matrix:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),rowrank="<<myrowrank<<",rowsize="<<rowsize<<std::endl;

  // floor and ceil values
  fnp = (int)floor(((double)n)/rowsize);
  cnp = (int)ceil( ((double)n)/rowsize);

  int rcvsize;
  // allocate memory for rcv buffer
  if ( myrowrank < ( n % rowsize ) ) 
    rcvsize = cnp*blockrows;
  else
    rcvsize = fnp*blockrows;

  //std::cout<<"DEBUG:mpi_jacobi::distribute_matrix:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),allocating memory for local_matrix="<<rcvsize<<std::endl;

  double *temp_matrix;
  temp_matrix=(double *)malloc(rcvsize*sizeof(double));
  *local_matrix = &temp_matrix[0];

  // rank of root 0,0 matrix in the row comm to be used in scatterv
  int rankrow00;
  int rowcoords00 = 0;
  MPI_Cart_rank(commrow,&rowcoords00,&rankrow00);

  //std::cout<<"DEBUG:mpi_jacobi::distribute_matrix:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),the row rank of ("<<rowcoords00<<",0) is "<<rankrow00<<std::endl;

 
  //disp = (int *)malloc(rowsize*sizeof(int));
  //ncount = (int *)malloc(rowsize*sizeof(int));

  MPI_Datatype  tmpcol,coltype;
  MPI_Type_vector(blockrows, 1, n, MPI_DOUBLE, &tmpcol);
  MPI_Type_create_resized(tmpcol, 0, 1*sizeof(double), &coltype);
  MPI_Type_commit(&coltype);
  MPI_Type_free(&tmpcol);

  // for cart 0,0 processor whcih is sender
  // prepare the count and disp arrays for scatterv
  if ( myrowrank == rankrow00 ){

    //std::cout<<"DEBUG:mpi_jacobi::distribute_matrix:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"), allocating for row disp/count arrays "<<std::endl;
  disp = (int *)malloc(rowsize*sizeof(int));
  ncount = (int *)malloc(rowsize*sizeof(int));

    for (int i=0;i<rowsize;i++) {

      if ( i < ( n % rowsize ) ) 
	ncount[i]=cnp;
      else
	ncount[i]=fnp;

      if ( i > 0 )
	disp[i]=disp[i-1]+ncount[i-1];
      else
	disp[i]=0;

      //std::cout<<"DEBUG:mpi_jacobi::distribute_matrix:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"), allocated for col="<<i<<", disp="<<disp[i]<<",count="<<ncount[i]<<std::endl;

    }
  }

  //std::cout<<"DEBUG:mpi_jacobi::distribute_matrix:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),rowrank="<<myrowrank<<",scatterv, for row with rank00="<<rankrow00<<std::endl;

  MPI_Barrier(commrow);

  // scatterv
  MPI_Scatterv(row_block_matrix,ncount,disp,
	       coltype, temp_matrix,rcvsize,
	       MPI_DOUBLE,
	       rankrow00,commrow);


  //std::cout<<"DEBUG:mpi_jacobi::distribute_matrix:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),Completed scatterv, with rank00="<<rankrow00<<std::endl;

  // print the rcv buffer
  //double *ptr = *local_matrix;

  //if ( ( mycoords[0] == 1 ) && (mycoords[1] == 1) ){

  //for (int i=0;i<rcvsize;i++){
      //std::cout<<"DEBUG:mpi_jacobi::distribute_matrix::n="<<n<<"(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),recieved from scatterv local_matrix["<<i<<"]="<<ptr[i]<<std::endl;
  //}
  //}

  MPI_Type_free(&coltype);

  if ( myrowrank == rankrow00 ){
  free(ncount);
  free(disp);
  free(row_block_matrix);
  }

  //std::cout<<"DEBUG:mpi_jacobi::distribute_matrix:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),Completed distributing the columns from column 0 "<<std::endl;

  //std::cout<<"DEBUG:mpi_jacobi::distribute_matrix:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),Completed distribute_matrix "<<std::endl;

}









void transpose_bcast_vector(const int n, double* col_vector, double* row_vector, MPI_Comm comm)
{
  // TODO
  //std::cout<<"DEBUG:mpi_jacobi::transpose_bcast_vector:begin with n="<<n<<std::endl;

  // rank in grid
  int myrank;
  int mycoords[2];
  MPI_Comm_rank(comm,&myrank);
  // i,j coord in grid
  MPI_Cart_coords(comm,myrank,2,mycoords);


  //get the row and column block size for each processor
  MPI_Comm commrow,commcol;
  int rdims[2] = {0,1};
  MPI_Cart_sub(comm, rdims,&commrow);
  int ldims[2] = {1,0};
  MPI_Cart_sub(comm,ldims,&commcol);

  //std::cout<<"DEBUG:mpi_jacobi::transpose_bcast_vector:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),created row cart sub for rank="<<myrank<<std::endl;

  // rank in the row and col
  int rowsize;
  int myrowrank;
  int colsize;
  int mycolrank;

  MPI_Comm_size(commrow,&rowsize);
  MPI_Comm_rank(commrow,&myrowrank);
  MPI_Comm_size(commcol,&colsize);
  MPI_Comm_rank(commcol,&mycolrank);


  // rank of root 0,0 matrix in the row comm to be used in mpi_recv
  int rankrow00;
  int rowcoords00 = 0;
  MPI_Cart_rank(commrow,&rowcoords00,&rankrow00);

  // get the rank of the ii processor to send to
  int rowdiagproccords = mycolrank;
  int rankrowdiag;
  MPI_Cart_rank(commrow,&rowdiagproccords,&rankrowdiag);

  // get the rank of the ii processor to send to
  int coldiagproccords = myrowrank;
  int rankcoldiag;
  MPI_Cart_rank(commcol,&coldiagproccords,&rankcoldiag);


  //std::cout<<"DEBUG:mpi_jacobi::transpose_broadcast_vector:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),rowrank="<<myrowrank<<",rowsize="<<rowsize<<",colrank="<<mycolrank<<",colsize="<<colsize<<std::endl;

  // get the column block size
  int colblocksize;
  int fnp = (int)floor(((double)n)/colsize);
  int cnp = (int)ceil( ((double)n)/colsize);

  if ( mycolrank < ( n % colsize )) 
    colblocksize = cnp;
  else
    colblocksize = fnp;

  //send from first col to i,i processors only
  if ( (mycoords[1] == 0 ) ) {

    if (mycoords[0] == 0 ) {

      for (int i=0;i<colblocksize;i++) {
	row_vector[i]=col_vector[i];
      }

    } else {

      MPI_Send(col_vector,colblocksize, MPI_DOUBLE,rankrowdiag,0,commrow);

      //std::cout<<"DEBUG:mpi_jacobi::transpose_bcast_vector:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),sending local_vector to rankrowdiag="<<rankrowdiag<<std::endl;
    }
  }


  //rcv in i,i processors from 0 processor in same row
  if ( (mycoords[0] == mycoords[1]) && (mycoords[1] != 0 ) ) {

    MPI_Recv(row_vector,colblocksize, MPI_DOUBLE,rankrow00,0,commrow,MPI_STATUS_IGNORE);

    //std::cout<<"DEBUG:mpi_jacobi::transpose_bcast_vector:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),recieving local_vector from rankrow00="<<rankrow00<<std::endl;

  }


  // Now broadcast from diagonal processor to all processors in a column
  MPI_Barrier(comm);

  int rowblocksize;
  fnp = (int)floor(((double)n)/rowsize);
  cnp = (int)ceil( ((double)n)/rowsize);

  if ( myrowrank < ( n % rowsize )) 
    rowblocksize = cnp;
  else
    rowblocksize = fnp;

  //std::cout<<"DEBUG:mpi_jacobi::transpose_bcast_vector:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),broadcasting the row_vector of size="<<rowblocksize<<",to all processors in a column, from processor col rank="<<rankcoldiag<<std::endl;

  // broadcast
  MPI_Bcast(row_vector, rowblocksize, MPI_DOUBLE, rankcoldiag, commcol);

  if ( ( mycoords[0] == 0 ) && (mycoords[1] == 0) ){

    // print the rcv buffer
    // double *ptr = row_vector;

    //for (int i=0;i<colblocksize;i++){
      //std::cout<<"DEBUG:mpi_jacobi::transpose_bcast_vector::n="<<n<<"(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),recieved from bcast row_vector["<<i<<"]="<<ptr[i]<<std::endl;
    //}

  }

  //std::cout<<"DEBUG:mpi_jacobi::transpose_bcast_vector:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),Completed transpose_bcast_vector "<<std::endl;

}















void distributed_matrix_vector_mult(const int n, double* local_A, double* local_x, double* local_y, MPI_Comm comm)
{
  // TODO

  //std::cout<<"DEBUG:mpi_jacobi::distributed_matrix_vector_mult:begin"<<std::endl;

  // rank in grid
  int myrank;
  int mycoords[2];
  MPI_Comm_rank(comm,&myrank);

  // i,j coord in grid
  MPI_Cart_coords(comm,myrank,2,mycoords);

  //get the row and column block size for each processor
  MPI_Comm commrow,commcol;
  int rdims[2] = {0,1};
  MPI_Cart_sub(comm, rdims,&commrow);
  int ldims[2] = {1,0};
  MPI_Cart_sub(comm,ldims,&commcol);

  // rank in the row
  int rowsize;
  int myrowrank;
  int colsize;
  int mycolrank;

  MPI_Comm_size(commrow,&rowsize);
  MPI_Comm_rank(commrow,&myrowrank);
  MPI_Comm_size(commcol,&colsize);
  MPI_Comm_rank(commcol,&mycolrank);

  int rankrow00;
  int rowcoords00 = 0;
  MPI_Cart_rank(commrow,&rowcoords00,&rankrow00);

  //std::cout<<"DEBUG:mpi_jacobi::distributed_matrix_vector_mult:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),rowrank="<<myrowrank<<",rowsize="<<rowsize<<",colrank="<<mycolrank<<",colsize="<<colsize<<std::endl;

  // get the row block size
  int rowblocksize;
  int fnp = (int)floor(((double)n)/rowsize);
  int cnp = (int)ceil( ((double)n)/rowsize);

  if ( myrowrank < ( n % rowsize ) ) 
    rowblocksize = cnp;
  else
    rowblocksize = fnp;

  // get the column block size
  int colblocksize;
  fnp = (int)floor(((double)n)/colsize);
  cnp = (int)ceil(((double)n)/colsize);
  if ( mycolrank < ( n % colsize ) ) 
    colblocksize = cnp;
  else
    colblocksize = fnp;

  //std::cout<<"DEBUG:mpi_jacobi::distributed_matrix_vector_mult:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),allocating memory for x row_vector="<<rowblocksize<<std::endl;

  double *row_vector=(double *)malloc(rowblocksize*sizeof(double));

  transpose_bcast_vector(n,local_x,row_vector,comm);

  // multiply the n/p matrix in each processor
  for (int i = 0;i<colblocksize;i++){
    local_y[i]=0;
    for (int j=0;j<rowblocksize;j++){
      local_y[i] = local_y[i] + local_A[colblocksize*j+i]*row_vector[j];  
    }
    //std::cout<<"DEBUG:mpi_jacobi::distributed_matrix_vector_mult:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),at local processor local_y["<<i<<"]="<<local_y[i]<<std::endl;
  }

  //sum up the results to the first column
  //std::cout<<"DEBUG:mpi_jacobi::distributed_matrix_vector_mult:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),completed local matrix multiplication, now reducing results to first column"<<std::endl;

  //reduce the matrix mult results to first column
  MPI_Barrier(comm);

  if ( mycoords[1] == 0 ) 
    MPI_Reduce(MPI_IN_PLACE,local_y,colblocksize,MPI_DOUBLE,MPI_SUM,rankrow00,commrow);
  else
    MPI_Reduce(&local_y[0],NULL,colblocksize,MPI_DOUBLE,MPI_SUM,rankrow00,commrow);

  free(row_vector);

  // print the results in the first col
  //if (mycoords[1] == 0 ) {
  //for ( int i=0;i < colblocksize;i++) {
      //std::cout<<"DEBUG:mpi_jacobi::distributed_matrix_vector_mult:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),reduced y local_y["<<i<<"]="<<local_y[i]<<std::endl;
  //}
  //}

}


















// Solves Ax = b using the iterative jacobi method
void distributed_jacobi(const int n, double* local_A, double* local_b, double* local_x,
			MPI_Comm comm, int max_iter, double l2_termination)
{

  // TODO
  //std::cout<<"DEBUG:mpi_jacobi::distributed_jacobi:begin"<<std::endl;


  // rank in grid
  int myrank;
  int mycoords[2];
  MPI_Comm_rank(comm,&myrank);
  // i,j coord in grid
  MPI_Cart_coords(comm,myrank,2,mycoords);


  //get the row and column block size for each processor
  MPI_Comm commrow,commcol;
  int rdims[2] = {0,1};
  MPI_Cart_sub(comm, rdims,&commrow);
  int ldims[2] = {1,0};
  MPI_Cart_sub(comm,ldims,&commcol);

  //std::cout<<"DEBUG:mpi_jacobi::distributed_jacobi:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),created row cart sub for rank="<<myrank<<std::endl;

  // rank in the row and col
  int rowsize;
  int myrowrank;
  int colsize;
  int mycolrank;

  MPI_Comm_size(commrow,&rowsize);
  MPI_Comm_rank(commrow,&myrowrank);
  MPI_Comm_size(commcol,&colsize);
  MPI_Comm_rank(commcol,&mycolrank);


  // rank of root 0,0 matrix in the row comm to be used in mpi_recv
  int rankrow00;
  int rowcoords00 = 0;
  MPI_Cart_rank(commrow,&rowcoords00,&rankrow00);

  // get the rank of the ii processor to recv from
  int rowdiagproccords = mycolrank;
  int rankrowdiag;
  MPI_Cart_rank(commrow,&rowdiagproccords,&rankrowdiag);

  //std::cout<<"DEBUG:mpi_jacobi::distributed_jacobi:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),rowrank="<<myrowrank<<",rowsize="<<rowsize<<",colrank="<<mycolrank<<",colsize="<<colsize<<std::endl;

  // get the row block size
  int rowblocksize;
  int fnp = (int)floor(((double)n)/rowsize);
  int cnp = (int)ceil( ((double)n)/rowsize);

  if ( myrowrank < ( n % rowsize ) ) 
    rowblocksize = cnp;
  else
    rowblocksize = fnp;

  // get the column block size
  int colblocksize;
  fnp = (int)floor(((double)n)/colsize);
  cnp = (int)ceil( ((double)n)/colsize);

  if ( mycolrank < ( n % colsize )) 
    colblocksize = cnp;
  else
    colblocksize = fnp;

  // remove local the non diagonal elements into r
  //double local_D[colblocksize];
  //double local_R[rowblocksize*colblocksize];
  double *local_D = (double *) malloc(colblocksize*sizeof(double));

  for (int i = 0;i<colblocksize;i++){
    for (int j = 0;j<rowblocksize;j++){
      ////std::cout<<"A("<<i<<","<<j<<")="<<A[i*n+j];
      if ( (mycoords[0] == mycoords[1]) && ( i==j )){
	local_D[i]=local_A[j*colblocksize+i];
	//local_R[j*colblocksize+i] = 0.0;
	//std::cout<<"DEBUG:mpi_jacobi::distributed_jacobi:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),local_D["<<i<<"]="<<local_D[i]<<std::endl;
      }
      //else
//	local_R[j*colblocksize+i]=local_A[j*colblocksize+i];

      //if (( mycoords[1]==1) && (mycoords[0] == 0 )) {
	//std::cout<<"DEBUG:mpi_jacobi::distributed_jacobi:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),local_R["<<i<<","<<j<<"]="<<local_R[j*colblocksize+i]<<std::endl;
      //}
    }
  }
  

  //send D to the first column
  //rcv in i,i processors from 0 processor in same row
  if ( (mycoords[0] == mycoords[1]) && (mycoords[1] != 0 ) ) {

    MPI_Send(local_D,colblocksize, MPI_DOUBLE,rankrow00,0,commrow);

    //std::cout<<"DEBUG:mpi_jacobi::distributed_jacobi:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),sending local_D from rankrowdiag="<<rankrowdiag<<", to rankrow00"<<rankrow00<<std::endl;

  }

  if ( (mycoords[1] == 0 ) && (mycoords[0] != 0) ) {

    MPI_Recv(local_D,colblocksize,MPI_DOUBLE,rankrowdiag,0,commrow,MPI_STATUS_IGNORE);

    //std::cout<<"DEBUG:mpi_jacobi::distributed_jacobi:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),recieving local_D from rankrowdiag="<<rankrowdiag<<std::endl;
    
  }


  for (int i=0;i<colblocksize;i++){
    local_x[i] = 0; 
  }

  double l2norm = l2_termination+1;
  int iter = 0;

  //double local_y[colblocksize];
  double *local_y = (double *)malloc(colblocksize*sizeof(double));

      MPI_Barrier(comm);

      // y=Ax
      distributed_matrix_vector_mult(n,local_A,local_x,local_y,comm);

  while ( (iter < max_iter) && ( l2norm > l2_termination ) )
    {

      //std::cout<<"DEBUG:jacobi.cpp::distributed_jacobi:::(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),starting Iteration="<<iter<<std::endl;


      //std::cout<<"DEBUG:jacobi.cpp::distributed_jacobi:::(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),in Iteration="<<iter<<",crossed barrier"<<std::endl;

      // y = Ax
      ///distributed_matrix_vector_mult(n,local_A,local_x,local_y,comm);

      //std::cout<<"DEBUG:jacobi.cpp::distributed_jacobi:::(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),in Iteration="<<iter<<",completed Rx"<<std::endl;

      // first col operation
      //  get the new local_x
      if ( mycoords[1] == 0 ) {
	// 1/D (b-Rx)
	for (int i = 0;i<colblocksize;i++){
	  local_x[i]=(local_b[i]-(local_y[i]-(local_D[i]*local_x[i])))/local_D[i];
	}
      }
 
      //if ( mycoords[1] == 0 ){
      //for (int i = 0;i<colblocksize;i++){
	  //std::cout<<"DEBUG:mpi_jacobi::distributed_jacobi:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),multiplied y=rx, local_y["<<i<<"]="<<local_y[i]<<std::endl;
      //}
      //}

      // Ax
      distributed_matrix_vector_mult(n,local_A,local_x,local_y,comm);
      //std::cout<<"DEBUG:jacobi.cpp::distributed_jacobi:::(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),in Iteration="<<iter<<",completed Ax"<<std::endl;

      //if ( mycoords[1] == 0 ){
      //for (int i = 0;i<colblocksize;i++){
	  //std::cout<<"DEBUG:mpi_jacobi::distributed_jacobi:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),multiplied y=ax local_y["<<i<<"]="<<local_y[i]<<std::endl;
      //}
      //}

      // ||Ax-B||
      double tl2norm=0;
      // first col operation
      if ( mycoords[1] == 0 ) {
	for (int i = 0;i<colblocksize;i++){
	  tl2norm = tl2norm + pow(local_b[i]-local_y[i],2);
	}

	//std::cout<<"DEBUG:jacobi.cpp::distributed_jacobi:::(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),Iteration="<<iter<<", MPI_ALL_reduce sending tl2norm="<<tl2norm<<std::endl;

      }

      MPI_Barrier(comm);

      MPI_Allreduce(&tl2norm,&l2norm,1,MPI_DOUBLE,MPI_SUM,comm);

      l2norm = sqrt(l2norm);

      //std::cout<<"DEBUG:jacobi.cpp::distributed_jacobi:::(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),completed Iteration="<<iter<<", with l2norm="<<l2norm<<std::endl;

      iter++;

    }

    free(local_D);
    free(local_y);

}









// wraps the distributed matrix vector multiplication
void mpi_matrix_vector_mult(const int n, double* A,
                            double* x, double* y, MPI_Comm comm)
{
  // distribute the array onto local processors!
  double* local_A = NULL;
  double* local_x = NULL;
  distribute_matrix(n, &A[0], &local_A, comm);
  distribute_vector(n, &x[0], &local_x, comm);

  // allocate local result space
  double* local_y = new double[block_decompose_by_dim(n, comm, 0)];
  distributed_matrix_vector_mult(n, local_A, local_x, local_y, comm);

  // gather results back to rank 0
  gather_vector(n, local_y, y, comm);
}




// wraps the distributed jacobi function
void mpi_jacobi(const int n, double* A, double* b, double* x, MPI_Comm comm,
                int max_iter, double l2_termination)
{
  // distribute the array onto local processors!
  double* local_A = NULL;
  double* local_b = NULL;
  distribute_matrix(n, &A[0], &local_A, comm);
  distribute_vector(n, &b[0], &local_b, comm);

  // allocate local result space
  double* local_x = new double[block_decompose_by_dim(n, comm, 0)];
  distributed_jacobi(n, local_A, local_b, local_x, comm, max_iter, l2_termination);

  // gather results back to rank 0
  gather_vector(n, local_x, x, comm);

  // rank in grid
  int myrank;
  int mycoords[2];
  MPI_Comm_rank(comm,&myrank);
  // i,j coord in grid
  MPI_Cart_coords(comm,myrank,2,mycoords);

  //if (( mycoords[1] == 0 ) && (mycoords[0] == 0)){
  //  std::cout<<"DEBUG:mpi_jacobi::distributed_jacobi:(i,j)=("<<mycoords[0]<<","<<mycoords[1]<<"),Results x = [";
  //  for (int i = 0;i<n;i++){
  //   std::cout<<x[i]<<" ,";
  //}
  // std::cout<<" ]"<<std::endl;
  //}

}
