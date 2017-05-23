# Makefile for HPC 6220 Programming Assignment 1
CXX=mpic++
#CCFLAGS=-Wall -g
# activate for compiler optimizations:
CCFLAGS=-Wall -O3
LDFLAGS=
LIBS=-pthread

# set up google test
GTEST_DIR = ./gtest
CCFLAGS += -I. #-I$(GTEST_DIR)




all: jacobi tests

test: tests
	echo "### TESTING SEQUENTIAL CODE ###";./seq_tests; \
	echo "### TESTING WITH 4 PROCESSES ###"; mpirun -np 4 ./mpi_tests \
	echo "### TESTING WITH 9 PROCESSES ###"; mpirun -np 9 ./mpi_tests

tests: seq_tests mpi_tests

jacobi: main.o jacobi.o mpi_jacobi.o utils.o
	$(CXX) $(LDFLAGS) -o $@ $^

seq_tests: seq_tests.o mpi_gtest.o gtest-all.o jacobi.o utils.o
	$(CXX) $(LDFLAGS) $(LIBS) -o $@ $^

mpi_tests: mpi_tests.o mpi_gtest.o gtest-all.o mpi_jacobi.o jacobi.o utils.o
	$(CXX) $(LDFLAGS) $(LIBS) -o $@ $^

gtest-all.o : $(GTEST_DIR)/gtest-all.cc $(GTEST_DIR)/gtest.h
	$(CXX) $(CCFLAGS) -c $(GTEST_DIR)/gtest-all.cc

%.o: %.cpp %.h
	$(CXX) $(CCFLAGS) -c $<

%.o: %.cpp
	$(CXX) $(CCFLAGS) -c $<

clean:
	rm -f *.o jacobi seq_tests mpi_tests
