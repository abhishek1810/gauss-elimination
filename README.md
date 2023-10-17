# gauss-elimination
Performing Guass elimination using different parallel techniques


# MPI

The code can be compiled using the following command

```
mpicc gauss_mpi.c -o gauss_mpi
```

For running the code

```
mpirun -np 2 ./gauss_mpi 4 0 1
```

Arguments are as follows
1. Array Dimension
2. Random seed 
3. Number of threads (Can be any value. I didn't change the code for accepting the arguments from the serial program)

# Pthreads

The code can be compiled using these commands

```
cc -pthread -o gauss_pthreads gauss_pthreads.c
```

For running the code

```
./gauss_pthreads 2000 0 10
```
Arguments are as follows
1. Array Dimension
2. Random seed
3. Number of threads

# OpenMp

The code can be compiled using these commands

```
cc -fopenmp -o gauss_openmp gauss_openmp.c
```

For running the code

```
./gauss_openmp 2000 0 10
```
Arguments are as follows
1. Array Dimension
2. Random seed
3. Number of threads
