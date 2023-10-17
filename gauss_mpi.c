/* Gaussian elimination without pivoting.
 */

/* ****** ADD YOUR CODE AT THE END OF THIS FILE. ******
 * You need not submit the provided code.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <time.h>
#include <mpi.h>

/* Program Parameters */
#define MAXN 3000 /* Max value of N */
int N;            /* Matrix size */
int numThreads;   /* Number of threads */

/* Matrices and vectors */
float A[MAXN][MAXN], B[MAXN], X[MAXN];
/* A * X = B, solve for X */

/* junk */
// #define randm() 4|2[uid]&3

/* Prototype */
void gauss(); /* The function you will provide.
               * It is this routine that is timed.
               * It is called only on the parent.
               */

/* Prototype */
/* This function implements the gaussian elimination routine*/
void routine(int norm, int my_rank, int p);

/* Prototype */
/* This function distrbutes matrix A based on static interleaving*/
void send_data_to_process(int my_rank, int p);

/* returns a seed for srand based on the time */
unsigned int time_seed()
{
  struct timeval t;
  struct timezone tzdummy;

  gettimeofday(&t, &tzdummy);
  return (unsigned int)(t.tv_usec);
}

/* Set the program parameters from the command-line arguments */
void parameters(int argc, char **argv)
{
  int seed = 0; /* Random seed */
  // char uid[32]; /*User name */

  /* Read command-line arguments */
  srand(time_seed()); /* Randomize */

  if (argc == 4)
  {
    seed = atoi(argv[2]);
    srand(seed);
    numThreads = atoi(argv[3]);
    printf("Random seed = %i\n", seed);
  }
  if (argc >= 2)
  {
    N = atoi(argv[1]);
    if (N < 1 || N > MAXN)
    {
      printf("N = %i is out of range.\n", N);
      exit(0);
    }
  }
  else
  {
    printf("Usage: %s <matrix_dimension> [random seed]\n",
           argv[0]);
    exit(0);
  }

  /* Print parameters */
  printf("\nMatrix dimension N = %i.\n", N);
}

/* Initialize A and B (and X to 0.0s) */
void initialize_inputs()
{
  int row, col;

  printf("\nInitializing...\n");
  for (col = 0; col < N; col++)
  {
    for (row = 0; row < N; row++)
    {
      A[row][col] = (float)rand() / 32768.0;
    }
    B[col] = (float)rand() / 32768.0;
    X[col] = 0.0;
  }
}

/* Print input matrices */
void print_inputs()
{
  int row, col;

  if (N < 17)
  {
    printf("\nA =\n\t");
    for (row = 0; row < N; row++)
    {
      for (col = 0; col < N; col++)
      {
        printf("%5.2f%s", A[row][col], (col < N - 1) ? ", " : ";\n\t");
      }
    }
    printf("\nB = [");
    for (col = 0; col < N; col++)
    {
      printf("%5.2f%s", B[col], (col < N - 1) ? "; " : "]\n");
    }
  }
}

void print_X()
{
  int row;

  if (N < 100)
  {
    printf("\nX = [");
    for (row = 0; row < N; row++)
    {
      printf("%5.2f%s", X[row], (row < N - 1) ? "; " : "]\n");
    }
  }
}

int main(int argc, char **argv)
{

  int my_rank, p;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  double starttime = 0.0;
  double endtime = 0.0;

  /* Timing variables */
  struct timeval etstart, etstop; /* Elapsed times using gettimeofday() */
  struct timezone tzdummy;
  // clock_t etstart2, etstop2;  /* Elapsed times using times() */
  unsigned long long usecstart, usecstop;
  struct tms cputstart, cputstop; /* CPU times for my processes */

  if (my_rank == 0)
  {
    /* Process program parameters */
    parameters(argc, argv);

    /* Initialize A and B */
    initialize_inputs();

    /* Print input matrices */
    print_inputs();

    /* Start Clock */
    printf("\nStarting clock.\n");
    gettimeofday(&etstart, &tzdummy);
    times(&cputstart);
    starttime = MPI_Wtime();

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }
  else
  {
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }

  gauss();

  if (my_rank == 0)
  {
    /* Stop Clock */
    gettimeofday(&etstop, &tzdummy);
    times(&cputstop);
    endtime = MPI_Wtime();
    printf("Stopped clock.\n");
    usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
    usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;

    /* Display output */
    print_X();

    /* Display timing results */
    printf("\nMPI_Wtime = %f s.\n", (endtime - starttime));

    /* Display timing results */
    printf("\nElapsed time = %g ms.\n",
           (float)(usecstop - usecstart) / (float)1000);

    printf("(CPU times are accurate to the nearest %g ms)\n",
           1.0 / (float)CLOCKS_PER_SEC * 1000.0);
    printf("My total CPU time for parent = %g ms.\n",
           (float)((cputstop.tms_utime + cputstop.tms_stime) -
                   (cputstart.tms_utime + cputstart.tms_stime)) /
               (float)CLOCKS_PER_SEC * 1000);
    printf("My system CPU time for parent = %g ms.\n",
           (float)(cputstop.tms_stime - cputstart.tms_stime) /
               (float)CLOCKS_PER_SEC * 1000);
    printf("My total CPU time for child processes = %g ms.\n",
           (float)((cputstop.tms_cutime + cputstop.tms_cstime) -
                   (cputstart.tms_cutime + cputstart.tms_cstime)) /
               (float)CLOCKS_PER_SEC * 1000);
    /* Contrary to the man pages, this appears not to include the parent */
    printf("--------------------------------------------\n");
  }

  MPI_Finalize();
  exit(0);
}

/* ------------------ Above Was Provided --------------------- */

/****** You will replace this routine with your own parallel version *******/
/* Provided global variables are MAXN, N, A[][], B[], and X[],
 * defined in the beginning of this code.  X[] is initialized to zeros.
 */
void gauss()
{
  int norm;
  int my_rank, p;

  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  // Sending data based on static interleaving
  send_data_to_process(my_rank, p);

  for (int norm = 0; norm < N; norm++)
  {
    // Broadcasting the row at index norm to all processes
    if (norm % p == my_rank)
    {
      MPI_Bcast(A[norm], N, MPI_FLOAT, my_rank, MPI_COMM_WORLD);
      MPI_Bcast(&B[norm], 1, MPI_FLOAT, my_rank, MPI_COMM_WORLD);
    }
    else
    {
      MPI_Bcast(A[norm], N, MPI_FLOAT, norm % p, MPI_COMM_WORLD);
      MPI_Bcast(&B[norm], 1, MPI_FLOAT, norm % p, MPI_COMM_WORLD);
    }

    // Implementing Gaussian elimination
    if (norm < (N - 1))
    {
      routine(norm, my_rank, p);
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }

  /* (Diagonal elements are not normalized to 1.  This is treated in back
   * substitution.)
   */

  /* Back substitution */
  if (my_rank == 0)
  {
    /* Back substitution */
    for (int row = N - 1; row >= 0; row--)
    {
      X[row] = B[row];
      for (int col = N - 1; col > row; col--)
      {
        X[row] -= A[row][col] * X[col];
      }
      X[row] /= A[row][row];
    }
  }
}

void routine(int norm, int my_rank, int p)
{
  int row, col;
  float multiplier;
  for (row = norm + 1; row < N; row++)
  {
    if (row % p == my_rank)
    {
      // if ( my_rank == 3 ) {
      //   printf("row - %d", row);
      //   printf("norm - %d", norm);
      // }
      multiplier = A[row][norm] / A[norm][norm];
      for (col = norm; col < N; col++)
      {
        A[row][col] -= A[norm][col] * multiplier;
      }
      B[row] -= B[norm] * multiplier;
      // if ( my_rank == 3 ) {
      //   print_inputs();
      // }
    }
  }
}

void send_data_to_process(int my_rank, int p)
{
  int source = 0;
  int dest;
  int i;
  MPI_Status status;

  if (my_rank == 0)
  {
    for (i = 0; i < N; i++)
    {
      dest = i % p;
      if (dest != 0)
      {
        // printf("Sending A[%d] and B[%d] to %d\n", i, i, dest);
        MPI_Send(A[i], N, MPI_FLOAT, dest, i, MPI_COMM_WORLD);
        MPI_Send(&B[i], 1, MPI_FLOAT, dest, i + N, MPI_COMM_WORLD);
      }
    }
  }
  else
  {
    for (i = 0; i < N; i++)
    {
      dest = i % p;
      if (dest == my_rank)
      {
        MPI_Recv(A[i], N, MPI_FLOAT, source, i, MPI_COMM_WORLD, &status);
        MPI_Recv(&B[i], 1, MPI_FLOAT, source, i + N, MPI_COMM_WORLD, &status);
      }
    }
  }
}
