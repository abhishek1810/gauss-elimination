/* Gaussian elimination without pivoting using PThreads.
 * Compile with "gcc -pthread -o gauss_pthreads gauss_pthreads.c"
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
#include <pthread.h>

/* Program Parameters */
#define MAXN 2000 /* Max value of N */
int N;            /* Matrix size */
int numThreads;   /* Number of threads */

/* Matrices and vectors */
volatile float A[MAXN][MAXN], B[MAXN], X[MAXN];
/* A * X = B, solve for X */

/* junk */
// #define randm() 4|2[uid]&3

/* Prototype */
void gauss(); /* The function you will provide.
               * It is this routine that is timed.
               * It is called only on the parent.
               */

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
  float a[3][3] = {49847.82, 55979.54, 65456.24, 45641.20, 13289.77, 35694.64, 58389.08, 14441.43, 3227.41};
  float b[3] = {9194.05, 37332.16, 44779.23};
  
  for (col = 0; col < N; col++)
  {
    for (row = 0; row < N; row++)
    {
      A[row][col] = (float)rand() / 32768.0;
      // A[row][col] = a[row][col];
    }
    B[col] = (float)rand() / 32768.0;
    // B[col] = b[col];
    X[col] = 0.0;
  }
}

/* Print input matrices */
void print_inputs()
{
  int row, col;

  if (N < 10)
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
  /* Timing variables */
  struct timeval etstart, etstop; /* Elapsed times using gettimeofday() */
  struct timezone tzdummy;
  // clock_t etstart2, etstop2;  /* Elapsed times using times() */
  unsigned long long usecstart, usecstop;
  struct tms cputstart, cputstop; /* CPU times for my processes */

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

  /* Gaussian Elimination */
  gauss();

  /* Stop Clock */
  gettimeofday(&etstop, &tzdummy);
  times(&cputstop);
  printf("Stopped clock.\n");
  usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
  usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;

  /* Display output */
  print_X();

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

  exit(0);
}

/* ------------------ Above Was Provided --------------------- */

/****** You will replace this routine with your own parallel version *******/
/* Provided global variables are MAXN, N, A[][], B[], and X[],
 * defined in the beginning of this code.  X[] is initialized to zeros.
 */

#define THREADS 10

// Declaration for the function executed by the threads
void *routine(void *_args);

// Structure to pass the required data to the threads
struct thread_args {
    int threadId;
    int norm;
};

void gauss()
{
  int norm, row, col; /* Normalization row, and zeroing
                       * element row and col */

  // Array to maintain array pointers
  pthread_t threads[numThreads];

  printf("Computing Parallelly.\n");

  /* Gaussian elimination */
  for (norm = 0; norm < N - 1; norm++)
  {
    // Loop to intialize the threads
    for (int i = 0; i < numThreads; i++)
    {
      struct thread_args *args = malloc (sizeof (struct thread_args));
      args->threadId = i;
      args->norm = norm;
      pthread_create(&threads[i], NULL, &routine, args);
    }
    // Synchronizing the threads
    for (int i = 0; i < numThreads; i++)
    {
      pthread_join(threads[i], NULL);
    }
  }
  /* (Diagonal elements are not normalized to 1.  This is treated in back
   * substitution.)
   */

  /* Back substitution */
  for (row = N - 1; row >= 0; row--)
  {
    X[row] = B[row];
    for (col = N - 1; col > row; col--)
    {
      X[row] -= A[row][col] * X[col];
    }
    X[row] /= A[row][row];
  }
}

// Routine executed by the threads
void *routine(void *_args)
{
  //Collecting the arguments passed to the thread
  struct thread_args *args = (struct thread_args *) _args;
  int norm = args->norm;
  int threadId = args->threadId;

  //Intializing variable needed for the loop
  int col, row = norm + threadId + 1;
  float multiplier;

  //Inner loop implementation 
  for (; row < N; row += THREADS)
  {
    multiplier = A[row][norm] / A[norm][norm];
    for (col = norm; col < N; col++)
    {
      A[row][col] -= A[norm][col] * multiplier;
    }
    B[row] -= B[norm] * multiplier;
  }
}
