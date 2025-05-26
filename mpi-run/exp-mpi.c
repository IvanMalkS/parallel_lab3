#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 3
#define N_TERMS 5000

typedef struct {
    double data[SIZE][SIZE];
} Matrix;

void MatrixPrint(const char* label, const double matrix[SIZE][SIZE]);
Matrix MatrixAdd(const double matrix1[SIZE][SIZE],
                 const double matrix2[SIZE][SIZE]);

Matrix MatrixMultiply(const double matrix1[SIZE][SIZE],
                     const double matrix2[SIZE][SIZE],
                     int rank, int num_procs);
void MatrixIdentity(double matrix[SIZE][SIZE]);
Matrix MatrixScalarMultiply(const double matrix[SIZE][SIZE],
                            const double scalar);
void MatrixSumMpiOp(void* invec, void* inoutvec, int* len,
                    MPI_Datatype* datatype);
void CalculateTaylorSum(const double A[SIZE][SIZE],
                        double local_taylor_sum[SIZE][SIZE], int rank,
                        int num_procs);
void InitializeMatrix(double matrix[SIZE][SIZE], double value);
void MatrixCopy(double dest[SIZE][SIZE], const double src[SIZE][SIZE]);

int main(int argc, char* argv[]) {
    int rank, num_procs;
    double A[SIZE][SIZE] = {
        {0.1, 0.4, 0.2}, {0.3, 0.0, 0.5}, {0.6, 0.2, 0.1}};
    double global_taylor_sum[SIZE][SIZE];
    double start_time, end_time, elapsed_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    InitializeMatrix(global_taylor_sum, 0.0);

    start_time = MPI_Wtime();

    double local_taylor_sum[SIZE][SIZE];
    CalculateTaylorSum(A, local_taylor_sum, rank, num_procs);

    MPI_Op matrix_sum_op;
    MPI_Op_create(MatrixSumMpiOp, 1, &matrix_sum_op);

    MPI_Reduce(local_taylor_sum, global_taylor_sum, SIZE * SIZE, MPI_DOUBLE,
               matrix_sum_op, 0, MPI_COMM_WORLD);

    MPI_Op_free(&matrix_sum_op);

    end_time = MPI_Wtime();

    if (rank == 0) {
        double identity[SIZE][SIZE];
        MatrixIdentity(identity);
        Matrix final_sum_struct = MatrixAdd(global_taylor_sum, identity);
        MatrixCopy(global_taylor_sum, final_sum_struct.data);
    }

    elapsed_time = end_time - start_time;

    if (rank == 0) {
        printf("Matrix size: %dx%d\n", SIZE, SIZE);
        printf("Number of terms: %d (+ Identity)\n", N_TERMS);
        printf("Number of processors: %d\n", num_procs);

        printf("Matrix: A (Initial) (%dx%d)\n", SIZE, SIZE);
        MatrixPrint("A", A);

        printf("Calculation finished.\n");
        printf("Execution time: %f seconds\n", elapsed_time);

        printf("Matrix: e^A (Result) (%dx%d)\n", SIZE, SIZE);
        MatrixPrint("e^A", global_taylor_sum);
    }

    MPI_Finalize();
    return 0;
}

void MatrixPrint(const char* label, const double matrix[SIZE][SIZE]) {
    for (int i = 0; i < SIZE; i++) {
        printf("[ ");
        for (int j = 0; j < SIZE; j++) {
            printf("%8.4f ", matrix[i][j]);
        }
        printf("]\n");
    }
}

Matrix MatrixAdd(const double matrix1[SIZE][SIZE],
                 const double matrix2[SIZE][SIZE]) {
    Matrix result;
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            result.data[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }
    return result;
}

Matrix MatrixMultiply(const double A[SIZE][SIZE], const double B[SIZE][SIZE], int rank, int num_procs) {
    Matrix partial_result;
    InitializeMatrix(partial_result.data, 0.0);

    int rows_per_proc = SIZE / num_procs;
    int extra_rows = SIZE % num_procs;
    int my_rows_count = (rank < extra_rows) ? rows_per_proc + 1 : rows_per_proc;
    int my_start_row = (rank < extra_rows)
                       ? rank * (rows_per_proc + 1)
                       : extra_rows * (rows_per_proc + 1) + (rank - extra_rows) * rows_per_proc;

    for (int i = my_start_row; i < my_start_row + my_rows_count; i++) {
        for (int j = 0; j < SIZE; j++) {
            for (int k_inner = 0; k_inner < SIZE; k_inner++) {
                partial_result.data[i][j] += A[i][k_inner] * B[k_inner][j];
            }
        }
    }

    Matrix global_result;

    MPI_Allreduce(partial_result.data,
                  global_result.data,
                  SIZE * SIZE,
                  MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD);

    return global_result;
}

void MatrixIdentity(double matrix[SIZE][SIZE]) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            matrix[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
}

Matrix MatrixScalarMultiply(const double matrix[SIZE][SIZE],
                            const double scalar) {
    Matrix result;
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            result.data[i][j] = matrix[i][j] * scalar;
        }
    }
    return result;
}

void MatrixSumMpiOp(void* invec, void* inoutvec, int* len,
                    MPI_Datatype* datatype) {
    double (*in_matrix_ptr)[SIZE] = (double(*)[SIZE])invec;
    double (*inout_matrix_ptr)[SIZE] = (double(*)[SIZE])inoutvec;

    if (*len != SIZE * SIZE) {
        fprintf(stderr, "Error in MatrixSumMpiOp: len mismatch! Expected %d, got %d\n", SIZE * SIZE, *len);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            inout_matrix_ptr[i][j] += in_matrix_ptr[i][j];
        }
    }
}

void CalculateTaylorSum(const double A[SIZE][SIZE],
                        double local_taylor_sum[SIZE][SIZE], int rank,
                        int num_procs) {
    InitializeMatrix(local_taylor_sum, 0.0);
    Matrix temp_matrix_struct;

    double A_P[SIZE][SIZE];
    if (num_procs == 0) {
        MPI_Abort(MPI_COMM_WORLD, 1);
        return;
    }
    if (num_procs == 1) {
        MatrixCopy(A_P, A);
    } else {
        MatrixCopy(A_P, A);

        for (int i = 1; i < num_procs; ++i) {
            temp_matrix_struct = MatrixMultiply(A_P, A, rank, num_procs);
            MatrixCopy(A_P, temp_matrix_struct.data);
        }
    }

    double current_term_val[SIZE][SIZE];

    long k_first_for_proc = (long)rank + 1;

    if (k_first_for_proc <= N_TERMS) {
        double A_power_k_first[SIZE][SIZE];
        MatrixIdentity(A_power_k_first);

        for (long p = 1; p <= k_first_for_proc; ++p) {
            temp_matrix_struct = MatrixMultiply(A_power_k_first, A, rank, num_procs);
            MatrixCopy(A_power_k_first, temp_matrix_struct.data);
        }

        double factorial_k_first = 1.0;
        for (long p = 1; p <= k_first_for_proc; ++p) {
            if (p > 0) factorial_k_first *= (double)p;
        }
         if (k_first_for_proc == 0) factorial_k_first = 1.0;

        temp_matrix_struct = MatrixScalarMultiply(A_power_k_first, 1.0 / factorial_k_first);
        MatrixCopy(current_term_val, temp_matrix_struct.data);

        Matrix new_sum_struct = MatrixAdd(local_taylor_sum, current_term_val);
        MatrixCopy(local_taylor_sum, new_sum_struct.data);

        for (long k_current = k_first_for_proc; k_current <= N_TERMS - num_procs; k_current += num_procs) {
            long k_next = k_current + num_procs;

            temp_matrix_struct = MatrixMultiply(current_term_val, A_P, rank, num_procs);
            MatrixCopy(current_term_val, temp_matrix_struct.data);

            double product_for_denominator = 1.0;
            for (long val = k_current + 1; val <= k_next; ++val) {
                 if (val > 0) product_for_denominator *= (double)val;
            }
            if (product_for_denominator == 0) {
                 fprintf(stderr, "Rank %d: product_for_denominator is zero at k_current=%ld, k_next=%ld\n", rank, k_current, k_next);
                 MPI_Abort(MPI_COMM_WORLD, 1);
            }

            temp_matrix_struct = MatrixScalarMultiply(current_term_val, 1.0 / product_for_denominator);
            MatrixCopy(current_term_val, temp_matrix_struct.data);

            new_sum_struct = MatrixAdd(local_taylor_sum, current_term_val);
            MatrixCopy(local_taylor_sum, new_sum_struct.data);
        }
    }
}

void InitializeMatrix(double matrix[SIZE][SIZE], double value) {
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            matrix[i][j] = value;
        }
    }
}

void MatrixCopy(double dest[SIZE][SIZE], const double src[SIZE][SIZE]) {
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            dest[i][j] = src[i][j];
        }
    }
}
