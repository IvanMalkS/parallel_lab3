#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#define SIZE 3 
#define N_TERMS 50000000 

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
        for (int r = 0; r < SIZE; ++r) {
            for (int c = 0; c < SIZE; ++c) {
                global_taylor_sum[r][c] = final_sum_struct.data[r][c];
            }
        }
    }

    elapsed_time = end_time - start_time;

    if (rank == 0) {
        printf("Matrix size: %dx%d\n", SIZE, SIZE);
        printf("Number of terms: %d (+ Identity)\n", N_TERMS);
        printf("Number of threads: %d\n", num_procs);

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
    if (label != NULL && label[0] != '\0') {
        printf("Matrix: %s (%dx%d)\n", label, SIZE, SIZE);
    }
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
    Matrix result;
    InitializeMatrix(result.data, 0.0);

    int rows_per_proc = SIZE / num_procs;
    int extra_rows = SIZE % num_procs;
    int my_rows = (rank < extra_rows) ? rows_per_proc + 1 : rows_per_proc;
    int my_start_row = (rank < extra_rows) 
                       ? rank * (rows_per_proc + 1) 
                       : extra_rows * (rows_per_proc + 1) + (rank - extra_rows) * rows_per_proc;

    for (int i = my_start_row; i < my_start_row + my_rows; i++) {
        for (int j = 0; j < SIZE; j++) {
            for (int k = 0; k < SIZE; k++) {
                result.data[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    MPI_Reduce(rank == 0 ? MPI_IN_PLACE : result.data, 
               result.data, 
               SIZE * SIZE, 
               MPI_DOUBLE, 
               MPI_SUM, 
               0, 
               MPI_COMM_WORLD);

    return result;
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
    double(*in_matrix)[SIZE] = (double(*)[SIZE])invec;
    double(*inout_matrix)[SIZE] = (double(*)[SIZE])inoutvec;
    if (*len != SIZE * SIZE) {
        fprintf(stderr, "Error in matrix_sum_mpi_op: len mismatch!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            inout_matrix[i][j] += in_matrix[i][j];
        }
    }
}

void CalculateTaylorSum(const double A[SIZE][SIZE],
                        double local_taylor_sum[SIZE][SIZE], int rank,
                        int num_procs) {
    double current_term[SIZE][SIZE];
    double temp_matrix[SIZE][SIZE];


    InitializeMatrix(local_taylor_sum, 0.0);
    MatrixIdentity(current_term);

    long k_start_idx, k_end_idx;
    long num_my_terms;

    long base_count = N_TERMS / num_procs;
    long extra_count = N_TERMS % num_procs;

    if (rank < extra_count) {
        k_start_idx = rank * (base_count + 1) + 1;
        num_my_terms = base_count + 1;
    } else {
        k_start_idx = rank * base_count + extra_count + 1;
        num_my_terms = base_count;
    }
    k_end_idx = k_start_idx + num_my_terms - 1;

    for (long k = k_start_idx; k <= k_end_idx; ++k) {
        Matrix term_A_prod_struct = MatrixMultiply(current_term, A, rank, num_procs);
        for (int r = 0; r < SIZE; ++r) {
            for (int c = 0; c < SIZE; ++c) {
                temp_matrix[r][c] = term_A_prod_struct.data[r][c];
            }
        }
        Matrix actual_term_k_struct =
            MatrixScalarMultiply(temp_matrix, 1.0 / (double)k);
        Matrix new_sum_struct = MatrixAdd(local_taylor_sum, actual_term_k_struct.data);
        for (int r = 0; r < SIZE; ++r) {
            for (int c = 0; c < SIZE; ++c) {
                local_taylor_sum[r][c] = new_sum_struct.data[r][c];
            }
        }
        for (int r = 0; r < SIZE; ++r) {
            for (int c = 0; c < SIZE; ++c) {
                current_term[r][c] = actual_term_k_struct.data[r][c];
            }
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
