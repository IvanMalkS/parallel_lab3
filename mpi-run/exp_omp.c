#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>


#define SIZE 3       
#define N_TERMS 50000000 

typedef struct {
    double data[SIZE][SIZE];
} Matrix;

void MatrixPrint(const char* label, const double matrix[SIZE][SIZE]);
Matrix MatrixAdd(const double matrix1[SIZE][SIZE],
                 const double matrix2[SIZE][SIZE]);
Matrix MatrixMultiply(const double matrix1[SIZE][SIZE],
                      const double matrix2[SIZE][SIZE]);
void MatrixIdentity(double matrix[SIZE][SIZE]);
Matrix MatrixScalarMultiply(const double matrix[SIZE][SIZE],
                            const double scalar);
void CalculateTaylorSum(const double A[SIZE][SIZE],
                        double taylor_sum[SIZE][SIZE], int num_threads);
void InitializeMatrix(double matrix[SIZE][SIZE], double value);

int main(int argc, char* argv[]) {
    double A[SIZE][SIZE] = {
        {0.1, 0.4, 0.2}, {0.3, 0.0, 0.5}, {0.6, 0.2, 0.1}};
    double taylor_sum[SIZE][SIZE];
    double start_time, end_time, elapsed_time;
    int num_threads;

    if (argc > 1) {
        num_threads = atoi(argv[1]);
    } else {
        num_threads = omp_get_max_threads();
    }

    omp_set_num_threads(num_threads);

    InitializeMatrix(taylor_sum, 0.0);

    start_time = omp_get_wtime();

    CalculateTaylorSum(A, taylor_sum, num_threads);

    end_time = omp_get_wtime();

    double identity[SIZE][SIZE];
    MatrixIdentity(identity);
    Matrix final_sum_struct = MatrixAdd(taylor_sum, identity);
    for (int r = 0; r < SIZE; ++r) {
        for (int c = 0; c < SIZE; ++c) {
            taylor_sum[r][c] = final_sum_struct.data[r][c];
        }
    }

    elapsed_time = end_time - start_time;

    printf("Matrix size: %dx%d\n", SIZE, SIZE);
    printf("Number of terms: %d (+ Identity)\n", N_TERMS);
    printf("Number of threads: %d\n", num_threads);

    printf("Matrix: A (Initial) (%dx%d)\n", SIZE, SIZE);
    MatrixPrint("A", A);

    printf("Calculation finished.\n");
    printf("Execution time: %f seconds\n", elapsed_time);

    printf("Matrix: e^A (Result) (%dx%d)\n", SIZE, SIZE);
    MatrixPrint("e^A", taylor_sum);

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

Matrix MatrixMultiply(const double matrix1[SIZE][SIZE],
                      const double matrix2[SIZE][SIZE]) {
    Matrix result;
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            result.data[i][j] = 0.0;
            for (int k = 0; k < SIZE; k++) {
                result.data[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
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

void CalculateTaylorSum(const double A[SIZE][SIZE],
                        double taylor_sum[SIZE][SIZE], int num_threads) {
    double current_term[SIZE][SIZE];
    double temp_matrix[SIZE][SIZE];
    InitializeMatrix(taylor_sum, 0.0);
    MatrixIdentity(current_term);      

    #pragma omp parallel num_threads(num_threads) default(none) shared(A, taylor_sum, current_term, temp_matrix, num_threads)
    {
        double local_sum[SIZE][SIZE];
        InitializeMatrix(local_sum, 0.0);

        #pragma omp for schedule(dynamic)
        for (long k = 1; k <= N_TERMS; ++k) {
            Matrix term_A_prod_struct = MatrixMultiply(current_term, A);
            for (int r = 0; r < SIZE; ++r) {
                for (int c = 0; c < SIZE; ++c) {
                    temp_matrix[r][c] = term_A_prod_struct.data[r][c];
                }
            }
            Matrix actual_term_k_struct =
                MatrixScalarMultiply(temp_matrix, 1.0 / (double)k);
            Matrix new_sum_struct = MatrixAdd(local_sum, actual_term_k_struct.data);
             for (int r = 0; r < SIZE; ++r) {
                for (int c = 0; c < SIZE; ++c) {
                    local_sum[r][c] = new_sum_struct.data[r][c];
                }
            }

            for (int r = 0; r < SIZE; ++r) {
                for (int c = 0; c < SIZE; ++c) {
                    current_term[r][c] = actual_term_k_struct.data[r][c];
                }
            }

        }

        #pragma omp critical
        {
           Matrix overall_sum_struct = MatrixAdd(taylor_sum, local_sum);
            for (int r = 0; r < SIZE; ++r) {
                for (int c = 0; c < SIZE; ++c) {
                   taylor_sum[r][c] = overall_sum_struct.data[r][c];
                }
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