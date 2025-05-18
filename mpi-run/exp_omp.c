#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 3
#define TERMS 50000000

double** create_matrix() {
    double** matrix = (double**)malloc(N * sizeof(double*));
    if (matrix == NULL) {
        perror("malloc failed");
        exit(EXIT_FAILURE);
    }
    double* data = (double*)malloc(N * N * sizeof(double));
    if (data == NULL) {
        perror("malloc failed");
        free(matrix);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < N; i++) {
        matrix[i] = &data[i * N];
    }
    return matrix;
}

void free_matrix(double** matrix) {
    if (matrix != NULL) {
        free(matrix[0]);
        free(matrix);
    }
}

void print_matrix(const char* name, double** matrix) {
    printf("Matrix: %s (%dx%d)\n", name, N, N);
    for (int i = 0; i < N; i++) {
        printf("  [");
        for (int j = 0; j < N; j++) {
            printf("%8.4f%s", matrix[i][j], (j == N - 1) ? "" : " ");
        }
        printf(" ]\n");
    }
    printf("\n");
}

void matrix_multiply(double** A, double** B, double** C) {
    #pragma omp simd collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

int main() {
    double start_time, end_time;

    double** A = create_matrix();
    double** global_term = create_matrix();
    double** result = create_matrix();

    A[0][0] = 0.1; A[0][1] = 0.4; A[0][2] = 0.2;
    A[1][0] = 0.3; A[1][1] = 0.0; A[1][2] = 0.5;
    A[2][0] = 0.6; A[2][1] = 0.2; A[2][2] = 0.1;

    printf("Matrix size: %dx%d\n", N, N);
    printf("Number of terms: %d (+ Identity)\n", TERMS);
    printf("Number of threads: %d\n", omp_get_max_threads());
    print_matrix("A (Initial)", A);

    printf("Starting calculation...\n");
    start_time = omp_get_wtime();

    #pragma omp parallel
    {
        double** local_term = create_matrix();
        double** temp = create_matrix();
        double** thread_result = create_matrix();

        #pragma omp single
        {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    global_term[i][j] = (i == j) ? 1.0 : 0.0;
                    result[i][j] = (i == j) ? 1.0 : 0.0;
                    thread_result[i][j] = 0.0;
                }
            }
        }

        #pragma omp barrier

        #pragma omp for schedule(static)
        for (int k = 1; k <= TERMS; k++) {
            if (k == 1) {
                matrix_multiply(global_term, A, temp);
                for (int i = 0; i < N; i++) {
                    for (int j = 0; j < N; j++) {
                        local_term[i][j] = temp[i][j] / k;
                    }
                }
            } else {
                matrix_multiply(local_term, A, temp);
                for (int i = 0; i < N; i++) {
                    for (int j = 0; j < N; j++) {
                        local_term[i][j] = temp[i][j] / k;
                    }
                }
            }

            #pragma omp simd collapse(2)
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    thread_result[i][j] += local_term[i][j];
                }
            }
        }

        #pragma omp critical
        {
            #pragma omp simd collapse(2)
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    result[i][j] += thread_result[i][j];
                }
            }
        }

        free_matrix(local_term);
        free_matrix(temp);
        free_matrix(thread_result);
    }

    end_time = omp_get_wtime();

    printf("Calculation finished.\n");
    printf("Execution time: %.6f seconds\n", end_time - start_time);
    print_matrix("e^A (Result)", result);

    free_matrix(A);
    free_matrix(global_term);
    free_matrix(result);
    return 0;
}