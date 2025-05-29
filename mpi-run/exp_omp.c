#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define SIZE 3
#define N_TERMS 5000

void MatrixPrint(const char* label, const double matrix[SIZE][SIZE]);
void MatrixAdd(double result[SIZE][SIZE], 
              const double matrix1[SIZE][SIZE],
              const double matrix2[SIZE][SIZE]);
void MatrixMultiply(double result[SIZE][SIZE],
                   const double matrix1[SIZE][SIZE],
                   const double matrix2[SIZE][SIZE]);
void MatrixIdentity(double matrix[SIZE][SIZE]);
void MatrixScalarMultiply(double result[SIZE][SIZE],
                         const double matrix[SIZE][SIZE],
                         const double scalar);
void InitializeMatrix(double matrix[SIZE][SIZE], double value);
void MatrixCopy(double dest[SIZE][SIZE], const double src[SIZE][SIZE]);
void MatrixPower(double result[SIZE][SIZE], const double matrix[SIZE][SIZE], int power);

int main() {
    double A[SIZE][SIZE] = {
        {0.1, 0.4, 0.2}, {0.3, 0.0, 0.5}, {0.6, 0.2, 0.1}};
    double taylor_sum[SIZE][SIZE] = {0};
    double identity[SIZE][SIZE];
    double start_time, end_time;

    start_time = omp_get_wtime();

    // Вычисление экспоненты матрицы с использованием ряда Тейлора
    #pragma omp parallel
    {
        double local_sum[SIZE][SIZE] = {0};
        double term[SIZE][SIZE];
        double A_power[SIZE][SIZE];
        double factorial = 1.0;
        
        // Первый член (k=0) - единичная матрица (обрабатывается после параллельного региона)
        
        // Начинаем с k=1
        MatrixCopy(A_power, A);
        factorial = 1.0;
        
        #pragma omp for schedule(static)
        for (int k = 1; k <= N_TERMS; k++) {
            factorial *= k;
            
            // Вычисляем A^k / k!
            MatrixScalarMultiply(term, A_power, 1.0 / factorial);
            
            // Добавляем к локальной сумме
            MatrixAdd(local_sum, local_sum, term);
            
            // Умножаем на A для следующей итерации
            if (k < N_TERMS) {
                double temp[SIZE][SIZE];
                MatrixMultiply(temp, A_power, A);
                MatrixCopy(A_power, temp);
            }
        }
        
        #pragma omp critical
        {
            MatrixAdd(taylor_sum, taylor_sum, local_sum);
        }
    }
    
    // Добавляем единичную матрицу (член для k=0)
    MatrixIdentity(identity);
    MatrixAdd(taylor_sum, taylor_sum, identity);

    end_time = omp_get_wtime();

    printf("Matrix size: %dx%d\n", SIZE, SIZE);
    printf("Number of terms: %d (+ Identity)\n", N_TERMS);
    printf("Number of threads: %d\n", omp_get_max_threads());

    printf("Matrix: A (Initial) (%dx%d)\n", SIZE, SIZE);
    MatrixPrint("A", A);

    printf("Calculation finished.\n");
    printf("Execution time: %f seconds\n", end_time - start_time);

    printf("Matrix: e^A (Result) (%dx%d)\n", SIZE, SIZE);
    MatrixPrint("e^A", taylor_sum);

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

void MatrixAdd(double result[SIZE][SIZE], 
              const double matrix1[SIZE][SIZE],
              const double matrix2[SIZE][SIZE]) {
    #pragma omp simd collapse(2)
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            result[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }
}

void MatrixMultiply(double result[SIZE][SIZE],
                   const double matrix1[SIZE][SIZE],
                   const double matrix2[SIZE][SIZE]) {
    #pragma omp simd collapse(2)
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            result[i][j] = 0.0;
            for (int k = 0; k < SIZE; k++) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
}

void MatrixIdentity(double matrix[SIZE][SIZE]) {
    #pragma omp simd collapse(2)
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            matrix[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
}

void MatrixScalarMultiply(double result[SIZE][SIZE],
                         const double matrix[SIZE][SIZE],
                         const double scalar) {
    #pragma omp simd collapse(2)
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            result[i][j] = matrix[i][j] * scalar;
        }
    }
}

void InitializeMatrix(double matrix[SIZE][SIZE], double value) {
    #pragma omp simd collapse(2)
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            matrix[i][j] = value;
        }
    }
}

void MatrixCopy(double dest[SIZE][SIZE], const double src[SIZE][SIZE]) {
    #pragma omp simd collapse(2)
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            dest[i][j] = src[i][j];
        }
    }
}

void MatrixPower(double result[SIZE][SIZE], const double matrix[SIZE][SIZE], int power) {
    double temp[SIZE][SIZE];
    MatrixIdentity(result);
    
    for (int p = 0; p < power; p++) {
        MatrixMultiply(temp, result, matrix);
        MatrixCopy(result, temp);
    }
}
