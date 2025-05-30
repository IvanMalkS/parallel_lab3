#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

#define MATRIX_ORDER 3
#define TAYLOR_SERIES_TERMS 50000000

// Выделение памяти для квадратной матрицы dimension x dimension
double** create_square_matrix(int dimension) {
    double** matrix_buffer;
    matrix_buffer = (double**)malloc(dimension * sizeof(double*));
    if (matrix_buffer == NULL) {
        fprintf(stderr, "FATAL ERROR: Cannot allocate memory for matrix row pointers.\n");
        return NULL;
    }
    // Единый блок памяти для всех элементов для эффективности MPI_Bcast/Reduce
    matrix_buffer[0] = (double*)calloc((long long)dimension * dimension, sizeof(double)); // calloc для инициализации нулями
    if (matrix_buffer[0] == NULL) {
        fprintf(stderr, "FATAL ERROR: Cannot allocate memory for matrix elements.\n");
        free(matrix_buffer);
        return NULL;
    }
    for (int i = 1; i < dimension; i++) {
        matrix_buffer[i] = matrix_buffer[0] + (long long)i * dimension;
    }
    return matrix_buffer;
}

// Освобождение памяти матрицы
void destroy_square_matrix(double** matrix_buffer) {
    if (matrix_buffer != NULL) {
        if (matrix_buffer[0] != NULL) {
            free(matrix_buffer[0]); // Освобождаем основной блок данных
        }
        free(matrix_buffer); // Освобождаем массив указателей
    }
}

// Вывод матрицы
void print_matrix_contents(const char* label, double** matrix_buffer, int dimension) {
    printf("Matrix '%s' (%dx%d):\n", label, dimension, dimension);
    for (int i = 0; i < dimension; i++) {
        printf("  [");
        for (int j = 0; j < dimension; j++) {
            printf("%8.4f%s", matrix_buffer[i][j], (j == dimension - 1) ? "" : ", ");
        }
        printf("]\n");
    }
    printf("\n");
}

// Матрица с нулями
void set_matrix_to_zeros(double** matrix_buffer, int dimension) {
    for (int i = 0; i < dimension; i++) {
        for (int j = 0; j < dimension; j++) {
            matrix_buffer[i][j] = 0.0;
        }
    }
}

// Инициализация единичной матрицы
void set_matrix_to_identity(double** matrix_buffer, int dimension) {
    set_matrix_to_zeros(matrix_buffer, dimension); // Сначала обнуляем
    for (int i = 0; i < dimension; i++) {
        matrix_buffer[i][i] = 1.0;
    }
}

// Копирование матрицы: dest_matrix = src_matrix
void copy_matrix_values(double** src_matrix, double** dest_matrix, int dimension) {
    for (int i = 0; i < dimension; i++) {
        for (int j = 0; j < dimension; j++) {
            dest_matrix[i][j] = src_matrix[i][j];
        }
    }
}

// Умножение матрицы на скаляр: matrix_buffer = matrix_buffer * scalar
void scale_matrix_elements(double** matrix_buffer, double scalar, int dimension) {
    for (int i = 0; i < dimension; i++) {
        for (int j = 0; j < dimension; j++) {
            matrix_buffer[i][j] *= scalar;
        }
    }
}

// Сложение матриц: target_matrix = target_matrix + source_matrix
void add_matrix_to_target(double** target_matrix, double** source_matrix, int dimension) {
    for (int i = 0; i < dimension; i++) {
        for (int j = 0; j < dimension; j++) {
            target_matrix[i][j] += source_matrix[i][j];
        }
    }
}

// Умножение матриц: result_matrix = matrix_left * matrix_right
void multiply_matrices_standard(double** matrix_left, double** matrix_right, double** result_matrix, int dimension) {
    set_matrix_to_zeros(result_matrix, dimension); // Обнуляем результат перед умножением
    for (int i = 0; i < dimension; i++) {
        for (int j = 0; j < dimension; j++) {
            for (int k = 0; k < dimension; k++) {
                result_matrix[i][j] += matrix_left[i][k] * matrix_right[k][j];
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Матрицы, необходимые для вычислений
    double** matrix_A_initial = create_square_matrix(MATRIX_ORDER); // Исходная матрица A
    double** current_taylor_term = create_square_matrix(MATRIX_ORDER); // Текущий вычисляемый член ряда A^k/k!
    double** previous_taylor_term = create_square_matrix(MATRIX_ORDER); // Предыдущий член ряда (для T_k = T_{k-1}*A/k)
    double** temp_matrix_product = create_square_matrix(MATRIX_ORDER); // Для промежуточного умножения
    double** process_partial_sum = create_square_matrix(MATRIX_ORDER); // Частичная сумма членов на данном процессе
    double** matrix_exponential_result = NULL; // Финальный результат e^A (только на процессе 0)

    if (mpi_rank == 0) {
        matrix_exponential_result = create_square_matrix(MATRIX_ORDER);
    }

    // Проверка успешности выделения памяти
    if (!matrix_A_initial || !current_taylor_term || !previous_taylor_term || !temp_matrix_product || !process_partial_sum || (mpi_rank == 0 && !matrix_exponential_result)) {
        fprintf(stderr, "[MPI Rank %d] CRITICAL: Memory allocation failed for matrices. Terminating.\n", mpi_rank);
        MPI_Abort(MPI_COMM_WORLD, 1); // Прерываем все процессы
    }

    // Инициализация исходной матрицы A на процессе 0
    if (mpi_rank == 0) {
        printf("MPI Calculation of Matrix Exponential e^A using Taylor Series\n");
        printf("Lab Specific Configuration:\n");
        printf("  - Matrix Order (N): %d\n", MATRIX_ORDER);
        printf("  - Taylor Series Terms (K_max): %d (A^1/1! to A^%d/%d!)\n", TAYLOR_SERIES_TERMS, TAYLOR_SERIES_TERMS, TAYLOR_SERIES_TERMS);
        printf("  - MPI Processes: %d\n\n", mpi_size);

        // начальные значения A
        matrix_A_initial[0][0] = 1; matrix_A_initial[0][1] = 1; matrix_A_initial[0][2] = 1;
        matrix_A_initial[1][0] = 1; matrix_A_initial[1][1] = 1; matrix_A_initial[1][2] = 1;
        matrix_A_initial[2][0] = 1;matrix_A_initial[2][1] = 1; matrix_A_initial[2][2] = 1;

        print_matrix_contents("Initial Matrix A", matrix_A_initial, MATRIX_ORDER); // Для отладки
    }

    // Рассылка матрицы A всем процессам
    MPI_Bcast(matrix_A_initial[0], MATRIX_ORDER * MATRIX_ORDER, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Инициализация частичной суммы на каждом процессе нулями
    set_matrix_to_zeros(process_partial_sum, MATRIX_ORDER);

    // Подготовка к вычислению первого члена ряда (A/1!)
    // previous_taylor_term будет "виртуальным" T_0 = I (единичная матрица)
    // Это нужно для формулы Term_k = (Term_{k-1} * A) / k, где k начинается с 1.
    set_matrix_to_identity(previous_taylor_term, MATRIX_ORDER);

    MPI_Barrier(MPI_COMM_WORLD); // Синхронизация перед началом замера времени
    double start_time_computation = MPI_Wtime();

    // Цикл по членам ряда Тейлора, начиная с k=1 (A/1!)
    for (int k_term_index = 1; k_term_index <= TAYLOR_SERIES_TERMS; k_term_index++) {
        // Вычисляем текущий член ряда: current_taylor_term = (previous_taylor_term * matrix_A_initial) / k_term_index
        multiply_matrices_standard(previous_taylor_term, matrix_A_initial, temp_matrix_product, MATRIX_ORDER);
        copy_matrix_values(temp_matrix_product, current_taylor_term, MATRIX_ORDER);
        scale_matrix_elements(current_taylor_term, 1.0 / (double)k_term_index, MATRIX_ORDER);

        // Распределение вычисления членов ряда по процессам
        // Процесс mpi_rank вычисляет члены, для которых (индекс_члена - 1) % mpi_size == mpi_rank
        // (k_term_index - 1) потому что k_term_index начинается с 1
        if (((k_term_index - 1) % mpi_size) == mpi_rank) {
            add_matrix_to_target(process_partial_sum, current_taylor_term, MATRIX_ORDER);
        }

        // Обновляем previous_taylor_term для следующей итерации
        copy_matrix_values(current_taylor_term, previous_taylor_term, MATRIX_ORDER);
    }

    MPI_Barrier(MPI_COMM_WORLD); // Синхронизация перед сборкой результатов
    double end_time_computation = MPI_Wtime();

    // Сбор (редукция) частичных сумм со всех процессов на процесс 0
    // Результат суммируется в matrix_exponential_result на процессе 0
    if (mpi_rank == 0) {
        set_matrix_to_zeros(matrix_exponential_result, MATRIX_ORDER); // Обнуляем перед MPI_Reduce
    }
    MPI_Reduce(process_partial_sum[0], // Отправляемый буфер (данные текущего процесса)
               matrix_exponential_result ? matrix_exponential_result[0] : NULL, // Принимающий буфер (только на root)
               MATRIX_ORDER * MATRIX_ORDER, // Количество элементов
               MPI_DOUBLE,                  // Тип данных
               MPI_SUM,                     // Операция (сумма)
               0,                           // Корневой процесс (rank 0)
               MPI_COMM_WORLD);             // Коммуникатор

    // На процессе 0: добавляем единичную матрицу I (нулевой член ряда) и выводим результат
    if (mpi_rank == 0) {
        set_matrix_to_identity(current_taylor_term, MATRIX_ORDER); // Используем current_taylor_term как временную для I
        add_matrix_to_target(matrix_exponential_result, current_taylor_term, MATRIX_ORDER); // Result = Result_from_Reduce + I

        double total_parallel_time = end_time_computation - start_time_computation;
        printf("\n--- Computation Finished ---\n");
        printf("Total parallel execution time: %.6f seconds\n", total_parallel_time); // Важная строка для парсера!

         print_matrix_contents("Final Result e^A (Sum of Terms + Identity)", matrix_exponential_result, MATRIX_ORDER);
    }

    // Освобождение всей выделенной памяти
    destroy_square_matrix(matrix_A_initial);
    destroy_square_matrix(current_taylor_term);
    destroy_square_matrix(previous_taylor_term);
    destroy_square_matrix(temp_matrix_product);
    destroy_square_matrix(process_partial_sum);
    if (mpi_rank == 0) {
        destroy_square_matrix(matrix_exponential_result);
    }

    MPI_Finalize();
    return 0;
}