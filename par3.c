#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"


#define VECTOR_SIZE 10
#define RANK_ROOT 0
#define TAU 0.01

void vector_fill(double * a, double * x, double * b, int n, double fill_value) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j)
                a[i * n + j] = 2;
            else
                a[i * n + j] = 1;
        }
    }
    for (int i = 0; i < n; i++) {
        x[i] = fill_value;
        b[i] = VECTOR_SIZE + 1;
    }
}

double run(int size, int rank, const double * a, double * x, double * b, double * y_total, int n) {
    int i, j;
    int n_partial = n / size;

    double *a_partial = (double *) malloc(n_partial * n * sizeof(double));//блоки строк матрицы на каждом процессе
    double *y_partial = (double *) malloc(n_partial * sizeof(double));//блоки результирующего вектора на каждом процессе
    //рассылаем вектор x
    MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(b, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //разделяем матрицу на горизонтальные полосы
    //шириной n_partial и отправляем процессам
    MPI_Scatter(a, n_partial * n, MPI_DOUBLE, a_partial, n_partial * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double norma = 0.0;
    double dop = 0.0;
    for (i = 0; i < n_partial; i++) {
        y_partial[i] = -1 * b[i];
        for (j = 0; j < VECTOR_SIZE; j++)
            y_partial[i] += a_partial[i * n + j] * x[j];

        if (rank != size - 1 || i < VECTOR_SIZE % n_partial){
            norma += y_partial[i] * y_partial[i];
            dop += b[i] * b[i];
        }
        y_partial[i] = x[i] - TAU * y_partial[i];
    }

    //собираем результат в нулевом процессе
    MPI_Gather(y_partial, n_partial, MPI_DOUBLE, y_total, n_partial, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double sum1 = 0.0;
    double sum2 = 0.0;
    MPI_Reduce(&dop, &sum2, 1, MPI_DOUBLE, MPI_SUM, RANK_ROOT, MPI_COMM_WORLD);
    MPI_Reduce(&norma, &sum1, 1, MPI_DOUBLE, MPI_SUM, RANK_ROOT, MPI_COMM_WORLD);
    free(a_partial);
    free(y_partial);
    return sqrt(sum1) / sqrt(sum2);
}

int main(int argc, char **argv) {
    int rank, size;
    double e = 0.00001;
    int n = VECTOR_SIZE;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    n += (size - n % size) % size;
    double *a = (double *) malloc(n * n * sizeof(double));//исходная матрица
    double *x = (double *) malloc(n * sizeof(double)); //исходный вектор
    double *b = (double *) malloc(n * sizeof(double)); //исходный вектор
    double *result = (double *) malloc(n * sizeof(double));// вектор-результат

    double t = MPI_Wtime();
    if (rank == 0) {
        vector_fill(a, x, b, n, 0);
    }
    double norma = 1;
    double dop = 0;
    double *nor = (double *) malloc(2 * sizeof(double));
    nor[0] = 1;
    while (nor[0] > e) {
        dop = run(size, rank, a, x, b, result, n);
        if (rank == RANK_ROOT)
            nor[0] = dop;
        MPI_Bcast(nor, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        for (int i = 0; i < n; i++)
            x[i] = result[i];
    }
    t = MPI_Wtime() - t;
    MPI_Finalize();
    if (rank == RANK_ROOT){
        for (int i = 0; i < VECTOR_SIZE; i++)
            printf("%10.5f\n", x[i]);
    }
    printf("time = %10.5f\n", t);
    free(a);
    free(b);
    free(x);
    free(result);
    return 0;
}
