#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static double *alloc_matrix(int n) {
    double *m = (double *)malloc((size_t)n * n * sizeof(double));
    if (!m) { fprintf(stderr, "Erro: malloc falhou (n=%d)\n", n); exit(1); }
    return m;
}

static void init_random(double *m, int n) {
    for (int i = 0; i < n * n; i++)
        m[i] = (double)rand() / RAND_MAX;
}

#define M(mat, i, j, n)  ((mat)[(i)*(n) + (j)])

static void matmul_sequential(const double *A,
                               const double *B,
                               double       *C,
                               int           n)
{
    memset(C, 0, (size_t)n * n * sizeof(double));

    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            double a_ik = M(A, i, k, n);
            for (int j = 0; j < n; j++) {
                M(C, i, j, n) += a_ik * M(B, k, j, n);
            }
        }
    }
}

static double max_abs_diff(const double *C1, const double *C2, int n) {
    double d = 0.0;
    for (int i = 0; i < n * n; i++) {
        double diff = fabs(C1[i] - C2[i]);
        if (diff > d) d = diff;
    }
    return d;
}

static void print_matrix(const double *m, int n, int rows, const char *label) {
    printf("%s (primeiros %d×%d):\n", label, rows, rows);
    for (int i = 0; i < rows && i < n; i++) {
        for (int j = 0; j < rows && j < n; j++)
            printf("%8.4f ", M(m, i, j, n));
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char *argv[]) {

    if (argc < 2) {
        fprintf(stderr, "Uso: %s <n> [repeticoes]\n", argv[0]);
        fprintf(stderr, "  n           — ordem da matriz quadrada\n");
        fprintf(stderr, "  repeticoes  — número de execuções (padrão: 5)\n");
        return 1;
    }

    int n    = atoi(argv[1]);
    int reps = (argc >= 3) ? atoi(argv[2]) : 5;

    if (n <= 0 || reps <= 0) {
        fprintf(stderr, "Erro: n e repeticoes devem ser positivos.\n");
        return 1;
    }

    printf("  Multiplicação de Matrizes — Sequencial O(n³)\n\n");
    printf("  n = %d  |  repetições = %d\n", n, reps);

    srand(42);

    double *A = alloc_matrix(n);
    double *B = alloc_matrix(n);
    double *C = alloc_matrix(n);

    init_random(A, n);
    init_random(B, n);

    if (n <= 6) {
        print_matrix(A, n, n, "A");
        print_matrix(B, n, n, "B");
    }

    matmul_sequential(A, B, C, n);

    double total = 0.0, tmin = 1e18, tmax = 0.0;
    double times[reps];

    for (int r = 0; r < reps; r++) {
        double t0 = now_seconds();
        matmul_sequential(A, B, C, n);
        double elapsed = now_seconds() - t0;

        times[r] = elapsed;
        total   += elapsed;
        if (elapsed < tmin) tmin = elapsed;
        if (elapsed > tmax) tmax = elapsed;

        printf("  Rodada %2d: %.6f s\n", r + 1, elapsed);
    }

    double mean = total / reps;

    /* Desvio padrão */
    double var = 0.0;
    for (int r = 0; r < reps; r++) var += (times[r] - mean) * (times[r] - mean);
    double stddev = sqrt(var / reps);

    printf("\n");
    printf("  Média  : %.6f s\n", mean);
    printf("  Mínimo : %.6f s\n", tmin);
    printf("  Máximo : %.6f s\n", tmax);
    printf("  StdDev : %.6f s\n", stddev);

    /* Estimativa de FLOPS (2*n³ operações fp) */
    double flops = 2.0 * (double)n * n * n;
    printf("  GFLOPS : %.3f\n\n", flops / mean / 1e9);

    if (n <= 6) print_matrix(C, n, n, "C = A × B");

    if (n <= 512) {
        FILE *f = fopen("seq_result.bin", "wb");
        if (f) {
            fwrite(&n, sizeof(int), 1, f);
            fwrite(C, sizeof(double), (size_t)n * n, f);
            fclose(f);
            printf("  Resultado salvo em seq_result.bin (para validação paralela)\n");
        }
    }

    free(A); free(B); free(C);
    return 0;
}