#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#define M(mat, i, j, n)  ((mat)[(i)*(n) + (j)])

typedef struct {
    const double *A;
    const double *B;
    double       *C;
    int           n;
    int           row_start;
    int           row_end;
    int           tid;
} ThreadArg;

static void *worker(void *arg) {
    ThreadArg *a = (ThreadArg *)arg;
    int n         = a->n;
    int rs        = a->row_start;
    int re        = a->row_end;

    memset(a->C + (size_t)rs * n, 0, (size_t)(re - rs) * n * sizeof(double));

    for (int i = rs; i < re; i++) {
        for (int k = 0; k < n; k++) {
            double a_ik = M(a->A, i, k, n);
            for (int j = 0; j < n; j++) {
                M(a->C, i, j, n) += a_ik * M(a->B, k, j, n);
            }
        }
    }
    return NULL;
}

static void matmul_parallel(const double *A,
                             const double *B,
                             double       *C,
                             int           n,
                             int           p)
{
    pthread_t  *threads = (pthread_t  *)malloc((size_t)p * sizeof(pthread_t));
    ThreadArg  *args    = (ThreadArg  *)malloc((size_t)p * sizeof(ThreadArg));
    if (!threads || !args) { fprintf(stderr, "malloc erro\n"); exit(1); }

    int base  = n / p;
    int extra = n % p;
    int row   = 0;

    for (int t = 0; t < p; t++) {
        int chunk = base + (t < extra ? 1 : 0);

        args[t].A         = A;
        args[t].B         = B;
        args[t].C         = C;
        args[t].n         = n;
        args[t].row_start = row;
        args[t].row_end   = row + chunk;
        args[t].tid       = t;
        row += chunk;

        pthread_create(&threads[t], NULL, worker, &args[t]);
    }

    for (int t = 0; t < p; t++)
        pthread_join(threads[t], NULL);

    free(threads);
    free(args);
}

static double *alloc_matrix(int n) {
    double *m = (double *)malloc((size_t)n * n * sizeof(double));
    if (!m) { fprintf(stderr, "malloc falhou\n"); exit(1); }
    return m;
}

static void init_random(double *m, int n, unsigned seed) {
    srand(seed);
    for (int i = 0; i < n * n; i++)
        m[i] = (double)rand() / RAND_MAX;
}

static double max_abs_diff(const double *C1, const double *C2, int n) {
    double d = 0.0;
    for (int i = 0; i < n * n; i++) {
        double diff = fabs(C1[i] - C2[i]);
        if (diff > d) d = diff;
    }
    return d;
}

static void print_matrix(const double *m, int n, int rows, const char *lbl) {
    printf("%s (primeiros %d×%d):\n", lbl, rows, rows);
    for (int i = 0; i < rows && i < n; i++) {
        for (int j = 0; j < rows && j < n; j++)
            printf("%8.4f ", M(m, i, j, n));
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char *argv[]) {

    if (argc < 3) {
        fprintf(stderr, "Uso: %s <n> <threads> [repeticoes]\n", argv[0]);
        fprintf(stderr, "  n          — ordem da matriz\n");
        fprintf(stderr, "  threads    — número de threads POSIX\n");
        fprintf(stderr, "  repeticoes — (padrão: 5)\n");
        return 1;
    }

    int n    = atoi(argv[1]);
    int p    = atoi(argv[2]);
    int reps = (argc >= 4) ? atoi(argv[3]) : 5;

    if (n <= 0 || p <= 0 || reps <= 0) {
        fprintf(stderr, "Erro: argumentos devem ser positivos.\n");
        return 1;
    }

    printf("  Multiplicação de Matrizes — Paralela (pthreads)\n");
    printf("  n = %d  |  threads = %d  |  repetições = %d\n\n", n, p, reps);

    double *A  = alloc_matrix(n);
    double *B  = alloc_matrix(n);
    double *C  = alloc_matrix(n);

    init_random(A, n, 42);
    init_random(B, n, 42);

    if (n <= 6) {
        print_matrix(A, n, n, "A");
        print_matrix(B, n, n, "B");
    }

    matmul_parallel(A, B, C, n, p);

    FILE *fval = fopen("seq_result.bin", "rb");
    if (fval) {
        int n_ref;
        fread(&n_ref, sizeof(int), 1, fval);
        if (n_ref == n) {
            double *C_ref = alloc_matrix(n);
            fread(C_ref, sizeof(double), (size_t)n * n, fval);
            fclose(fval);
            double err = max_abs_diff(C, C_ref, n);
            printf("  Validação vs. sequencial: erro máx = %.2e  %s\n\n",
                   err, err < 1e-8 ? "[OK]" : "[FALHOU]");
            free(C_ref);
        } else {
            fclose(fval);
            printf("  (seq_result.bin tem n=%d, ignorado)\n\n", n_ref);
        }
    } else {
        // printf("  (seq_result.bin não encontrado — execute matmul_seq primeiro)\n\n");
    }

    double total = 0.0, tmin = 1e18, tmax = 0.0;
    double times[reps];

    for (int r = 0; r < reps; r++) {
        double t0 = now_seconds();
        matmul_parallel(A, B, C, n, p);
        double elapsed = now_seconds() - t0;

        times[r]  = elapsed;
        total    += elapsed;
        if (elapsed < tmin) tmin = elapsed;
        if (elapsed > tmax) tmax = elapsed;

        printf("  Rodada %2d: %.6f s\n", r + 1, elapsed);
    }

    double mean = total / reps;
    double var  = 0.0;
    for (int r = 0; r < reps; r++) var += (times[r] - mean) * (times[r] - mean);
    double stddev = sqrt(var / reps);

    printf("\n");
    printf("  Média  : %.6f s\n", mean);
    printf("  Mínimo : %.6f s\n", tmin);
    printf("  Máximo : %.6f s\n", tmax);
    printf("  StdDev : %.6f s\n", stddev);
    printf("  GFLOPS : %.3f\n",   2.0 * (double)n * n * n / mean / 1e9);

    if (n <= 6) print_matrix(C, n, n, "C = A × B");

    free(A); free(B); free(C);
    return 0;
}