/*
 * Vectorscan (Hyperscan) benchmark — the same pattern sets and haystacks
 * as bench/bench_set.mojo, so the numbers are directly comparable.
 *
 * This is the ONE comparison that is genuinely apples-to-apples. Every
 * other baseline (re.finditer per pattern, re alternation, pyahocorasick)
 * answers a weaker question or handles fewer constructs. Hyperscan's
 * block-mode contract is the one emberregex copied: report every id at
 * every position where some match of it ends. Same work, same output.
 *
 * Build (handled by bench/bench_compare_hyperscan.py):
 *   cc -O2 -I .pixi/envs/hs/include comparisons/bench_hyperscan.c \
 *      .pixi/envs/hs/lib/libhs.a -lstdc++ -o comparisons/bench_hyperscan
 *
 * Output: one "name\ttime_us\tmatches" line per row on stdout.
 * Database compile time is NOT included in any measurement — emberregex
 * pays it at build time, Hyperscan at startup, and the point here is
 * scan throughput.
 */

#include <hs/hs.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define REPS 5
#define ITERS 200

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

/* --- corpora, byte-identical to bench/bench_set.mojo --------------------- */

#define HAYSTACK_LEN (16 * 1024)
#define HAYSTACK_LEN_64K (64 * 1024)

static char *sparse_haystack(size_t length, size_t *out_len) {
    const char *filler =
        "the quick brown fox jumps over hazy rivers and empty plains ";
    const char *tail = " cat w17a [42] ERROR 1500ms done\n";
    char *s = malloc(length + 256);
    size_t n = 0;
    while (n < length - 200) {
        size_t k = strlen(filler);
        memcpy(s + n, filler, k);
        n += k;
    }
    size_t tk = strlen(tail);
    memcpy(s + n, tail, tk);
    n += tk;
    while (n < length) s[n++] = 'z';
    *out_len = n;
    return s;
}

static char *dense_haystack(size_t length, size_t *out_len) {
    const char *unit =
        "[7] cat dog w03a w59a ERROR timeout 12ms conn=9 GET /api retry done\n";
    char *s = malloc(length + 256);
    size_t n = 0;
    size_t k = strlen(unit);
    while (n < length) {
        memcpy(s + n, unit, k);
        n += k;
    }
    *out_len = n;
    return s;
}

/* --- pattern sets -------------------------------------------------------- */

static const char *TEDDY8[] = {"cat",   "dog",   "bird",  "fish",
                               "snake", "mouse", "horse", "tiger"};

static const char *LOG[] = {"ERROR", "WARN",     "timeout", "\\d+ms",
                            "conn=\\d+", "retry", "fatal",  "GET /[a-z]+"};

static const char *ROSE_FULL[] = {"ERROR",     "WARN",  "timeout",
                                  "took \\d+", "conn=\\d+", "retry",
                                  "fatal",     "GET /[a-z]+"};

/* w00a .. w63a */
static char teddy64_storage[64][8];
static const char *TEDDY64[64];
static void init_teddy64(void) {
    for (int i = 0; i < 64; i++) {
        snprintf(teddy64_storage[i], sizeof(teddy64_storage[i]), "w%d%da",
                 i / 10, i % 10);
        TEDDY64[i] = teddy64_storage[i];
    }
}

/* --- scanning ------------------------------------------------------------ */

static int on_match(unsigned int id, unsigned long long from,
                    unsigned long long to, unsigned int flags, void *ctx) {
    (void)id;
    (void)from;
    (void)to;
    (void)flags;
    (*(long long *)ctx)++;
    return 0; /* keep going: we want every match, like emberregex */
}

static void run(const char *name, const char **pats, unsigned int count,
                const char *data, size_t len) {
    unsigned int *flags = calloc(count, sizeof(unsigned int));
    unsigned int *ids = calloc(count, sizeof(unsigned int));
    for (unsigned int i = 0; i < count; i++) ids[i] = i;

    hs_database_t *db = NULL;
    hs_compile_error_t *err = NULL;
    if (hs_compile_multi(pats, flags, ids, count, HS_MODE_BLOCK, NULL, &db,
                         &err) != HS_SUCCESS) {
        fprintf(stderr, "%s: compile failed: %s\n", name,
                err ? err->message : "?");
        hs_free_compile_error(err);
        free(flags);
        free(ids);
        return;
    }

    hs_scratch_t *scratch = NULL;
    if (hs_alloc_scratch(db, &scratch) != HS_SUCCESS) {
        fprintf(stderr, "%s: scratch alloc failed\n", name);
        hs_free_database(db);
        free(flags);
        free(ids);
        return;
    }

    long long matches = 0;
    /* warm up */
    hs_scan(db, data, (unsigned int)len, 0, scratch, on_match, &matches);

    double best = 1e30;
    for (int r = 0; r < REPS; r++) {
        matches = 0;
        double t0 = now_us();
        for (int i = 0; i < ITERS; i++) {
            hs_scan(db, data, (unsigned int)len, 0, scratch, on_match,
                    &matches);
        }
        double dt = (now_us() - t0) / ITERS;
        if (dt < best) best = dt;
    }
    printf("%s\t%.4f\t%lld\n", name, best, matches / ITERS);

    hs_free_scratch(scratch);
    hs_free_database(db);
    free(flags);
    free(ids);
}

/* NOTE: the first execution of a freshly built binary reads ~40% slow on
 * the early rows (macOS validates the new signature on first exec). An
 * in-process warmup does NOT fix it — it is a per-execution cost, not a
 * per-database one. bench_compare_hyperscan.py runs this binary twice and
 * keeps the second. Run it by hand and ignore the first run's top rows. */

int main(void) {
    init_teddy64();
    size_t sl, dl, s64l;
    char *sparse = sparse_haystack(HAYSTACK_LEN, &sl);
    char *dense = dense_haystack(HAYSTACK_LEN, &dl);
    char *sparse64 = sparse_haystack(HAYSTACK_LEN_64K, &s64l);

    run("teddy8_sparse_16k", TEDDY8, 8, sparse, sl);
    run("teddy8_dense_16k", TEDDY8, 8, dense, dl);
    run("teddy64_sparse_16k", TEDDY64, 64, sparse, sl);
    run("teddy64_dense_16k", TEDDY64, 64, dense, dl);
    run("log_sparse_16k", LOG, 8, sparse, sl);
    run("log_dense_16k", LOG, 8, dense, dl);
    run("log_sparse_64k", LOG, 8, sparse64, s64l);
    run("full_sparse_64k", ROSE_FULL, 8, sparse64, s64l);
    run("full_dense_16k", ROSE_FULL, 8, dense, dl);

    free(sparse);
    free(dense);
    free(sparse64);
    return 0;
}
