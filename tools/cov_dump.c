/* Counter dump for run_coverage.py's default (inline-aware) mode.
 *
 * The instrumented IR exports the three globals below; this destructor
 * writes the counters as little-endian u64s when the test binary exits
 * normally. It lives in C so the IR never has to call libc itself: Mojo
 * declares fclose/write with its own integer prototypes, which a textual
 * IR call would have to match exactly. */
#include <stdint.h>
#include <stdio.h>

extern uint64_t __emberregex_cov_ctrs[];
extern const uint64_t __emberregex_cov_n;
extern const char __emberregex_cov_path[];

__attribute__((destructor)) static void emberregex_cov_dump(void) {
    FILE *f = fopen(__emberregex_cov_path, "wb");
    if (!f) {
        return;
    }
    fwrite(__emberregex_cov_ctrs, sizeof(uint64_t), __emberregex_cov_n, f);
    fclose(f);
}
