/*
 * PCRE2 JIT benchmark — covers the shared BenchIds from bench_static.mojo.
 *
 * Build (handled by bench/bench_compare_pcre2.py):
 *   cmake -S comparisons/pcre2 -B comparisons/pcre2_build \
 *         -DCMAKE_BUILD_TYPE=Release \
 *         -DPCRE2_BUILD_PCRE2GREP=OFF -DPCRE2_BUILD_TESTS=OFF \
 *         -DPCRE2_SUPPORT_JIT=ON -DBUILD_SHARED_LIBS=OFF
 *   cmake --build comparisons/pcre2_build -- -j$(nproc 2>/dev/null || echo 4)
 *   cc -O2 -I comparisons/pcre2_build -I comparisons/pcre2/src \
 *      comparisons/bench_pcre2.c \
 *      comparisons/pcre2_build/libpcre2-8.a \
 *      -o comparisons/bench_pcre2
 *
 * Output: one "name\ttime_us" line per benchmark to stdout.
 * JIT compile time is NOT included in any measurement.
 */

#define PCRE2_CODE_UNIT_WIDTH 8
#include "pcre2.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---------------------------------------------------------------------------
 * Config
 * --------------------------------------------------------------------------- */

#define REPS          5       /* repetitions; take the minimum */
#define ITERS_NORMAL  10000   /* iterations for fast, short-input benchmarks   */
#define ITERS_MEDIUM  1000    /* findall/replace heavy, multiline, log search  */
#define ITERS_LIGHT   100     /* 1MB inputs and very slow pathological         */

/* ---------------------------------------------------------------------------
 * Timing
 * --------------------------------------------------------------------------- */

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1000.0;
}

/* ---------------------------------------------------------------------------
 * PCRE2 helpers
 * --------------------------------------------------------------------------- */

typedef struct {
    pcre2_code        *re;
    int                has_jit;
} RE;

static RE jit_compile(const char *pattern, uint32_t options) {
    int errcode;
    PCRE2_SIZE erroffset;
    pcre2_code *re = pcre2_compile((PCRE2_SPTR)pattern,
                                   PCRE2_ZERO_TERMINATED,
                                   options, &errcode, &erroffset, NULL);
    if (!re) {
        PCRE2_UCHAR buf[256];
        pcre2_get_error_message(errcode, buf, sizeof(buf));
        fprintf(stderr, "pcre2_compile error at offset %zu: %s  (pattern: %s)\n",
                erroffset, (char *)buf, pattern);
        exit(1);
    }
    int rc = pcre2_jit_compile(re, PCRE2_JIT_COMPLETE);
    RE result;
    result.re      = re;
    result.has_jit = (rc == 0);
    if (!result.has_jit) {
        fprintf(stderr, "[warn] JIT compile failed (rc=%d) for pattern: %s\n",
                rc, pattern);
    }
    return result;
}

/* Convenience wrappers that fall back to pcre2_match if JIT unavailable. */
static int do_match(RE *r, PCRE2_SPTR subj, size_t len,
                    PCRE2_SIZE offset, uint32_t flags,
                    pcre2_match_data *md) {
    return pcre2_jit_match(r->re, subj, len, offset, flags, md, NULL);
}

/* ---------------------------------------------------------------------------
 * Benchmark helpers
 * --------------------------------------------------------------------------- */

/* search — finds first match anywhere (no PCRE2_ANCHORED) */
static double bench_search(RE *r, const char *subj, size_t slen, int iters) {
    pcre2_match_data *md = pcre2_match_data_create_from_pattern(r->re, NULL);
    double best = 1e18;
    for (int rep = 0; rep < REPS; rep++) {
        double t0 = now_us();
        for (int i = 0; i < iters; i++)
            do_match(r, (PCRE2_SPTR)subj, slen, 0, 0, md);
        double us = (now_us() - t0) / iters;
        if (us < best) best = us;
    }
    pcre2_match_data_free(md);
    return best;
}

/* match — anchored at start (PCRE2_ANCHORED) */
static double bench_match(RE *r, const char *subj, size_t slen, int iters) {
    pcre2_match_data *md = pcre2_match_data_create_from_pattern(r->re, NULL);
    double best = 1e18;
    for (int rep = 0; rep < REPS; rep++) {
        double t0 = now_us();
        for (int i = 0; i < iters; i++)
            do_match(r, (PCRE2_SPTR)subj, slen, 0, PCRE2_ANCHORED, md);
        double us = (now_us() - t0) / iters;
        if (us < best) best = us;
    }
    pcre2_match_data_free(md);
    return best;
}


/* replace — uses pcre2_substitute with PCRE2_SUBSTITUTE_GLOBAL */
static double bench_replace(RE *r, const char *subj, size_t slen,
                             const char *repl, size_t rlen,
                             int iters) {
    static PCRE2_UCHAR outbuf[1 << 24]; /* 16 MB shared output buffer */
    double best = 1e18;
    for (int rep = 0; rep < REPS; rep++) {
        double t0 = now_us();
        for (int i = 0; i < iters; i++) {
            PCRE2_SIZE outlen = sizeof(outbuf);
            pcre2_substitute(r->re,
                             (PCRE2_SPTR)subj, slen,
                             0,
                             PCRE2_SUBSTITUTE_GLOBAL | PCRE2_SUBSTITUTE_EXTENDED,
                             NULL, NULL,
                             (PCRE2_SPTR)repl, rlen,
                             outbuf, &outlen);
        }
        double us = (now_us() - t0) / iters;
        if (us < best) best = us;
    }
    return best;
}


/* Print result line */
static void emit(const char *name, double us) {
    printf("%s\t%.3f\n", name, us);
}

/* ---------------------------------------------------------------------------
 * Input generators
 * --------------------------------------------------------------------------- */

/* Returns malloc'd buffer: 'c' repeated n times, then suffix, then '\0'. */
static char *repeat_char(char c, size_t n, const char *suffix) {
    size_t slen = suffix ? strlen(suffix) : 0;
    char *buf = (char *)malloc(n + slen + 1);
    if (!buf) { perror("malloc"); exit(1); }
    memset(buf, c, n);
    if (suffix) memcpy(buf + n, suffix, slen);
    buf[n + slen] = '\0';
    return buf;
}

/* Returns malloc'd buffer: word repeated n times, joined by sep. */
static char *repeat_word(const char *word, const char *sep, int n) {
    size_t wlen = strlen(word), slen = strlen(sep);
    size_t total = wlen * n + slen * (n > 0 ? n - 1 : 0) + 1;
    char *buf = (char *)malloc(total);
    if (!buf) { perror("malloc"); exit(1); }
    char *p = buf;
    for (int i = 0; i < n; i++) {
        if (i > 0) { memcpy(p, sep, slen); p += slen; }
        memcpy(p, word, wlen); p += wlen;
    }
    *p = '\0';
    return buf;
}

/* Returns malloc'd buffer: 100 log lines (line 750 = ERROR, rest INFO). */
static char *make_log_lines(int n) {
    /* max ~80 chars per line */
    char *buf = (char *)malloc((size_t)n * 80 + 1);
    if (!buf) { perror("malloc"); exit(1); }
    char *p = buf;
    for (int i = 0; i < n; i++) {
        int len;
        if (i == 750) {
            len = sprintf(p, "2026-03-21 14:30:05 [ERROR] Something broke\n");
        } else {
            len = sprintf(p, "2026-03-21 14:30:05 [INFO] All good line %d\n", i);
        }
        p += len;
    }
    /* remove trailing newline */
    if (p > buf && *(p-1) == '\n') { p--; }
    *p = '\0';
    return buf;
}

/* Returns malloc'd buffer: n lines "line {i} some text here", joined by '\n' */
static char *make_lines(int n) {
    char *buf = (char *)malloc((size_t)n * 30 + 1);
    if (!buf) { perror("malloc"); exit(1); }
    char *p = buf;
    for (int i = 0; i < n; i++) {
        if (i > 0) { *p++ = '\n'; }
        p += sprintf(p, "line %d some text here", i);
    }
    *p = '\0';
    return buf;
}

/* ---------------------------------------------------------------------------
 * Main
 * --------------------------------------------------------------------------- */

int main(void) {

    /* -----------------------------------------------------------------------
     * Pre-allocate all inputs
     * --------------------------------------------------------------------- */
    char *in_100B        = repeat_char('a', 94, "needle");
    char *in_10KB        = repeat_char('a', 10000, "needle");
    char *in_100KB       = repeat_char('a', 100000, "needle");
    char *in_1MB         = repeat_char('a', 1000000, "needle");
    char *in_class_10KB  = repeat_char('a', 9990, "xyzxyzxyz");
    char *in_nomatch_100KB = repeat_char('a', 100000, NULL);
    char *in_100_lines   = make_lines(100);
    char *in_log_lines   = make_log_lines(1000);
    char *in_dots_500    = repeat_char('a', 500, NULL);
    char *in_replace_50  = repeat_word("42", " text ", 50);
    char *in_dotstar_5K  = repeat_char('a', 5000, "x");
    char *in_dotstar_miss_5K = repeat_char('a', 5000, NULL);

    /* Fixed-length string inputs (stack OK) */
    const char *in_hello_world       = "hello world";
    const char *in_say_hello         = "say hello world today";
    char *in_aaaa_10KB               = repeat_char('a', 10000, NULL);
    const char *in_dotall_body       = "<body>\nline1\nline2\nline3\n</body>";
    const char *in_date              = "2026-03-21";
    const char *in_email             = "user@example.com";
    const char *in_password          = "MyP4ssw0rd";
    const char *in_delta             = "delta";
    const char *in_pi                = "pi";
    const char *in_sigma             = "sigma";
    const char *in_john_doe          = "John Doe";
    const char *in_patho_16          = "aaaaaaaaaaaaaaaa";
    const char *in_patho_backref     = "hello hello hello";
    const char *in_patho_nested      = "aaaaaaaaaaaaaaaa";
    const char *in_url               = "https://www.example.com/path/to/page?q=1&r=2";
    const char *in_phone             = "(555) 123-4567";
    const char *in_hex               = "#1a2B3c";
    const char *in_semver            = "12.34.56-beta.1";
    const char *in_kv                = "host=localhost port=5432 db=mydb user=admin timeout=30";
    const char *in_html = "<html><head><title>Test</title></head>"
                          "<body><div class=\"x\"><p>Hello</p>"
                          "<a href=\"#\">Link</a></div></body></html>";
    const char *in_ws_raw            = "hello   world\t\tfoo  bar\n\nbaz   qux";
    const char *in_inline_ig         = "HeLLo WoRLd";
    const char *in_inline_ml         = "info: ok\nwarn: hmm\nerror: bad\ninfo: ok";
    const char *in_dfa               = "abc123def";
    const char *in_backref           = "abc123abc";

    /* -----------------------------------------------------------------------
     * Compile + JIT-compile ALL patterns (excluded from timing)
     * --------------------------------------------------------------------- */

    RE re_needle       = jit_compile("needle",    0);
    RE re_xyz          = jit_compile("[xyz]+",    0);
    RE re_zzz_6        = jit_compile("zzzzzz",   0);
    RE re_hello        = jit_compile("^hello",   0);
    RE re_world_dollar = jit_compile("world$",   0);
    RE re_word_b       = jit_compile("\\bworld\\b", 0);
    RE re_orld_b       = jit_compile("\\borld\\b",  0);
    RE re_zzz_bol      = jit_compile("^zzz",     0);
    RE re_ml_bol       = jit_compile("(?m)^\\w+",   0);
    RE re_ml_eol       = jit_compile("(?m)\\w+$",   0);
    RE re_dotall       = jit_compile("(?s)<body>.*</body>", 0);
    RE re_date         = jit_compile("(?P<year>\\d{4})-(?P<month>\\d{2})-(?P<day>\\d{2})", 0);
    RE re_email_named  = jit_compile("(?P<a>\\w+)@(?P<b>\\w+)\\.(?P<c>\\w+)", 0);
    RE re_email_pos    = jit_compile("(\\w+)@(\\w+)\\.(\\w+)", 0);
    RE re_neg_la       = jit_compile("\\w+(?!@)",   0);
    RE re_neg_lb       = jit_compile("(?<!\\d)\\w+", 0);
    RE re_password     = jit_compile("(?=.*[a-z])(?=.*[A-Z])(?=.*\\d).{8,}", 0);
    RE re_alt_4        = jit_compile("alpha|beta|gamma|delta", 0);
    RE re_alt_16       = jit_compile(
        "alpha|beta|gamma|delta|epsilon|zeta|eta|theta"
        "|iota|kappa|lambda|mu|nu|xi|omicron|pi", 0);
    RE re_digits       = jit_compile("\\d+",      0);
    RE re_dot          = jit_compile(".",          0);
    RE re_replace_backref = jit_compile("(\\w+) (\\w+)", 0);
    RE re_patho_opt16  = jit_compile(
        "a?a?a?a?a?a?a?a?a?a?a?a?a?a?a?a?aaaaaaaaaaaaaaaa", 0);
    RE re_dotstar_anch = jit_compile("^.*x$",     0);
    RE re_dotstar_miss = jit_compile(".*x",        0);
    RE re_triple_bref  = jit_compile("(\\w+)\\s\\1\\s\\1", 0);
    RE re_nested_q     = jit_compile("([a-z]+[0-9]+)+x", 0);
    RE re_url          = jit_compile("(https?|ftp)://([^/\\s]+)(/[^\\s]*)?", 0);
    RE re_phone        = jit_compile("\\(?\\d{3}\\)?[-.\\s]?\\d{3}[-.\\s]?\\d{4}", 0);
    RE re_hex          = jit_compile("#[0-9a-fA-F]{6}", 0);
    RE re_semver       = jit_compile(
        "(\\d+)\\.(\\d+)\\.(\\d+)(?:-(\\w+(?:\\.\\w+)*))?", 0);
    RE re_ws           = jit_compile("\\s+",      0);
    RE re_log_error    = jit_compile("\\[ERROR\\].*", 0);
    RE re_inline_ig    = jit_compile("(?i)hello world", 0);
    RE re_inline_ml    = jit_compile("(?m)^error.*$",   0);
    RE re_dfa          = jit_compile("[a-z]+\\d+[a-z]+", 0);
    RE re_pike         = jit_compile("([a-z]+)(\\d+)([a-z]+)", 0);
    RE re_backref_e    = jit_compile("([a-z]+)\\d+\\1", 0);

    /* -----------------------------------------------------------------------
     * 1. Throughput scaling
     * --------------------------------------------------------------------- */
    emit("throughput_literal_100B",   bench_search(&re_needle, in_100B,  strlen(in_100B),  ITERS_NORMAL));
    emit("throughput_literal_10KB",   bench_search(&re_needle, in_10KB,  strlen(in_10KB),  ITERS_NORMAL));
    emit("throughput_literal_100KB",  bench_search(&re_needle, in_100KB, strlen(in_100KB), ITERS_MEDIUM));
    emit("throughput_literal_1MB",    bench_search(&re_needle, in_1MB,   strlen(in_1MB),   ITERS_LIGHT));
    emit("throughput_class_10KB",     bench_search(&re_xyz,    in_class_10KB, strlen(in_class_10KB), ITERS_NORMAL));
    emit("throughput_nomatch_100KB",  bench_search(&re_zzz_6,  in_nomatch_100KB, strlen(in_nomatch_100KB), ITERS_MEDIUM));

    /* -----------------------------------------------------------------------
     * 2. Anchors
     * --------------------------------------------------------------------- */
    emit("anchor_bol",                bench_search(&re_hello,        in_hello_world, strlen(in_hello_world), ITERS_NORMAL));
    emit("anchor_eol",                bench_search(&re_world_dollar,  in_hello_world, strlen(in_hello_world), ITERS_NORMAL));
    emit("anchor_word_boundary",      bench_search(&re_word_b,        in_say_hello,   strlen(in_say_hello),   ITERS_NORMAL));
    emit("anchor_word_boundary_miss", bench_search(&re_orld_b,        in_say_hello,   strlen(in_say_hello),   ITERS_NORMAL));
    emit("anchor_bol_miss_10KB",      bench_search(&re_zzz_bol,       in_aaaa_10KB,   strlen(in_aaaa_10KB),   ITERS_NORMAL));

    /* -----------------------------------------------------------------------
     * 3. Multiline / DOTALL
     * --------------------------------------------------------------------- */
    emit("dotall_multiline_body",           bench_match(&re_dotall,   in_dotall_body, strlen(in_dotall_body), ITERS_NORMAL));

    /* -----------------------------------------------------------------------
     * 4. Named groups
     * --------------------------------------------------------------------- */
    emit("named_group_date",       bench_match(&re_date,        in_date,  strlen(in_date),  ITERS_NORMAL));
    emit("named_group_email",      bench_match(&re_email_named, in_email, strlen(in_email), ITERS_NORMAL));
    emit("positional_group_email", bench_match(&re_email_pos,   in_email, strlen(in_email), ITERS_NORMAL));

    /* -----------------------------------------------------------------------
     * 5. Negative lookaround
     * --------------------------------------------------------------------- */
    emit("neg_lookahead",                 bench_search(&re_neg_la,  in_hello_world, strlen(in_hello_world), ITERS_NORMAL));
    emit("neg_lookbehind",                bench_search(&re_neg_lb,  in_hello_world, strlen(in_hello_world), ITERS_NORMAL));
    emit("password_validation_lookahead", bench_match(&re_password, in_password,    strlen(in_password),    ITERS_NORMAL));

    /* -----------------------------------------------------------------------
     * 6. Alternation scaling
     * --------------------------------------------------------------------- */
    emit("alternation_4",       bench_match(&re_alt_4,  in_delta, strlen(in_delta), ITERS_NORMAL));
    emit("alternation_16",      bench_match(&re_alt_16, in_pi,    strlen(in_pi),    ITERS_NORMAL));
    emit("alternation_16_miss", bench_match(&re_alt_16, in_sigma, strlen(in_sigma), ITERS_NORMAL));

    /* -----------------------------------------------------------------------
     * 7. (Removed findall scaling)
     * --------------------------------------------------------------------- */

    /* -----------------------------------------------------------------------
     * 8. Replace scaling
     * --------------------------------------------------------------------- */
    emit("replace_50_matches",   bench_replace(&re_digits, in_replace_50, strlen(in_replace_50),
                                               "NUM", 3, ITERS_MEDIUM));
    emit("replace_named_backref",bench_replace(&re_replace_backref, in_john_doe, strlen(in_john_doe),
                                               "$2, $1", 6, ITERS_NORMAL));

    /* -----------------------------------------------------------------------
     * 9. (Removed split scaling)
     * --------------------------------------------------------------------- */

    /* -----------------------------------------------------------------------
     * 10. Pathological
     * --------------------------------------------------------------------- */
    emit("pathological_optional_16",          bench_match(&re_patho_opt16, in_patho_16,    strlen(in_patho_16),    ITERS_LIGHT));
    emit("pathological_dotstar_anchored_5K",  bench_match(&re_dotstar_anch, in_dotstar_5K, strlen(in_dotstar_5K),  ITERS_LIGHT));
    emit("pathological_dotstar_miss_5K",      bench_match(&re_dotstar_miss, in_dotstar_miss_5K, strlen(in_dotstar_miss_5K), ITERS_LIGHT));
    emit("pathological_triple_backref",       bench_match(&re_triple_bref, in_patho_backref, strlen(in_patho_backref), ITERS_NORMAL));
    emit("pathological_nested_quantifier_miss",bench_match(&re_nested_q,   in_patho_nested,  strlen(in_patho_nested),  ITERS_LIGHT));

    /* -----------------------------------------------------------------------
     * 11. Real-world patterns
     * --------------------------------------------------------------------- */
    emit("realworld_url_parse",             bench_match(&re_url,       in_url,       strlen(in_url),       ITERS_NORMAL));
    emit("realworld_phone",                 bench_match(&re_phone,     in_phone,     strlen(in_phone),     ITERS_NORMAL));
    emit("realworld_hex_color",             bench_match(&re_hex,       in_hex,       strlen(in_hex),       ITERS_NORMAL));
    emit("realworld_semver",                bench_match(&re_semver,    in_semver,    strlen(in_semver),     ITERS_NORMAL));
    emit("realworld_ws_normalize",          bench_replace(&re_ws,      in_ws_raw,    strlen(in_ws_raw),
                                                          " ", 1, ITERS_NORMAL));
    emit("realworld_log_search_1000_lines", bench_search(&re_log_error, in_log_lines, strlen(in_log_lines), ITERS_MEDIUM));

    /* -----------------------------------------------------------------------
     * 12. Inline flags
     * --------------------------------------------------------------------- */
    emit("inline_ignorecase",     bench_match(&re_inline_ig, in_inline_ig, strlen(in_inline_ig), ITERS_NORMAL));
    emit("inline_multiline_search",bench_search(&re_inline_ml,in_inline_ml, strlen(in_inline_ml), ITERS_NORMAL));

    /* -----------------------------------------------------------------------
     * 13. Engine comparison
     * --------------------------------------------------------------------- */
    emit("engine_dfa_no_capture",       bench_match(&re_dfa,      in_dfa,      strlen(in_dfa),      ITERS_NORMAL));
    emit("engine_pike_with_capture",    bench_match(&re_pike,     in_dfa,      strlen(in_dfa),      ITERS_NORMAL));
    emit("engine_backtrack_with_backref",bench_match(&re_backref_e,in_backref, strlen(in_backref),  ITERS_NORMAL));

    /* -----------------------------------------------------------------------
     * Cleanup
     * --------------------------------------------------------------------- */
    free(in_100B); free(in_10KB); free(in_100KB); free(in_1MB);
    free(in_class_10KB); free(in_nomatch_100KB); free(in_100_lines);
    free(in_log_lines); free(in_dots_500);
    free(in_replace_50); free(in_dotstar_5K);
    free(in_dotstar_miss_5K); free(in_aaaa_10KB);

    return 0;
}
