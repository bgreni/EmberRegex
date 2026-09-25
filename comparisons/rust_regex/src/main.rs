//! Rust `regex` side of bench/bench_compare_rust.py.
//!
//!   bench_rust_regex bench            mirror of bench/bench.mojo
//!   bench_rust_regex rebar <manifest> rebar subset (manifest written by
//!                                     bench_compare_rust.py)
//!
//! Output: one `name\tus_per_op\tfingerprint` line per benchmark. `bench`
//! reports the mean (what Mojo's `Bench` reports as "met"); `rebar` the
//! median of per-op samples, exactly as comparisons/bench_rebar.mojo does.
//!
//! Every regex is a `regex::bytes::Regex` built with `unicode(false)` unless
//! a rebar benchmark asks for Unicode: emberregex's `\w`, `\d`, `\s`, `\b`
//! and `(?i)` are ASCII by default, so this is the same language.

use regex::bytes::{CaptureLocations, Regex, RegexBuilder};
use std::hint::black_box;
use std::time::{Duration, Instant};

fn build(pat: &str, unicode: bool, casei: bool) -> Result<Regex, regex::Error> {
    RegexBuilder::new(pat)
        .unicode(unicode)
        .case_insensitive(casei)
        .size_limit(100 << 20)
        .build()
}

// ---------------------------------------------------------------------------
// bench.mojo mirror
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum Verb {
    /// emberregex `match()` is a full match: compiled as `\A(?:pat)\z`.
    Match,
    Search,
    Findall,
    /// Replacement in Rust `$N` syntax.
    Replace(&'static str),
    Split,
}

struct Case {
    name: &'static str,
    pat: &'static str,
    verb: Verb,
    input: Vec<u8>,
}

fn case(name: &'static str, pat: &'static str, verb: Verb, input: impl Into<Vec<u8>>) -> Case {
    Case { name, pat, verb, input: input.into() }
}

fn rep(s: &str, n: usize) -> String {
    s.repeat(n)
}

fn repeat_with_sep(word: &str, sep: &str, n: usize) -> String {
    vec![word; n].join(sep)
}

fn make_lines(n: usize) -> String {
    (0..n).map(|i| format!("line {i} some text here")).collect::<Vec<_>>().join("\n")
}

fn lcg_prose(n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n);
    let mut x: u64 = 12345;
    while out.len() < n {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let r = ((x >> 33) % 40) as u8;
        out.push(match r {
            0..=25 => b'a' + r,
            26..=29 => b' ',
            30..=32 => b'A' + r - 30,
            33..=35 => b'0' + r - 33,
            36 => b',',
            37 => b'.',
            38 => b'\n',
            _ => b'_',
        });
    }
    out
}

fn filler_plus(filler: &str, n: usize, tail: &str) -> String {
    filler.repeat(n) + tail
}

fn log_lines() -> String {
    (0..1000)
        .map(|i| {
            if i == 750 {
                "2026-03-21 14:30:05 [ERROR] Something broke".to_string()
            } else {
                format!("2026-03-21 14:30:05 [INFO] All good line {i}")
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Width of emberregex's SIMD-literal benches: `simd_width_of[DType.uint8]()`.
const SIMD_W: usize = if cfg!(target_arch = "aarch64") { 16 } else { 32 };

fn bench_cases() -> Vec<Case> {
    use Verb::*;
    let simd_lit: &'static str = Box::leak("a".repeat(SIMD_W).into_boxed_str());
    let alt16 = "alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|omicron|pi";
    let email = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}";
    let sheng64 = "cat|cow|dog|doe|bat|bit|fig|fin|gum|gas|hen|hex|jam|jab|kit|keg\
                   |lap|lab|mop|mob|net|nap|owl|oak|pin|pit|rat|rib|sun|sit|tap|[0-9]{3}";
    let html = "<html><head><title>Test</title></head><body><div \
                class=\"x\"><p>Hello</p><a href=\"#\">Link</a></div></body></html>";
    let e_filler = "the everyday sentence keeps several e letters here ";
    let sparse_filler = rep("lorem ipsum 42 dolor sit amet ", 43);
    let long_tokens = vec!["abcdefghij0123456789abcdefghij0123456789"; 40].join(" ") + " user@example.com";
    vec![
        case("simd_literal_match", simd_lit, Match, simd_lit),
        case("simd_literal_search", simd_lit, Search, make_lines(100) + "\n" + simd_lit),
        case("static_dfa_literal_match", "abcdefghij", Match, "abcdefghij"),
        case("static_dfa_char_class_26", "[a-z]+", Match, "abcdefghijklmnopqrstuvwxyz"),
        case("static_dfa_alternation_8", "cat|dog|bird|fish|frog|snake|mouse|horse", Match, "horse"),
        case("static_dfa_quantifier_bounded", "[a-z]{5,10}[0-9]{3,5}", Match, "abcdefg1234"),
        case("static_capture_email", r"(\w+)@(\w+)\.(\w+)", Match, "user@example.com"),
        case("static_nested_groups", r"((\w+)(-(\w+))*)@(\w+)", Match, "foo-bar-baz@host"),
        case("capture_search_miss_100KB", r"(\w+)@(\w+)\.com", Search, rep("user@example.org x.com ", 100 * 1024 / 23)),
        case("capture_findall_sparse_64KB", r"(\d+)-(\d+)", Findall, repeat_with_sep("123-4567", &sparse_filler, 50) + &sparse_filler),
        case("reverse_suffix_search_64KB", r"\w+\.txt", Search, rep("log.tx err.txx data.ttx x.t wo.rd ", 64 * 1024 / 34)),
        case("reverse_inner_search_64KB", r"[a-z]+://[^ ]+", Search, rep("svc: api / level: info /x msg: ok :/ trace: nine ", 64 * 1024 / 49)),
        case("onepass_match_kv", r"(?:([a-z])|(\d)|[=;&])+", Match, "host=db01&port=5432&user=admin&retry=55&"),
        case("onepass_findall_2KB", r"(?:(x)|(y)|z)+", Findall, rep("xyz", 340) + " " + &rep("zyx", 340)),
        case("static_greedy_tag", "<(.+)>", Match, "<a>hello</a>"),
        case("static_lazy_tag", "<(.+?)>", Match, "<a>hello</a>"),
        case("static_backref", r"(\w+)\s\1", Match, "hello hello"),
        case("static_html_tag", r"<([a-z]+)>[^<]*</\1>", Match, "<div>content</div>"),
        case("static_search_short_11B", "world", Search, "hello world"),
        case("static_search_medium_1KB", "needle", Search, rep("a", 500) + "needle" + &rep("b", 500)),
        case("static_search_long_20KB", "needle", Search, rep("a", 10000) + "needle" + &rep("b", 10000)),
        case("static_search_miss_10KB", "zzzzz", Search, rep("a", 10000)),
        case("static_search_date_capture", r"(\d{4})-(\d{2})-(\d{2})", Search, rep("x", 200) + "2026-03-21" + &rep("y", 200)),
        case("static_findall_numbers", "[0-9]+", Findall, "abc 12 def 345 ghi 6789 jkl 0 mno 42 pqr 100"),
        case("static_replace_numbers", "[0-9]+", Replace("NUM"), "abc 12 def 345 ghi 6789 jkl 0 mno 42 pqr 100"),
        case("static_replace_with_backref", r"(\w+)=(\w+)", Replace("${2}=${1}"), "a=1 b=2 c=3 d=4 e=5"),
        case("static_split_delimiters", r"[,;\s]+", Split, "one, two; three  four,five;six seven , eight"),
        case("static_explicit_case_range", "[a-zA-Z]+", Match, "HeLLoWoRLdFoOBaR"),
        case("static_ignorecase", "(?i)[a-z]+", Match, "HeLLoWoRLdFoOBaR"),
        case("static_lookahead_positive", r"\w+(?=@)", Search, "user@host"),
        case("static_lookbehind_positive", r"(?<=@)\w+", Search, "user@host"),
        case("static_pathological_optional_8", "a?a?a?a?a?a?a?a?aaaaaaaa", Match, "aaaaaaaa"),
        case("static_dotstar_1K", ".*x", Match, rep("a", 1000) + "x"),
        case("static_dotstar_search_1K", ".*x", Search, rep("a", 1000) + "x" + "bbb"),
        case("static_bol_alternation_miss_10KB", "^(?:ab|cd)", Search, rep("x", 10_000)),
        case("static_replace_alternation", "cat|dog", Replace("pet"), repeat_with_sep("a cat and a dog here", " ", 30)),
        case("static_ignorecase_search_2KB", "(?i)error", Search, filler_plus(e_filler, 40, "an ERRor appeared")),
        case("static_teddy_prefix_search_2KB", r"(?:GET|POST|PUT) /\w+", Search,
             filler_plus("ts=12 host=web01 status=200 bytes=512 ref=none agent=x ", 35, "POST /submit")),
        case("static_url_search_2KB", "[a-z]+://[a-z.]+", Search,
             filler_plus("svc: api level: info msg: ok elapsed: three trace: nine ", 35, "see http://example.com now")),
        case("static_ignorecase_alternation_2KB", "(?i)(?:error|warning|fatal)", Search, filler_plus(e_filler, 40, "then a FATAL crash")),
        case("pathological_pike_search_miss_600B", "(a+)+b", Search, rep("a", 600)),
        case("static_nested_quantifier", "([a-z]+[0-9]+)+x", Match, "abc123def456ghi789x"),
        case("lf_dfa_lazy_findall_64KB", "<.*?>", Findall, repeat_with_sep("<tag>", " text ", 64 * 1024 / 14)),
        case("word_boundary_findall_64KB", r"\b\w+\b", Findall, lcg_prose(64 * 1024)),
        case("match_single_byte_run_20KB", "a+e|x", Match, rep("a", 20480) + "e"),
        case("lf_dfa_class_run_search_20KB", "[a-z]+x", Search, rep("a", 20 * 1024) + "x"),
        case("static_realworld_email", email, Match, "john.doe+test@example.co.uk"),
        case("static_email_search_2KB", email, Search, make_lines(80) + " contact us at first.last@example.com today"),
        case("static_email_search_long_tokens", email, Search, long_tokens),
        case("static_realworld_ipv4", r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", Match, "192.168.1.100"),
        case("static_realworld_log_parse", r"(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}) \[(\w+)\] (.*)", Match,
             "2026-03-21 14:30:05 [ERROR] Connection timeout after 30s"),
        case("static_realworld_csv_fields", "[^,]+", Findall, "field1,field2,field3,field4,field5,field6,field7,field8"),
        case("throughput_literal_100B", "needle", Search, rep("a", 94) + "needle"),
        case("throughput_literal_10KB", "needle", Search, rep("a", 10000) + "needle"),
        case("throughput_literal_100KB", "needle", Search, rep("a", 100000) + "needle"),
        case("throughput_literal_1MB", "needle", Search, rep("a", 1000000) + "needle"),
        case("throughput_class_10KB", "[xyz]+", Search, rep("a", 9990) + "xyzxyzxyz"),
        case("throughput_nomatch_100KB", "zzzzzz", Search, rep("a", 100000)),
        case("anchor_bol", "^hello", Search, "hello world"),
        case("anchor_eol", "world$", Search, "hello world"),
        case("anchor_word_boundary", r"\bworld\b", Search, "say hello world today"),
        case("anchor_word_boundary_miss", r"\borld\b", Search, "say hello world today"),
        case("anchor_bol_miss_10KB", "^zzz", Search, rep("a", 10000)),
        case("multiline_bol_findall_100_lines", r"(?m)^\w+", Findall, make_lines(100)),
        case("multiline_eol_findall_100_lines", r"(?m)\w+$", Findall, make_lines(100)),
        case("dotall_multiline_body", "(?s)<body>.*</body>", Match, "<body>\nline1\nline2\nline3\n</body>"),
        case("named_group_date", r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})", Match, "2026-03-21"),
        case("named_group_email", r"(?P<a>\w+)@(?P<b>\w+)\.(?P<c>\w+)", Match, "user@example.com"),
        case("positional_group_email", r"(\w+)@(\w+)\.(\w+)", Match, "user@example.com"),
        case("neg_lookahead", r"\w+(?!@)", Search, "hello world"),
        case("neg_lookbehind", r"(?<!\d)\w+", Search, "hello world"),
        case("password_validation_lookahead", r"(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}", Match, "MyP4ssw0rd"),
        case("alternation_4", "alpha|beta|gamma|delta", Match, "delta"),
        case("alternation_4_search_2KB", "alpha|beta|gamma|delta", Search, make_lines(80) + " delta"),
        case("alternation_16", alt16, Match, "pi"),
        case("sheng64_alt_32_search_2KB", sheng64, Search, make_lines(80) + " tap"),
        case("alternation_16_miss", alt16, Match, "sigma"),
        case("findall_3_matches", r"\d+", Findall, "a1b2c3"),
        case("findall_100_matches", r"\d+", Findall, repeat_with_sep("42", " word ", 100)),
        case("findall_500_dot_matches", ".", Findall, rep("a", 500)),
        case("replace_50_matches", r"\d+", Replace("NUM"), repeat_with_sep("42", " text ", 50)),
        case("replace_named_backref", r"(\w+) (\w+)", Replace("${2}, ${1}"), "John Doe"),
        case("split_100_parts", "[,;|]+", Split, repeat_with_sep("word", ",", 100)),
        case("pathological_optional_16", "a?a?a?a?a?a?a?a?a?a?a?a?a?a?a?a?aaaaaaaaaaaaaaaa", Match, rep("a", 16)),
        case("pathological_dotstar_anchored_5K", "^.*x$", Match, rep("a", 5000) + "x"),
        case("pathological_dotstar_miss_5K", ".*x", Match, rep("a", 5000)),
        case("pathological_triple_backref", r"(\w+)\s\1\s\1", Match, "hello hello hello"),
        case("pathological_nested_quantifier_miss", "([a-z]+[0-9]+)+x", Match, rep("a1", 800) + "ax"),
        case("memo_ambiguous_plus_miss_1500", "(a|aa)+b", Search, rep("a", 1500) + "c"),
        case("memo_ambiguous_plus_in_span_1500", "(a|aa)+c|a+b", Search, rep("a", 1500) + "b"),
        case("realworld_url_parse", r"(https?|ftp)://([^/\s]+)(/[^\s]*)?", Match, "https://www.example.com/path/to/page?q=1&r=2"),
        case("realworld_phone", r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", Match, "(555) 123-4567"),
        case("realworld_hex_color", "#[0-9a-fA-F]{6}", Match, "#1a2B3c"),
        case("realworld_semver", r"(\d+)\.(\d+)\.(\d+)(?:-(\w+(?:\.\w+)*))?", Match, "12.34.56-beta.1"),
        // Lookahead: Rust regex rejects these two, so no haystack is built.
        case("counted_repeat_search_2KB", r"(?=[a-z])([a-z]{3,7})\d", Search, ""),
        case("counted_repeat_giveback_2KB", r"(?=[a-z])([a-z]{3,7})[a-z]x", Search, ""),
        case("realworld_key_value_findall", r"(\w+)=(\S+)", Findall, "host=localhost port=5432 db=mydb user=admin timeout=30"),
        case("realworld_html_tag_findall", r"<(\w+)[^>]*>", Findall, html),
        case("realworld_ws_normalize", r"\s+", Replace(" "), "hello   world\t\tfoo  bar\n\nbaz   qux"),
        case("realworld_log_search_1000_lines", r"\[ERROR\].*", Search, log_lines()),
        case("inline_ignorecase", "(?i)hello world", Match, "HeLLo WoRLd"),
        case("inline_multiline_search", "(?m)^error.*$", Search, "info: ok\nwarn: hmm\nerror: bad\ninfo: ok"),
        case("engine_dfa_no_capture", r"[a-z]+\d+[a-z]+", Match, "abc123def"),
        case("engine_pike_with_capture", r"([a-z]+)(\d+)([a-z]+)", Match, "abc123def"),
        case("engine_backtrack_with_backref", r"([a-z]+)\d+\1", Match, "abc123abc"),
    ]
}

struct Prepared {
    re: Regex,
    locs: CaptureLocations,
    groups: bool,
    verb: Verb,
}

impl Prepared {
    fn new(c: &Case) -> Result<Prepared, regex::Error> {
        let pat = match c.verb {
            Verb::Match => format!(r"\A(?:{})\z", c.pat),
            _ => c.pat.to_string(),
        };
        let re = build(&pat, false, false)?;
        let locs = re.capture_locations();
        let groups = re.captures_len() > 1;
        Ok(Prepared { re, locs, groups, verb: c.verb })
    }

    /// One operation, shaped like emberregex's verb: a span (with the
    /// slots filled when there are groups), or an owned result list.
    fn run(&mut self, hay: &[u8]) -> Fp {
        match self.verb {
            Verb::Match | Verb::Search => {
                let m = if self.groups {
                    self.re.captures_read(&mut self.locs, hay)
                } else {
                    self.re.find(hay)
                };
                Fp::Span(m.map(|m| (m.start(), m.end())))
            }
            Verb::Findall => {
                // Python/emberregex findall: group 1's text when the pattern
                // has groups, else the whole match. (No bench pattern can
                // match empty, so plain `end` is the next start.)
                let out: Vec<Vec<u8>> = if self.groups {
                    let mut out = Vec::new();
                    let mut at = 0;
                    while let Some(m) = self.re.captures_read_at(&mut self.locs, hay, at) {
                        out.push(self.locs.get(1).map_or(Vec::new(), |(s, e)| hay[s..e].to_vec()));
                        at = m.end();
                    }
                    out
                } else {
                    self.re.find_iter(hay).map(|m| m.as_bytes().to_vec()).collect()
                };
                Fp::N(black_box(out).len())
            }
            Verb::Replace(repl) => Fp::Len(black_box(self.re.replace_all(hay, repl.as_bytes())).len()),
            Verb::Split => {
                let parts: Vec<Vec<u8>> = self.re.split(hay).map(|s| s.to_vec()).collect();
                Fp::N(black_box(parts).len())
            }
        }
    }
}

/// What one op produced, printed so a run can be checked against emberregex.
enum Fp {
    Span(Option<(usize, usize)>),
    N(usize),
    Len(usize),
}

impl std::fmt::Display for Fp {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Fp::Span(Some((s, e))) => write!(f, "True {s} {e}"),
            Fp::Span(None) => write!(f, "False"),
            Fp::N(n) => write!(f, "n={n}"),
            Fp::Len(n) => write!(f, "len={n}"),
        }
    }
}

/// Mean per-op time over ~0.5 s after a 0.1 s warmup, in batches of 100.
fn time_mean(mut f: impl FnMut()) -> f64 {
    let warm = Instant::now();
    while warm.elapsed() < Duration::from_millis(100) {
        for _ in 0..100 {
            f();
        }
    }
    let start = Instant::now();
    let mut n = 0u64;
    while start.elapsed() < Duration::from_millis(500) {
        for _ in 0..100 {
            f();
        }
        n += 100;
    }
    start.elapsed().as_secs_f64() * 1e6 / n as f64
}

fn run_bench() {
    for c in bench_cases() {
        let mut p = match Prepared::new(&c) {
            Ok(p) => p,
            Err(e) => {
                let why = e.to_string().lines().last().unwrap_or("").trim().to_string();
                println!("{}\tNA\tunsupported: {}", c.name, why);
                continue;
            }
        };
        let fp = p.run(&c.input);
        let us = time_mean(|| {
            black_box(p.run(black_box(&c.input)));
        });
        println!("{}\t{:.3}\tin={} {}", c.name, us, c.input.len(), fp);
    }
}

// ---------------------------------------------------------------------------
// rebar subset
// ---------------------------------------------------------------------------

/// bstr's `lines()`: split on `\n`, drop a trailing `\r`, no empty last line.
fn lines(hay: &[u8]) -> Vec<&[u8]> {
    let mut out: Vec<&[u8]> = hay.split(|&b| b == b'\n').collect();
    if hay.is_empty() || hay.ends_with(b"\n") {
        out.pop();
    }
    for l in out.iter_mut() {
        if let Some(s) = l.strip_suffix(b"\r") {
            *l = s;
        }
    }
    out
}

/// Matching groups (group 0 included) over every match, like rebar's
/// count-captures loop.
fn count_groups(re: &Regex, locs: &mut CaptureLocations, hay: &[u8]) -> usize {
    let mut count = 0;
    let mut at = 0;
    while at <= hay.len() {
        let Some(m) = re.captures_read_at(locs, hay, at) else { break };
        count += (0..locs.len()).filter(|&i| locs.get(i).is_some()).count();
        at = if m.end() > m.start() { m.end() } else { m.end() + 1 };
    }
    count
}

/// Median per-op time: a 0.25 s warmup sizes a batch of ~1 ms, then batches
/// run for 1 s (>= 3 of them). Batching keeps the clock's resolution out of
/// the numbers (Mojo's `perf_counter_ns` is 1 us on macOS).
fn time_median(mut f: impl FnMut() -> usize) -> (f64, usize) {
    let count = f();
    let warm = Instant::now();
    let mut n = 0u32;
    while n == 0 || warm.elapsed() < Duration::from_millis(250) {
        black_box(f());
        n += 1;
    }
    let per_op = warm.elapsed().as_secs_f64() / n as f64;
    let batch = ((1e-3 / per_op) as usize).max(1);
    let mut samples = Vec::new();
    let start = Instant::now();
    while samples.len() < 3 || start.elapsed() < Duration::from_secs(1) {
        let t = Instant::now();
        for _ in 0..batch {
            black_box(f());
        }
        samples.push(t.elapsed().as_secs_f64() / batch as f64);
    }
    samples.sort_by(f64::total_cmp);
    (samples[samples.len() / 2] * 1e6, count)
}

fn run_rebar(manifest: &str) {
    let text = std::fs::read_to_string(manifest).expect("read manifest");
    for row in text.lines() {
        let f: Vec<&str> = row.splitn(7, '\t').collect();
        let [name, model, unicode, casei, _expected, hay_path, pat] = f[..] else {
            panic!("bad manifest row: {row}")
        };
        let re = match build(pat, unicode == "1", casei == "1") {
            Ok(re) => re,
            Err(e) => {
                println!("{name}\tNA\tunsupported: {}", e.to_string().lines().last().unwrap_or(""));
                continue;
            }
        };
        let hay = std::fs::read(hay_path).expect("read haystack");
        let hay_lines = lines(&hay);
        let mut locs = re.capture_locations();
        let (us, count) = match model {
            "count" => time_median(|| re.find_iter(&hay).count()),
            "count-spans" => time_median(|| re.find_iter(&hay).map(|m| m.len()).sum()),
            "count-captures" => time_median(|| count_groups(&re, &mut locs, &hay)),
            "grep" => time_median(|| hay_lines.iter().filter(|l| re.is_match(l)).count()),
            "grep-captures" => time_median(|| hay_lines.iter().map(|l| count_groups(&re, &mut locs, l)).sum()),
            _ => panic!("unknown model {model}"),
        };
        println!("{name}\t{us:.3}\t{count}");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("bench") => run_bench(),
        Some("rebar") => run_rebar(args.get(2).expect("rebar <manifest>")),
        // Reference spans for engine checks: match count and a hash of
        // every (start, end), over `limit` bytes of `file` (0 = all).
        Some("spans") => {
            let pat = args.get(2).expect("spans <pattern> <file> <unicode 0|1> <limit>");
            let hay = std::fs::read(&args[3]).expect("read haystack");
            let limit: usize = args[5].parse().unwrap();
            let hay = if limit > 0 && limit < hay.len() { &hay[..limit] } else { &hay[..] };
            let re = build(pat, args[4] == "1", false).expect("pattern");
            let (mut n, mut h) = (0u64, 0u64);
            for m in re.find_iter(hay) {
                n += 1;
                h = (h.wrapping_mul(1000003) + m.start() as u64 * 31 + m.end() as u64) % (1 << 61);
            }
            println!("{n} {h}");
        }
        _ => eprintln!("usage: bench_rust_regex bench | rebar <manifest>"),
    }
}
