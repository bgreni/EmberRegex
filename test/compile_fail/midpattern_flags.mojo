# A bare inline-flag group after pattern content must be rejected: the old
# behavior applied it retroactively to the WHOLE pattern, which matches no
# other engine (Python/JS error; PCRE2/Perl/Ruby/Rust apply rightward only).
# We take the Python side: compile error.
# EXPECT-ERROR: global flags not at the start
from emberregex import Regex


def main():
    var re = Regex["a(?i)b"]()
    _ = re.search("ab")
