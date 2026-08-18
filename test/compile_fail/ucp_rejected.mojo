# (*UCP) must be rejected: accepting it as a UTF8 alias silently keeps
# \d \w \s \b ASCII, the opposite of PCRE's UCP contract.
# EXPECT-ERROR: (*UCP) is not supported
from emberregex import Regex


def main():
    var re = Regex["(*UCP)\\w+"]()
    _ = re.search("abc")
