"""Tests for real-world regex patterns."""

from emberregex import Regex
from std.testing import assert_true, assert_false, TestSuite


def test_email_pattern() raises:
    var re = Regex["[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"]()
    assert_true(re.match("user@example.com").matched)
    assert_false(re.match("not-an-email").matched)


def test_ip_address() raises:
    var re = Regex["\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}"]()
    assert_true(re.match("192.168.1.1").matched)
    assert_false(re.match("abc.def.ghi.jkl").matched)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
