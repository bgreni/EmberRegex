"""EmberRegex - A high-performance regex library for Mojo."""

from .compile import compile, try_compile, CompiledRegex
from .static import StaticRegex
from .result import MatchResult
from .errors import RegexError
from .flags import RegexFlags
