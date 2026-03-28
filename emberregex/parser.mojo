"""Recursive descent regex parser.

Grammar (Milestone 1):
    regex      = alternation
    alternation = concat ('|' concat)*
    concat     = quantified+
    quantified = atom ('*' | '+' | '?')?
    atom       = CHAR | '.' | '[' charset ']' | '(' regex ')' | '\\' ESCAPE
"""

from .constants import (
    CHAR_A_LOWER,
    CHAR_A_UPPER,
    CHAR_BACKSLASH,
    CHAR_BANG,
    CHAR_B_LOWER,
    CHAR_B_UPPER,
    CHAR_CARET,
    CHAR_COLON,
    CHAR_COMMA,
    CHAR_CR,
    CHAR_DOLLAR,
    CHAR_DOT,
    CHAR_D_LOWER,
    CHAR_D_UPPER,
    CHAR_EQUALS,
    CHAR_GREATER_THAN,
    CHAR_I_LOWER,
    CHAR_LBRACE,
    CHAR_LBRACKET,
    CHAR_LESS_THAN,
    CHAR_LPAREN,
    CHAR_MINUS,
    CHAR_M_LOWER,
    CHAR_NEWLINE,
    CHAR_NINE,
    CHAR_ONE,
    CHAR_PIPE,
    CHAR_PLUS,
    CHAR_P_UPPER,
    CHAR_QUESTION,
    CHAR_RBRACE,
    CHAR_RBRACKET,
    CHAR_RPAREN,
    CHAR_S,
    CHAR_SPACE,
    CHAR_STAR,
    CHAR_S_LOWER,
    CHAR_TAB,
    CHAR_UNDERSCORE,
    CHAR_W_LOWER,
    CHAR_W_UPPER,
    CHAR_ZERO,
    CHAR_Z_LOWER,
    CHAR_Z_UPPER,
    CHAR_n,
    CHAR_r,
    CHAR_t,
)
from .ast import AST, ASTNode, ASTNodeKind, AnchorKind
from .charset import CharSet, CharRange
from .errors import RegexError
from .flags import RegexFlags


struct Parser[origin: Origin](Movable):
    """Recursive descent parser for regex patterns."""

    var pattern: Span[Byte, Self.origin]
    var pos: Int
    var ast: AST
    var inline_flags: RegexFlags  # collected from (?i), (?m), (?s) in the pattern

    def __init__(out self, pattern: Span[Byte, Self.origin]):
        self.pattern = pattern
        self.pos = 0
        self.ast = AST()
        self.inline_flags = RegexFlags()

    def parse(mut self) raises -> AST:
        """Parse the pattern and return the AST."""
        var root = self._parse_alternation()
        self.ast.root = root
        if self.pos < len(self.pattern):
            raise Error(
                String.write(RegexError("Unexpected character", self.pos))
            )
        # Build bitmaps for all charsets
        for i in range(len(self.ast.charsets)):
            self.ast.charsets[i].build_bitmap()
        # Store inline flags on the AST so callers can access them
        self.ast.flags = self.inline_flags
        var result = self.ast^
        self.ast = AST()
        return result^

    def _peek(self) -> Byte:
        """Look at the current character without consuming it."""
        if self.pos >= len(self.pattern):
            return Byte(0)
        return self.pattern.unsafe_get(self.pos)

    def _advance(mut self) -> Byte:
        """Consume and return the current character."""
        var ch = self._peek()
        self.pos += 1
        return ch

    def _at_end(self) -> Bool:
        return self.pos >= len(self.pattern)

    def _expect(mut self, ch: Byte) raises:
        if self._at_end() or self._peek() != ch:
            raise Error(
                String.write(
                    RegexError(
                        "Expected '" + chr(Int(ch)) + "'",
                        self.pos,
                    )
                )
            )
        self.pos += 1

    # --- Grammar productions ---

    def _parse_alternation(mut self) raises -> Int:
        """alternation = concat ('|' concat)*"""
        var first = self._parse_concat()
        if self._at_end() or self._peek() != CHAR_PIPE:
            return first

        var alternatives = List[Int]()
        alternatives.append(first)
        while not self._at_end() and self._peek() == CHAR_PIPE:
            self.pos += 1  # consume '|'
            alternatives.append(self._parse_concat())

        var node = ASTNode.alternation(alternatives^)
        return self.ast.add_node(node^)

    def _parse_concat(mut self) raises -> Int:
        """concat = quantified+"""
        var parts = List[Int]()
        while (
            not self._at_end()
            and self._peek() != CHAR_PIPE
            and self._peek() != CHAR_RPAREN
        ):
            parts.append(self._parse_quantified())

        if len(parts) == 0:
            # Empty alternative — create empty concat
            var node = ASTNode.concat(List[Int]())
            return self.ast.add_node(node^)
        if len(parts) == 1:
            return parts[0]

        var node = ASTNode.concat(parts^)
        return self.ast.add_node(node^)

    def _parse_quantified(mut self) raises -> Int:
        """quantified = atom ('*' | '+' | '?' | '{n,m}')?  '?'?"""
        var atom_idx = self._parse_atom()

        if self._at_end():
            return atom_idx

        var ch = self._peek()
        var min_rep = 0
        var max_rep = 0
        var has_quantifier = False

        if ch == CHAR_STAR:
            self.pos += 1
            min_rep = 0
            max_rep = -1
            has_quantifier = True
        elif ch == CHAR_PLUS:
            self.pos += 1
            min_rep = 1
            max_rep = -1
            has_quantifier = True
        elif ch == CHAR_QUESTION:
            self.pos += 1
            min_rep = 0
            max_rep = 1
            has_quantifier = True
        elif ch == CHAR_LBRACE:
            var result = self._try_parse_repetition()
            if result[0]:
                min_rep = result[1]
                max_rep = result[2]
                has_quantifier = True

        if not has_quantifier:
            return atom_idx

        # Check for lazy modifier
        var greedy = True
        if not self._at_end() and self._peek() == CHAR_QUESTION:
            self.pos += 1
            greedy = False

        return self.ast.add_node(
            ASTNode.quantifier(atom_idx, min_rep, max_rep, greedy)
        )

    def _try_parse_repetition(mut self) raises -> Tuple[Bool, Int, Int]:
        """Try to parse {n}, {n,}, {n,m}. Returns (success, min, max).

        If parsing fails (not a valid repetition), restores position.
        """
        var save_pos = self.pos
        self.pos += 1  # consume '{'

        if self._at_end() or not self._is_digit(self._peek()):
            self.pos = save_pos
            return (False, 0, 0)

        var min_val = self._parse_int()

        if self._at_end():
            self.pos = save_pos
            return (False, 0, 0)

        var next_ch = self._peek()
        if next_ch == CHAR_RBRACE:
            # {n} — exact
            self.pos += 1
            return (True, min_val, min_val)
        elif next_ch == CHAR_COMMA:
            self.pos += 1  # consume ','
            if self._at_end():
                self.pos = save_pos
                return (False, 0, 0)
            if self._peek() == CHAR_RBRACE:
                # {n,} — unbounded
                self.pos += 1
                return (True, min_val, -1)
            elif self._is_digit(self._peek()):
                # {n,m}
                var max_val = self._parse_int()
                if self._at_end() or self._peek() != CHAR_RBRACE:
                    self.pos = save_pos
                    return (False, 0, 0)
                self.pos += 1  # consume '}'
                if max_val < min_val:
                    raise Error(
                        String.write(
                            RegexError(
                                "Invalid repetition: min ("
                                + String(min_val)
                                + ") > max ("
                                + String(max_val)
                                + ")",
                                save_pos,
                            )
                        )
                    )
                return (True, min_val, max_val)
            else:
                self.pos = save_pos
                return (False, 0, 0)
        else:
            self.pos = save_pos
            return (False, 0, 0)

    def _parse_int(mut self) -> Int:
        """Parse a decimal integer from the current position."""
        var result = 0
        while not self._at_end() and self._is_digit(self._peek()):
            result = result * 10 + Int(self._peek() - CHAR_ZERO)
            self.pos += 1
        return result

    @staticmethod
    def _is_digit(ch: Byte) -> Bool:
        return ch >= CHAR_ZERO and ch <= CHAR_NINE

    def _parse_atom(mut self) raises -> Int:
        """atom = CHAR | '.' | '[' charset ']' | '(' regex ')' | '\\\\' ESCAPE | '^' | '$'
        """
        if self._at_end():
            raise Error(
                String.write(RegexError("Unexpected end of pattern", self.pos))
            )

        var ch = self._peek()
        if ch == CHAR_DOT:
            self.pos += 1
            return self.ast.add_node(ASTNode.dot())
        elif ch == CHAR_LBRACKET:
            return self._parse_char_class()
        elif ch == CHAR_LPAREN:
            return self._parse_group()
        elif ch == CHAR_BACKSLASH:
            return self._parse_escape()
        elif ch == CHAR_CARET:
            self.pos += 1
            return self.ast.add_node(ASTNode.anchor(AnchorKind.BOL))
        elif ch == CHAR_DOLLAR:
            self.pos += 1
            return self.ast.add_node(ASTNode.anchor(AnchorKind.EOL))
        elif ch == CHAR_STAR or ch == CHAR_PLUS or ch == CHAR_QUESTION:
            raise Error(
                String.write(
                    RegexError(
                        "Quantifier without preceding element",
                        self.pos,
                    )
                )
            )
        elif ch == CHAR_RPAREN:
            raise Error(String.write(RegexError("Unmatched ')'", self.pos)))
        else:
            self.pos += 1
            return self.ast.add_node(ASTNode.literal(UInt32(ch)))

    def _parse_group(mut self) raises -> Int:
        """Parse a group: (regex), (?:regex), (?=), (?!), (?<=), (?<!), (?P<name>).
        """
        self.pos += 1  # consume '('

        var group_index = -1  # -1 = non-capturing by default

        # Check for group modifiers
        if not self._at_end() and self._peek() == CHAR_QUESTION:
            self.pos += 1  # consume '?'
            if self._at_end():
                raise Error(
                    String.write(
                        RegexError(
                            "Unexpected end of pattern after '(?'", self.pos
                        )
                    )
                )
            var modifier = self._peek()
            if modifier == CHAR_COLON:
                self.pos += 1  # consume ':'
                # Non-capturing group — group_index stays -1
            elif modifier == CHAR_EQUALS:
                self.pos += 1  # consume '='
                var inner = self._parse_alternation()
                self._expect(CHAR_RPAREN)
                return self.ast.add_node(ASTNode.lookahead(inner, False))
            elif modifier == CHAR_BANG:
                self.pos += 1  # consume '!'
                var inner = self._parse_alternation()
                self._expect(CHAR_RPAREN)
                return self.ast.add_node(ASTNode.lookahead(inner, True))
            elif modifier == CHAR_LESS_THAN:
                self.pos += 1  # consume '<'
                if self._at_end():
                    raise Error(
                        String.write(
                            RegexError("Unexpected end after '(?<'", self.pos)
                        )
                    )
                var next_ch = self._peek()
                if next_ch == CHAR_EQUALS:
                    self.pos += 1  # consume '='
                    var inner = self._parse_alternation()
                    self._expect(CHAR_RPAREN)
                    return self.ast.add_node(ASTNode.lookbehind(inner, False))
                elif next_ch == CHAR_BANG:
                    self.pos += 1  # consume '!'
                    var inner = self._parse_alternation()
                    self._expect(CHAR_RPAREN)
                    return self.ast.add_node(ASTNode.lookbehind(inner, True))
                else:
                    raise Error(
                        String.write(
                            RegexError(
                                "Unknown lookbehind modifier '(?<"
                                + chr(Int(next_ch))
                                + "'",
                                self.pos - 2,
                            )
                        )
                    )
            elif modifier == CHAR_P_UPPER:
                self.pos += 1  # consume 'P'
                self._expect(CHAR_LESS_THAN)
                var name = self._parse_group_name()
                self._expect(CHAR_GREATER_THAN)
                self.ast.group_count += 1
                group_index = self.ast.group_count
                self.ast.group_names[name^] = group_index
            elif (
                modifier == CHAR_I_LOWER
                or modifier == CHAR_M_LOWER
                or modifier == CHAR_S_LOWER
            ):
                # Inline flags: (?i), (?m), (?s) or (?i:...) etc.
                var inline_flags = self._parse_inline_flags()
                if not self._at_end() and self._peek() == CHAR_RPAREN:
                    # (?ims) — set flags globally
                    self.pos += 1  # consume ')'
                    self.inline_flags = RegexFlags(
                        self.inline_flags.value | inline_flags.value
                    )
                    # Return empty concat node
                    var node = ASTNode.concat(List[Int]())
                    return self.ast.add_node(node^)
                elif not self._at_end() and self._peek() == CHAR_COLON:
                    self.pos += 1  # consume ':'
                    # (?ims:...) — scoped flags, treat as non-capturing group
                    # Flags are set globally for now (scoped flags would need
                    # per-node flag tracking)
                    self.inline_flags = RegexFlags(
                        self.inline_flags.value | inline_flags.value
                    )
                else:
                    raise Error(
                        String.write(
                            RegexError(
                                "Expected ')' or ':' after inline flags",
                                self.pos,
                            )
                        )
                    )
            else:
                raise Error(
                    String.write(
                        RegexError(
                            "Unknown group modifier '(?"
                            + chr(Int(modifier))
                            + "'",
                            self.pos - 1,
                        )
                    )
                )
        else:
            # Capturing group
            self.ast.group_count += 1
            group_index = self.ast.group_count

        var inner = self._parse_alternation()
        self._expect(CHAR_RPAREN)

        if group_index == -1:
            # Non-capturing: just return the inner node directly
            return inner

        return self.ast.add_node(ASTNode.group(inner, group_index))

    def _parse_inline_flags(mut self) -> RegexFlags:
        """Parse inline flag characters (i, m, s) and return the flags."""
        var flags = RegexFlags()
        while not self._at_end():
            var ch = self._peek()
            if ch == CHAR_I_LOWER:
                flags = RegexFlags(flags.value | RegexFlags.IGNORECASE)
                self.pos += 1
            elif ch == CHAR_M_LOWER:
                flags = RegexFlags(flags.value | RegexFlags.MULTILINE)
                self.pos += 1
            elif ch == CHAR_S_LOWER:
                flags = RegexFlags(flags.value | RegexFlags.DOTALL)
                self.pos += 1
            else:
                break
        return flags^

    def _parse_group_name(mut self) raises -> String:
        """Parse a group name (letters, digits, underscores)."""
        var start = self.pos
        while not self._at_end() and self._peek() != CHAR_GREATER_THAN:
            var ch = self._peek()
            if not (
                (ch >= CHAR_A_LOWER and ch <= CHAR_Z_LOWER)
                or (ch >= CHAR_A_UPPER and ch <= CHAR_Z_UPPER)
                or (ch >= CHAR_ZERO and ch <= CHAR_NINE)
                or ch == CHAR_UNDERSCORE
            ):
                raise Error(
                    String.write(
                        RegexError(
                            "Invalid group name: '" + chr(Int(ch)) + "'",
                            self.pos,
                        )
                    )
                )
            self.pos += 1
        if self.pos == start:
            raise Error(String.write(RegexError("Empty group name", self.pos)))
        return String(unsafe_from_utf8=self.pattern[start : self.pos])

    def _parse_escape(mut self) raises -> Int:
        """Parse a backslash escape sequence."""
        self.pos += 1  # consume '\\'
        if self._at_end():
            raise Error(
                String.write(
                    RegexError(
                        "Trailing backslash",
                        self.pos - 1,
                    )
                )
            )

        var ch = self._advance()

        # Word boundary anchors
        if ch == CHAR_B_LOWER:
            return self.ast.add_node(ASTNode.anchor(AnchorKind.WORD_BOUNDARY))
        elif ch == CHAR_B_UPPER:
            return self.ast.add_node(
                ASTNode.anchor(AnchorKind.NOT_WORD_BOUNDARY)
            )

        # Backreferences \1 through \9
        if ch >= CHAR_ONE and ch <= CHAR_NINE:
            var group_index = Int(ch - CHAR_ZERO)
            return self.ast.add_node(ASTNode.backreference(group_index))

        # Shorthand character classes
        if ch == CHAR_D_LOWER or ch == CHAR_D_UPPER:
            var cs = CharSet.digit()
            if ch == CHAR_D_UPPER:
                cs.negate()
            cs.build_bitmap()
            var cs_idx = self.ast.add_charset(cs^)
            var node = ASTNode.char_class(cs_idx, ch == CHAR_D_UPPER)
            return self.ast.add_node(node^)
        elif ch == CHAR_W_LOWER or ch == CHAR_W_UPPER:
            var cs = CharSet.word()
            if ch == CHAR_W_UPPER:
                cs.negate()
            cs.build_bitmap()
            var cs_idx = self.ast.add_charset(cs^)
            return self.ast.add_node(
                ASTNode.char_class(cs_idx, ch == CHAR_W_UPPER)
            )
        elif ch == CHAR_S_LOWER or ch == CHAR_S:
            var cs = CharSet.whitespace()
            if ch == CHAR_S:
                cs.negate()
            cs.build_bitmap()
            var cs_idx = self.ast.add_charset(cs^)
            return self.ast.add_node(ASTNode.char_class(cs_idx, ch == CHAR_S))

        # Literal character escapes
        if ch == CHAR_t:
            return self.ast.add_node(ASTNode.literal(UInt32(CHAR_TAB)))
        elif ch == CHAR_n:
            return self.ast.add_node(ASTNode.literal(UInt32(CHAR_NEWLINE)))
        elif ch == CHAR_r:
            return self.ast.add_node(ASTNode.literal(UInt32(CHAR_CR)))

        # Metacharacter escapes
        if (
            ch == CHAR_BACKSLASH
            or ch == CHAR_DOT
            or ch == CHAR_STAR
            or ch == CHAR_PLUS
            or ch == CHAR_QUESTION
            or ch == CHAR_LBRACKET
            or ch == CHAR_RBRACKET
            or ch == CHAR_LPAREN
            or ch == CHAR_RPAREN
            or ch == CHAR_PIPE
            or ch == CHAR_LBRACE
            or ch == CHAR_RBRACE
            or ch == CHAR_CARET
            or ch == CHAR_DOLLAR
        ):
            return self.ast.add_node(ASTNode.literal(UInt32(ch)))

        raise Error(
            String.write(
                RegexError(
                    "Invalid escape sequence '\\" + chr(Int(ch)) + "'",
                    self.pos - 2,
                )
            )
        )

    def _parse_char_class(mut self) raises -> Int:
        """Parse a character class: [abc], [a-z], [^abc], etc."""
        self.pos += 1  # consume '['
        var negated = False
        if not self._at_end() and self._peek() == CHAR_CARET:
            negated = True
            self.pos += 1

        var cs = CharSet()

        # Handle ']' as first character in class (literal)
        if not self._at_end() and self._peek() == CHAR_RBRACKET:
            cs.add_range(UInt32(CHAR_RBRACKET), UInt32(CHAR_RBRACKET))
            self.pos += 1

        while not self._at_end() and self._peek() != CHAR_RBRACKET:
            var ch = self._advance()
            if ch == CHAR_BACKSLASH:
                # Escape inside char class
                if self._at_end():
                    raise Error(
                        String.write(
                            RegexError(
                                "Trailing backslash in character class",
                                self.pos - 1,
                            )
                        )
                    )
                var esc = self._peek()
                # Shorthand classes inside char class
                if esc == CHAR_D_LOWER:
                    self.pos += 1
                    cs.add_range(UInt32(CHAR_ZERO), UInt32(CHAR_NINE))
                    continue
                elif esc == CHAR_D_UPPER:
                    # \D inside a class is complex — skip for now, just add ranges
                    self.pos += 1
                    cs.add_range(0, UInt32(CHAR_ZERO) - 1)
                    cs.add_range(UInt32(CHAR_NINE) + 1, 127)
                    continue
                elif esc == CHAR_W_LOWER:
                    self.pos += 1
                    cs.add_range(UInt32(CHAR_A_LOWER), UInt32(CHAR_Z_LOWER))
                    cs.add_range(UInt32(CHAR_A_UPPER), UInt32(CHAR_Z_UPPER))
                    cs.add_range(UInt32(CHAR_ZERO), UInt32(CHAR_NINE))
                    cs.add_range(
                        UInt32(CHAR_UNDERSCORE), UInt32(CHAR_UNDERSCORE)
                    )
                    continue
                elif esc == CHAR_W_UPPER:
                    # \W inside class — hard to represent, skip detailed handling
                    self.pos += 1
                    continue
                elif esc == CHAR_S_LOWER:
                    self.pos += 1
                    cs.add_range(UInt32(CHAR_SPACE), UInt32(CHAR_SPACE))
                    cs.add_range(UInt32(CHAR_TAB), UInt32(CHAR_TAB))
                    cs.add_range(UInt32(CHAR_NEWLINE), UInt32(CHAR_NEWLINE))
                    cs.add_range(UInt32(CHAR_CR), UInt32(CHAR_CR))
                    cs.add_range(0x0B, 0x0B)
                    cs.add_range(0x0C, 0x0C)
                    continue
                elif esc == CHAR_S:
                    self.pos += 1
                    continue
                elif esc == CHAR_t:
                    self.pos += 1
                    ch = CHAR_TAB
                elif esc == CHAR_n:
                    self.pos += 1
                    ch = CHAR_NEWLINE
                elif esc == CHAR_r:
                    self.pos += 1
                    ch = CHAR_CR
                else:
                    ch = self._advance()

            # Check for range: a-z
            if not self._at_end() and self._peek() == CHAR_MINUS:
                # Peek ahead to see if this is a range or a literal '-' at end
                if (
                    self.pos + 1 < len(self.pattern)
                    and self.pattern.unsafe_get(self.pos + 1) != CHAR_RBRACKET
                ):
                    self.pos += 1  # consume '-'
                    var hi_ch = self._advance()
                    if hi_ch == CHAR_BACKSLASH:
                        if self._at_end():
                            raise Error(
                                String.write(
                                    RegexError(
                                        "Trailing backslash in character class",
                                        self.pos - 1,
                                    )
                                )
                            )
                        hi_ch = self._advance()
                    cs.add_range(UInt32(ch), UInt32(hi_ch))
                else:
                    cs.add_range(UInt32(ch), UInt32(ch))
            else:
                cs.add_range(UInt32(ch), UInt32(ch))

        if self._at_end():
            raise Error(
                String.write(
                    RegexError("Unterminated character class", self.pos)
                )
            )
        self.pos += 1  # consume ']'

        if negated:
            cs.negate()

        var cs_idx = self.ast.add_charset(cs^)
        return self.ast.add_node(ASTNode.char_class(cs_idx, negated))


def parse(pattern: String) raises -> AST:
    """Parse a regex pattern string into an AST.

    Inline flags (e.g. ``(?i)``) are stored in ``ast.flags``.
    """
    var p = Parser(pattern.as_bytes())
    return p.parse()
