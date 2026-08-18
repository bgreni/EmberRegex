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
    CHAR_C_LOWER,
    CHAR_CARET,
    CHAR_COLON,
    CHAR_COMMA,
    CHAR_CR,
    CHAR_DOLLAR,
    CHAR_DOT,
    CHAR_D_LOWER,
    CHAR_FF,
    CHAR_H_LOWER,
    CHAR_H_UPPER,
    CHAR_D_UPPER,
    CHAR_EQUALS,
    CHAR_F_LOWER,
    CHAR_F_UPPER,
    CHAR_GREATER_THAN,
    CHAR_G_LOWER,
    CHAR_HASH,
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
    CHAR_P_LOWER,
    CHAR_P_UPPER,
    CHAR_QUESTION,
    CHAR_RBRACE,
    CHAR_RBRACKET,
    CHAR_RPAREN,
    CHAR_S,
    CHAR_SEVEN,
    CHAR_SPACE,
    CHAR_STAR,
    CHAR_S_LOWER,
    CHAR_TAB,
    CHAR_U_LOWER,
    CHAR_VTAB,
    CHAR_V_LOWER,
    CHAR_V_UPPER,
    CHAR_U_UPPER,
    CHAR_UNDERSCORE,
    CHAR_W_LOWER,
    CHAR_W_UPPER,
    CHAR_X_LOWER,
    CHAR_ZERO,
    CHAR_Z_LOWER,
    CHAR_Z_UPPER,
    CHAR_n,
    CHAR_r,
    CHAR_t,
)
from .ast import AST, ASTNode, ASTNodeKind, AnchorKind
from .charset import CharSet, CharRange
from .utf8 import negate_ranges, unicode_property
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
        self._consume_verbs()
        var root = self._parse_alternation()
        self.ast.root = root
        if self.pos < len(self.pattern):
            raise Error(String(RegexError("Unexpected character", self.pos)))
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
                String(
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
        while True:
            self._skip_verbose()
            if (
                self._at_end()
                or self._peek() == CHAR_PIPE
                or self._peek() == CHAR_RPAREN
            ):
                break
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

        self._skip_verbose()

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
                        String(
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

    @staticmethod
    def _is_flag_char(ch: Byte) -> Bool:
        return (
            ch == CHAR_I_LOWER
            or ch == CHAR_M_LOWER
            or ch == CHAR_S_LOWER
            or ch == CHAR_X_LOWER
            or ch == CHAR_U_LOWER
        )

    def _parse_hex_digits(mut self, count: Int) raises -> UInt32:
        """Parse exactly `count` hex digits and return their value."""
        var value: UInt32 = 0
        for _ in range(count):
            if self._at_end():
                raise Error(String(RegexError("Expected hex digit", self.pos)))
            var ch = self._advance()
            if ch >= CHAR_ZERO and ch <= CHAR_NINE:
                value = value * 16 + UInt32(ch - CHAR_ZERO)
            elif ch >= CHAR_A_LOWER and ch <= CHAR_F_LOWER:
                value = value * 16 + UInt32(ch - CHAR_A_LOWER + 10)
            elif ch >= CHAR_A_UPPER and ch <= CHAR_F_UPPER:
                value = value * 16 + UInt32(ch - CHAR_A_UPPER + 10)
            else:
                raise Error(
                    String(
                        RegexError(
                            "Invalid hex digit '" + chr(Int(ch)) + "'",
                            self.pos - 1,
                        )
                    )
                )
        return value

    def _skip_verbose(mut self):
        """Skip whitespace and # comments when verbose mode is active."""
        if not self.inline_flags.verbose():
            return
        while not self._at_end():
            var ch = self._peek()
            if ch == CHAR_HASH:
                # Skip until end of line
                while not self._at_end() and self._peek() != CHAR_NEWLINE:
                    self.pos += 1
            elif (
                ch == CHAR_SPACE
                or ch == CHAR_TAB
                or ch == CHAR_NEWLINE
                or ch == CHAR_CR
            ):
                self.pos += 1
            else:
                break

    def _parse_atom(mut self) raises -> Int:
        """atom = CHAR | '.' | '[' charset ']' | '(' regex ')' | '\\\\' ESCAPE | '^' | '$'
        """
        if self._at_end():
            raise Error(
                String(RegexError("Unexpected end of pattern", self.pos))
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
                String(
                    RegexError(
                        "Quantifier without preceding element",
                        self.pos,
                    )
                )
            )
        elif ch == CHAR_RPAREN:
            raise Error(String(RegexError("Unmatched ')'", self.pos)))
        else:
            self.pos += 1
            # In UTF-8 mode a multi-byte character is ONE atom, so a
            # following quantifier applies to the whole codepoint. Byte
            # mode leaves each byte its own literal, which concatenates
            # to the same sequence but makes `α+` quantify only the last
            # byte — the reason this branch exists.
            if self.inline_flags.unicode() and ch >= 0xC0:
                var extra = 1
                var cp = UInt32(ch) & 0x1F
                if ch >= 0xF0:
                    extra = 3
                    cp = UInt32(ch) & 0x07
                elif ch >= 0xE0:
                    extra = 2
                    cp = UInt32(ch) & 0x0F
                for _ in range(extra):
                    if self._at_end():
                        break
                    var cont = self._advance()
                    cp = (cp << 6) | (UInt32(cont) & 0x3F)
                return self.ast.add_node(ASTNode.literal(cp))
            return self.ast.add_node(ASTNode.literal(UInt32(ch)))

    def _consume_verbs(mut self) raises:
        """Consume leading `(*UTF8)` verbs; `(*UCP)` is rejected loudly.

        PCRE and Hyperscan spell UTF-8 mode this way; `(?u)` is the same
        switch in this library's inline-flag syntax.
        """
        while (
            self.pos + 2 < len(self.pattern)
            and self.pattern.unsafe_get(self.pos) == CHAR_LPAREN
            and self.pattern.unsafe_get(self.pos + 1) == CHAR_STAR
        ):
            var close = self.pos + 2
            while (
                close < len(self.pattern)
                and self.pattern.unsafe_get(close) != CHAR_RPAREN
            ):
                close += 1
            if close >= len(self.pattern):
                return
            var name = String("")
            for i in range(self.pos + 2, close):
                name += String(chr(Int(self.pattern.unsafe_get(i))))
            if name == "UTF8" or name == "UTF":
                self.inline_flags = RegexFlags(
                    self.inline_flags.value | RegexFlags.UNICODE
                )
                self.pos = close + 1
            elif name == "UCP":
                raise Error(
                    String(
                        RegexError(
                            (
                                "(*UCP) is not supported: UTF-8 mode does"
                                " not give \\d \\w \\s \\b their Unicode"
                                " meanings (PCRE's UCP contract); use"
                                " (*UTF8) or (?u) for codepoint classes"
                                " and \\p{...} for Unicode shorthands"
                            ),
                            self.pos,
                        )
                    )
                )
            else:
                raise Error(
                    String(
                        RegexError("Unknown verb '(*" + name + ")'", self.pos)
                    )
                )

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
                    String(
                        RegexError(
                            "Unexpected end of pattern after '(?'", self.pos
                        )
                    )
                )
            var modifier = self._peek()
            if modifier == CHAR_HASH:
                # (?# comment) — consumed and contributes nothing. An
                # empty CONCAT is what the NFA builder turns into
                # epsilon, so the comment disappears entirely.
                self.pos += 1  # consume '#'
                while not self._at_end() and self._peek() != CHAR_RPAREN:
                    self.pos += 1
                self._expect(CHAR_RPAREN)
                var empty = ASTNode(ASTNodeKind.CONCAT)
                return self.ast.add_node(empty^)
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
                        String(
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
                        String(
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
            elif modifier == CHAR_HASH:
                # Inline comment: (?#...) — skip until closing ')'
                self.pos += 1  # consume '#'
                while not self._at_end() and self._peek() != CHAR_RPAREN:
                    self.pos += 1
                self._expect(CHAR_RPAREN)
                var node = ASTNode.concat(List[Int]())
                return self.ast.add_node(node^)
            elif Self._is_flag_char(modifier) or modifier == CHAR_MINUS:
                # Inline flags: (?i), (?m), (?s), (?x), (?i-m), (?-i), etc.
                var add_flags, remove_flags = self._parse_inline_flags()
                if not self._at_end() and self._peek() == CHAR_RPAREN:
                    # (?flags) or (?-flags) — apply globally
                    self.pos += 1  # consume ')'
                    self.inline_flags = RegexFlags(
                        (self.inline_flags.value | add_flags.value)
                        & ~remove_flags.value
                    )
                    var node = ASTNode.concat(List[Int]())
                    return self.ast.add_node(node^)
                elif not self._at_end() and self._peek() == CHAR_COLON:
                    self.pos += 1  # consume ':'
                    # Temporarily apply verbose change during inner parse
                    var saved_flags = self.inline_flags
                    self.inline_flags = RegexFlags(
                        (self.inline_flags.value | add_flags.value)
                        & ~remove_flags.value
                    )
                    var inner = self._parse_alternation()
                    self.inline_flags = saved_flags
                    self._expect(CHAR_RPAREN)
                    return self.ast.add_node(
                        ASTNode.scoped_flags(inner, add_flags, remove_flags)
                    )
                else:
                    raise Error(
                        String(
                            RegexError(
                                "Expected ')' or ':' after inline flags",
                                self.pos,
                            )
                        )
                    )
            else:
                raise Error(
                    String(
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

    def _parse_one_flag(mut self) -> Int:
        """Consume one flag character and return its bitmask, or 0 if not a flag.
        """
        var ch = self._peek()
        if ch == CHAR_I_LOWER:
            self.pos += 1
            return RegexFlags.IGNORECASE
        elif ch == CHAR_M_LOWER:
            self.pos += 1
            return RegexFlags.MULTILINE
        elif ch == CHAR_S_LOWER:
            self.pos += 1
            return RegexFlags.DOTALL
        elif ch == CHAR_X_LOWER:
            self.pos += 1
            return RegexFlags.VERBOSE
        elif ch == CHAR_U_LOWER:
            self.pos += 1
            return RegexFlags.UNICODE
        return 0

    def _parse_inline_flags(mut self) -> Tuple[RegexFlags, RegexFlags]:
        """Parse inline flag chars (i, m, s, x) with optional removal via '-'.

        Returns (add_flags, remove_flags).
        """
        var add_val = 0
        var remove_val = 0

        # Collect flags to add
        while not self._at_end():
            var bit = self._parse_one_flag()
            if bit == 0:
                break
            add_val |= bit

        # Collect flags to remove after '-'
        if not self._at_end() and self._peek() == CHAR_MINUS:
            self.pos += 1  # consume '-'
            while not self._at_end():
                var bit = self._parse_one_flag()
                if bit == 0:
                    break
                remove_val |= bit

        return (RegexFlags(add_val), RegexFlags(remove_val))

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
                    String(
                        RegexError(
                            "Invalid group name: '" + chr(Int(ch)) + "'",
                            self.pos,
                        )
                    )
                )
            self.pos += 1
        if self.pos == start:
            raise Error(String(RegexError("Empty group name", self.pos)))
        return String(unsafe_from_utf8=self.pattern[start : self.pos])

    def _parse_escape(mut self) raises -> Int:
        """Parse a backslash escape sequence."""
        self.pos += 1  # consume '\\'
        if self._at_end():
            raise Error(
                String(
                    RegexError(
                        "Trailing backslash",
                        self.pos - 1,
                    )
                )
            )

        var ch = self._advance()

        # String anchors: \A, \z, \Z. Unlike ^ and $ these pin to the
        # STRING, so (?m) does not promote them (see AnchorKind.BOS/EOS).
        # \Z is Python's (end of string), not PCRE's before-trailing-\n.
        if ch == CHAR_A_UPPER:
            return self.ast.add_node(ASTNode.anchor(AnchorKind.BOS))
        elif ch == CHAR_Z_LOWER or ch == CHAR_Z_UPPER:
            return self.ast.add_node(ASTNode.anchor(AnchorKind.EOS))

        # Horizontal / vertical whitespace classes, PCRE and Hyperscan
        # semantics. NOTE the deliberate divergence from Python, where \v
        # is the single vertical-tab character rather than a class
        # (decided 2026-07-27; Hyperscan parity is this plan's goal and
        # \v previously errored, so nothing silently changed meaning).
        if ch == CHAR_H_LOWER or ch == CHAR_H_UPPER:
            var cs = CharSet()
            cs.add_range(UInt32(CHAR_SPACE), UInt32(CHAR_SPACE))
            cs.add_range(UInt32(CHAR_TAB), UInt32(CHAR_TAB))
            if ch == CHAR_H_UPPER:
                cs.negate()
            cs.build_bitmap()
            var cs_idx = self.ast.add_charset(cs^)
            return self.ast.add_node(
                ASTNode.char_class(cs_idx, ch == CHAR_H_UPPER)
            )
        if ch == CHAR_V_LOWER or ch == CHAR_V_UPPER:
            var cs = CharSet()
            cs.add_range(UInt32(CHAR_NEWLINE), UInt32(CHAR_CR))  # \n \v \f \r
            if ch == CHAR_V_UPPER:
                cs.negate()
            cs.build_bitmap()
            var cs_idx = self.ast.add_charset(cs^)
            return self.ast.add_node(
                ASTNode.char_class(cs_idx, ch == CHAR_V_UPPER)
            )

        # Word boundary anchors
        if ch == CHAR_B_LOWER:
            return self.ast.add_node(ASTNode.anchor(AnchorKind.WORD_BOUNDARY))
        elif ch == CHAR_B_UPPER:
            return self.ast.add_node(
                ASTNode.anchor(AnchorKind.NOT_WORD_BOUNDARY)
            )

        # Octal escapes: \0 plus up to two octal digits (Python's
        # reading: \0 = NUL, \07 = BEL, \012 = LF). A non-octal digit
        # stops the parse, so \08 is NUL followed by literal '8'.
        # Non-zero-leading forms (\1..\9) stay backreferences below.
        if ch == CHAR_ZERO:
            var cp = UInt32(0)
            var ndig = 0
            while (
                ndig < 2
                and not self._at_end()
                and self._peek() >= CHAR_ZERO
                and self._peek() <= CHAR_SEVEN
            ):
                cp = (cp << 3) | UInt32(self._advance() - CHAR_ZERO)
                ndig += 1
            return self.ast.add_node(ASTNode.literal(cp))

        # Backreferences \1 through \9
        if ch >= CHAR_ONE and ch <= CHAR_NINE:
            var group_index = Int(ch - CHAR_ZERO)
            if group_index > self.ast.group_count:
                raise Error(
                    String(
                        RegexError(
                            "Invalid backreference \\"
                            + String(group_index)
                            + ": group does not exist",
                            self.pos - 2,
                        )
                    )
                )
            return self.ast.add_node(ASTNode.backreference(group_index))

        # Named / numeric backreferences: \g<name> or \g<N>
        if ch == CHAR_G_LOWER:
            self._expect(CHAR_LESS_THAN)
            if not self._at_end() and Self._is_digit(self._peek()):
                var n = self._parse_int()
                self._expect(CHAR_GREATER_THAN)
                if n < 1 or n > self.ast.group_count:
                    raise Error(
                        String(
                            RegexError(
                                "Invalid backreference \\g<"
                                + String(n)
                                + ">: group does not exist",
                                self.pos - 1,
                            )
                        )
                    )
                return self.ast.add_node(ASTNode.backreference(n))
            var name = self._parse_group_name()
            self._expect(CHAR_GREATER_THAN)
            var maybe_idx = self.ast.group_names.get(name)
            if maybe_idx:
                return self.ast.add_node(
                    ASTNode.backreference(maybe_idx.value())
                )
            raise Error(
                String(
                    RegexError(
                        "Unknown group name '" + name + "' in \\g<>",
                        self.pos - name.byte_length() - 2,
                    )
                )
            )

        # Hex escape: \xHH
        if ch == CHAR_X_LOWER:
            var cp = self._parse_hex_digits(2)
            return self.ast.add_node(ASTNode.literal(cp))

        # Unicode escapes: \uHHHH and \UHHHHHHHH
        if ch == CHAR_U_LOWER:
            var cp = self._parse_hex_digits(4)
            if cp > 255 and not self.inline_flags.unicode():
                raise Error(
                    String(
                        RegexError(
                            (
                                "Unicode code point > U+00FF needs UTF-8 mode"
                                " — prefix the pattern with (?u) or (*UTF8)"
                            ),
                            self.pos - 5,
                        )
                    )
                )
            return self.ast.add_node(ASTNode.literal(cp))
        if ch == CHAR_U_UPPER:
            var cp = self._parse_hex_digits(8)
            if cp > 255 and not self.inline_flags.unicode():
                raise Error(
                    String(
                        RegexError(
                            (
                                "Unicode code point > U+00FF needs UTF-8 mode"
                                " — prefix the pattern with (?u) or (*UTF8)"
                            ),
                            self.pos - 9,
                        )
                    )
                )
            return self.ast.add_node(ASTNode.literal(cp))

        # Control character: \cX  (value = X & 0x1F)
        if ch == CHAR_C_LOWER:
            if self._at_end():
                raise Error(
                    String(
                        RegexError("Expected character after \\c", self.pos - 1)
                    )
                )
            var ctrl = self._advance()
            var cp = UInt32(ctrl) & 0x1F
            return self.ast.add_node(ASTNode.literal(cp))

        # Unicode properties: \p{L}, \P{Nd}, \p{Greek}. The ranges are
        # CODEPOINT ranges; UTF-8 mode compiles them to byte sequences,
        # and byte mode keeps only their ASCII part (see nfa.mojo).
        if ch == CHAR_P_LOWER or ch == CHAR_P_UPPER:
            var negated_prop = ch == CHAR_P_UPPER
            self._expect(CHAR_LBRACE)
            var pname = String("")
            while not self._at_end() and self._peek() != CHAR_RBRACE:
                pname += String(chr(Int(self._advance())))
            self._expect(CHAR_RBRACE)
            var known = True
            var pranges = unicode_property(pname, known)
            if not known:
                raise Error(
                    String(
                        RegexError(
                            "Unknown Unicode property '\\p{" + pname + "}'",
                            self.pos,
                        )
                    )
                )
            if negated_prop:
                pranges = negate_ranges(pranges)
            var pcs = CharSet()
            for i in range(len(pranges) // 2):
                pcs.add_range(
                    UInt32(pranges[2 * i]), UInt32(pranges[2 * i + 1])
                )
            pcs.build_bitmap()
            var pidx = self.ast.add_charset(pcs^)
            return self.ast.add_node(ASTNode.char_class(pidx, False))

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
            String(
                RegexError(
                    "Invalid escape sequence '\\" + chr(Int(ch)) + "'",
                    self.pos - 2,
                )
            )
        )

    def _parse_cc_codepoint(mut self) raises -> UInt32:
        """Parse one code-point value inside a char class (after consuming the
        leading char or backslash).  Does NOT handle shorthand classes like
        \\d/\\w/\\s — those must be detected before calling this helper.

        Consumes only single-codepoint escapes and literal bytes.
        Returns the code-point value as UInt32.
        """
        var ch = self._advance()
        if ch != CHAR_BACKSLASH:
            # In UTF-8 mode a multi-byte character written directly in the
            # class is ONE member, not several. Byte mode keeps the old
            # reading (each byte a member), which is what ROADMAP §3
            # records as the byte-mode charset question.
            if self.inline_flags.unicode() and ch >= 0xC0:
                var extra = 1
                if ch >= 0xF0:
                    extra = 3
                elif ch >= 0xE0:
                    extra = 2
                var cp: UInt32
                if ch >= 0xF0:
                    cp = UInt32(ch) & 0x07
                elif ch >= 0xE0:
                    cp = UInt32(ch) & 0x0F
                else:
                    cp = UInt32(ch) & 0x1F
                for _ in range(extra):
                    if self._at_end():
                        break
                    var cont = self._advance()
                    cp = (cp << 6) | (UInt32(cont) & 0x3F)
                return cp
            return UInt32(ch)

        # It is an escape
        if self._at_end():
            raise Error(
                String(
                    RegexError(
                        "Trailing backslash in character class", self.pos - 1
                    )
                )
            )
        var esc = self._advance()
        if esc == CHAR_t:
            return UInt32(CHAR_TAB)
        elif esc == CHAR_n:
            return UInt32(CHAR_NEWLINE)
        elif esc == CHAR_r:
            return UInt32(CHAR_CR)
        # NOTE: charset context deliberately does NOT consume octal
        # digits after \0 (atom-level \012 is LF; [\012] stays
        # {NUL,'1','2'}). Aligning this with the atom path (and
        # Python) is a known open item — do not "fix" the atom path
        # to match this one.
        elif esc == CHAR_ZERO:
            return 0
        elif esc == CHAR_X_LOWER:
            return self._parse_hex_digits(2)
        elif esc == CHAR_U_LOWER:
            var cp = self._parse_hex_digits(4)
            if cp > 255 and not self.inline_flags.unicode():
                raise Error(
                    String(
                        RegexError(
                            (
                                "Unicode code point > U+00FF needs UTF-8 mode"
                                " — prefix the pattern with (?u) or (*UTF8)"
                            ),
                            self.pos - 5,
                        )
                    )
                )
            return cp
        elif esc == CHAR_U_UPPER:
            var cp = self._parse_hex_digits(8)
            if cp > 255 and not self.inline_flags.unicode():
                raise Error(
                    String(
                        RegexError(
                            (
                                "Unicode code point > U+00FF needs UTF-8 mode"
                                " — prefix the pattern with (?u) or (*UTF8)"
                            ),
                            self.pos - 9,
                        )
                    )
                )
            return cp
        elif esc == CHAR_C_LOWER:
            if self._at_end():
                raise Error(
                    String(
                        RegexError("Expected character after \\c", self.pos - 1)
                    )
                )
            return UInt32(self._advance()) & 0x1F
        # Any other escaped char is taken literally (metacharacter or letter)
        return UInt32(esc)

    @staticmethod
    def _posix_class(name: String, mut cs: CharSet) -> Bool:
        """Add a POSIX `[:name:]` class to `cs`. ASCII definitions, which
        is what byte mode can express. Returns False for an unknown name
        so the caller can report it rather than silently matching
        nothing."""
        if name == "alpha":
            cs.add_range(65, 90)
            cs.add_range(97, 122)
        elif name == "digit":
            cs.add_range(48, 57)
        elif name == "alnum":
            cs.add_range(48, 57)
            cs.add_range(65, 90)
            cs.add_range(97, 122)
        elif name == "upper":
            cs.add_range(65, 90)
        elif name == "lower":
            cs.add_range(97, 122)
        elif name == "space":
            cs.add_range(9, 13)
            cs.add_range(32, 32)
        elif name == "blank":
            cs.add_range(9, 9)
            cs.add_range(32, 32)
        elif name == "punct":
            cs.add_range(33, 47)
            cs.add_range(58, 64)
            cs.add_range(91, 96)
            cs.add_range(123, 126)
        elif name == "xdigit":
            cs.add_range(48, 57)
            cs.add_range(65, 70)
            cs.add_range(97, 102)
        elif name == "word":
            cs.add_range(48, 57)
            cs.add_range(65, 90)
            cs.add_range(95, 95)
            cs.add_range(97, 122)
        elif name == "cntrl":
            cs.add_range(0, 31)
            cs.add_range(127, 127)
        elif name == "print":
            cs.add_range(32, 126)
        elif name == "graph":
            cs.add_range(33, 126)
        else:
            return False
        return True

    def _try_posix_class(mut self, mut cs: CharSet) raises -> Bool:
        """At `[:`, consume a POSIX class and add it to `cs`.

        Returns False without consuming anything when the input is a
        plain `[` inside the class (which is a literal bracket), so the
        caller can fall through to normal parsing.
        """
        if self._peek() != CHAR_LBRACKET:
            return False
        if self.pos + 1 >= len(self.pattern):
            return False
        if self.pattern.unsafe_get(self.pos + 1) != CHAR_COLON:
            return False
        var start = self.pos
        self.pos += 2  # consume '[:'
        var negated = False
        if not self._at_end() and self._peek() == CHAR_CARET:
            negated = True
            self.pos += 1
        var name = String("")
        while not self._at_end() and self._peek() != CHAR_COLON:
            name += String(chr(Int(self._advance())))
        # Expect ':]'
        if (
            self._at_end()
            or self.pos + 1 >= len(self.pattern)
            or self.pattern.unsafe_get(self.pos + 1) != CHAR_RBRACKET
        ):
            self.pos = start
            return False  # not actually a POSIX class; treat '[' literally
        self.pos += 2  # consume ':]'
        if negated:
            # `[:^alpha:]` is the complement WITHIN the enclosing class,
            # so build it standalone and add the inverted byte set.
            var tmp = CharSet()
            if not Self._posix_class(name, tmp):
                raise Error(
                    String(
                        RegexError(
                            "Unknown POSIX class '[:" + name + ":]'", start
                        )
                    )
                )
            tmp.build_bitmap()
            for b in range(256):
                if not tmp.contains(UInt32(b)):
                    cs.add_range(UInt32(b), UInt32(b))
            return True
        if not Self._posix_class(name, cs):
            raise Error(
                String(
                    RegexError("Unknown POSIX class '[:" + name + ":]'", start)
                )
            )
        return True

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
            # POSIX bracket expressions: [[:alpha:]], [[:^digit:]].
            if self._try_posix_class(cs):
                continue
            # Check for shorthand classes first (they add multiple ranges)
            if self._peek() == CHAR_BACKSLASH and self.pos + 1 < len(
                self.pattern
            ):
                var esc = self.pattern.unsafe_get(self.pos + 1)
                if esc == CHAR_D_LOWER:
                    self.pos += 2
                    cs.add_range(UInt32(CHAR_ZERO), UInt32(CHAR_NINE))
                    continue
                elif esc == CHAR_D_UPPER:
                    self.pos += 2
                    cs.add_range(0, UInt32(CHAR_ZERO) - 1)
                    cs.add_range(UInt32(CHAR_NINE) + 1, 255)
                    continue
                elif esc == CHAR_W_LOWER:
                    self.pos += 2
                    cs.add_range(UInt32(CHAR_A_LOWER), UInt32(CHAR_Z_LOWER))
                    cs.add_range(UInt32(CHAR_A_UPPER), UInt32(CHAR_Z_UPPER))
                    cs.add_range(UInt32(CHAR_ZERO), UInt32(CHAR_NINE))
                    cs.add_range(
                        UInt32(CHAR_UNDERSCORE), UInt32(CHAR_UNDERSCORE)
                    )
                    continue
                elif esc == CHAR_W_UPPER:
                    self.pos += 2
                    cs.add_range(0, 47)
                    cs.add_range(58, 64)
                    cs.add_range(91, 94)
                    cs.add_range(96, 96)
                    cs.add_range(123, 255)
                    continue
                elif esc == CHAR_S_LOWER:
                    self.pos += 2
                    cs.add_range(UInt32(CHAR_SPACE), UInt32(CHAR_SPACE))
                    cs.add_range(UInt32(CHAR_TAB), UInt32(CHAR_TAB))
                    cs.add_range(UInt32(CHAR_NEWLINE), UInt32(CHAR_NEWLINE))
                    cs.add_range(UInt32(CHAR_CR), UInt32(CHAR_CR))
                    cs.add_range(0x0B, 0x0B)
                    cs.add_range(0x0C, 0x0C)
                    continue
                elif esc == CHAR_S:
                    self.pos += 2
                    cs.add_range(0, 8)
                    cs.add_range(14, 31)
                    cs.add_range(33, 255)
                    continue

            # Single code-point (literal or single-char escape)
            var lo = self._parse_cc_codepoint()

            # Check for range: lo-hi
            if (
                not self._at_end()
                and self._peek() == CHAR_MINUS
                and self.pos + 1 < len(self.pattern)
                and self.pattern.unsafe_get(self.pos + 1) != CHAR_RBRACKET
            ):
                self.pos += 1  # consume '-'
                # A shorthand class (\d, \w, \s, ...) is not a valid range
                # endpoint; _parse_cc_codepoint would silently read it as a
                # literal letter (e.g. [a-\d] becoming [a-d]).
                if self._peek() == CHAR_BACKSLASH and self.pos + 1 < len(
                    self.pattern
                ):
                    var esc = self.pattern.unsafe_get(self.pos + 1)
                    if (
                        esc == CHAR_D_LOWER
                        or esc == CHAR_D_UPPER
                        or esc == CHAR_W_LOWER
                        or esc == CHAR_W_UPPER
                        or esc == CHAR_S_LOWER
                        or esc == CHAR_S
                    ):
                        raise Error(
                            String(
                                RegexError(
                                    (
                                        "Bad character range: shorthand class"
                                        " cannot be a range endpoint"
                                    ),
                                    self.pos,
                                )
                            )
                        )
                var hi = self._parse_cc_codepoint()
                if hi < lo:
                    raise Error(
                        String(
                            RegexError("Invalid character range", self.pos - 2)
                        )
                    )
                cs.add_range(lo, hi)
            else:
                cs.add_range(lo, lo)

        if self._at_end():
            raise Error(
                String(RegexError("Unterminated character class", self.pos))
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
