"""Recursive descent regex parser.

Grammar (Milestone 1):
    regex      = alternation
    alternation = concat ('|' concat)*
    concat     = quantified+
    quantified = atom ('*' | '+' | '?')?
    atom       = CHAR | '.' | '[' charset ']' | '(' regex ')' | '\\' ESCAPE
"""

from .ast import AST, ASTNode, ASTNodeKind, AnchorKind
from .charset import CharSet, CharRange
from .errors import RegexError
from .flags import RegexFlags


struct Parser(Movable):
    """Recursive descent parser for regex patterns."""

    var pattern: String
    var pos: Int
    var ast: AST

    def __init__(out self, pattern: String):
        self.pattern = pattern
        self.pos = 0
        self.ast = AST()

    def parse(mut self) raises -> AST:
        """Parse the pattern and return the AST."""
        var root = self._parse_alternation()
        self.ast.root = root
        if self.pos < len(self.pattern):
            raise Error(
                String.write(
                    RegexError("Unexpected character", self.pos)
                )
            )
        # Build bitmaps for all charsets
        for i in range(len(self.ast.charsets)):
            self.ast.charsets[i].build_bitmap()
        var result = self.ast^
        self.ast = AST()
        return result^

    def _peek(self) -> Int:
        """Look at the current character without consuming. Returns -1 at end."""
        if self.pos >= len(self.pattern):
            return -1
        return Int(self.pattern.as_bytes()[self.pos])

    def _advance(mut self) -> Int:
        """Consume and return the current character."""
        var ch = self._peek()
        self.pos += 1
        return ch

    def _at_end(self) -> Bool:
        return self.pos >= len(self.pattern)

    def _expect(mut self, ch: Int) raises:
        if self._at_end() or self._peek() != ch:
            raise Error(
                String.write(
                    RegexError(
                        "Expected '" + chr(ch) + "'",
                        self.pos,
                    )
                )
            )
        self.pos += 1

    # --- Grammar productions ---

    def _parse_alternation(mut self) raises -> Int:
        """alternation = concat ('|' concat)*"""
        var first = self._parse_concat()
        if self._at_end() or self._peek() != ord("|"):
            return first

        var alternatives = List[Int]()
        alternatives.append(first)
        while not self._at_end() and self._peek() == ord("|"):
            self.pos += 1  # consume '|'
            alternatives.append(self._parse_concat())

        var node = ASTNode.alternation(alternatives^)
        return self.ast.add_node(node^)

    def _parse_concat(mut self) raises -> Int:
        """concat = quantified+"""
        var parts = List[Int]()
        while not self._at_end() and self._peek() != ord("|") and self._peek() != ord(")"):
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

        if ch == ord("*"):
            self.pos += 1
            min_rep = 0
            max_rep = -1
            has_quantifier = True
        elif ch == ord("+"):
            self.pos += 1
            min_rep = 1
            max_rep = -1
            has_quantifier = True
        elif ch == ord("?"):
            self.pos += 1
            min_rep = 0
            max_rep = 1
            has_quantifier = True
        elif ch == ord("{"):
            var result = self._try_parse_repetition()
            if result[0]:
                min_rep = result[1]
                max_rep = result[2]
                has_quantifier = True

        if not has_quantifier:
            return atom_idx

        # Check for lazy modifier
        var greedy = True
        if not self._at_end() and self._peek() == ord("?"):
            self.pos += 1
            greedy = False

        var node = ASTNode.quantifier(atom_idx, min_rep, max_rep, greedy)
        return self.ast.add_node(node^)

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
        if next_ch == ord("}"):
            # {n} — exact
            self.pos += 1
            return (True, min_val, min_val)
        elif next_ch == ord(","):
            self.pos += 1  # consume ','
            if self._at_end():
                self.pos = save_pos
                return (False, 0, 0)
            if self._peek() == ord("}"):
                # {n,} — unbounded
                self.pos += 1
                return (True, min_val, -1)
            elif self._is_digit(self._peek()):
                # {n,m}
                var max_val = self._parse_int()
                if self._at_end() or self._peek() != ord("}"):
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
            result = result * 10 + (self._peek() - ord("0"))
            self.pos += 1
        return result

    @staticmethod
    def _is_digit(ch: Int) -> Bool:
        return ch >= ord("0") and ch <= ord("9")

    def _parse_atom(mut self) raises -> Int:
        """atom = CHAR | '.' | '[' charset ']' | '(' regex ')' | '\\\\' ESCAPE | '^' | '$'"""
        if self._at_end():
            raise Error(
                String.write(
                    RegexError("Unexpected end of pattern", self.pos)
                )
            )

        var ch = self._peek()

        if ch == ord("."):
            self.pos += 1
            var node = ASTNode.dot()
            return self.ast.add_node(node^)
        elif ch == ord("["):
            return self._parse_char_class()
        elif ch == ord("("):
            return self._parse_group()
        elif ch == ord("\\"):
            return self._parse_escape()
        elif ch == ord("^"):
            self.pos += 1
            var node = ASTNode.anchor(AnchorKind.BOL)
            return self.ast.add_node(node^)
        elif ch == ord("$"):
            self.pos += 1
            var node = ASTNode.anchor(AnchorKind.EOL)
            return self.ast.add_node(node^)
        elif ch == ord("*") or ch == ord("+") or ch == ord("?"):
            raise Error(
                String.write(
                    RegexError(
                        "Quantifier without preceding element",
                        self.pos,
                    )
                )
            )
        elif ch == ord(")"):
            raise Error(
                String.write(
                    RegexError("Unmatched ')'", self.pos)
                )
            )
        else:
            self.pos += 1
            var node = ASTNode.literal(UInt32(ch))
            return self.ast.add_node(node^)

    def _parse_group(mut self) raises -> Int:
        """Parse a group: (regex), (?:regex), (?=), (?!), (?<=), (?<!), (?P<name>)."""
        self.pos += 1  # consume '('

        var group_index = -1  # -1 = non-capturing by default

        # Check for group modifiers
        if not self._at_end() and self._peek() == ord("?"):
            self.pos += 1  # consume '?'
            if self._at_end():
                raise Error(
                    String.write(
                        RegexError("Unexpected end of pattern after '(?'", self.pos)
                    )
                )
            var modifier = self._peek()
            if modifier == ord(":"):
                self.pos += 1  # consume ':'
                # Non-capturing group — group_index stays -1
            elif modifier == ord("="):
                self.pos += 1  # consume '='
                var inner = self._parse_alternation()
                self._expect(ord(")"))
                var node = ASTNode.lookahead(inner, False)
                return self.ast.add_node(node^)
            elif modifier == ord("!"):
                self.pos += 1  # consume '!'
                var inner = self._parse_alternation()
                self._expect(ord(")"))
                var node = ASTNode.lookahead(inner, True)
                return self.ast.add_node(node^)
            elif modifier == ord("<"):
                self.pos += 1  # consume '<'
                if self._at_end():
                    raise Error(
                        String.write(
                            RegexError("Unexpected end after '(?<'", self.pos)
                        )
                    )
                var next_ch = self._peek()
                if next_ch == ord("="):
                    self.pos += 1  # consume '='
                    var inner = self._parse_alternation()
                    self._expect(ord(")"))
                    var node = ASTNode.lookbehind(inner, False)
                    return self.ast.add_node(node^)
                elif next_ch == ord("!"):
                    self.pos += 1  # consume '!'
                    var inner = self._parse_alternation()
                    self._expect(ord(")"))
                    var node = ASTNode.lookbehind(inner, True)
                    return self.ast.add_node(node^)
                else:
                    raise Error(
                        String.write(
                            RegexError(
                                "Unknown lookbehind modifier '(?<" + chr(next_ch) + "'",
                                self.pos - 2,
                            )
                        )
                    )
            elif modifier == ord("P"):
                self.pos += 1  # consume 'P'
                self._expect(ord("<"))
                var name = self._parse_group_name()
                self._expect(ord(">"))
                self.ast.group_count += 1
                group_index = self.ast.group_count
                self.ast.group_names[name^] = group_index
            elif modifier == ord("i") or modifier == ord("m") or modifier == ord("s"):
                # Inline flags: (?i), (?m), (?s) or (?i:...) etc.
                var inline_flags = self._parse_inline_flags()
                if not self._at_end() and self._peek() == ord(")"):
                    # (?ims) — set flags globally
                    self.pos += 1  # consume ')'
                    self.ast.flags = RegexFlags(self.ast.flags.value | inline_flags.value)
                    # Return empty concat node
                    var node = ASTNode.concat(List[Int]())
                    return self.ast.add_node(node^)
                elif not self._at_end() and self._peek() == ord(":"):
                    self.pos += 1  # consume ':'
                    # (?ims:...) — scoped flags, treat as non-capturing group
                    # Flags are set globally for now (scoped flags would need
                    # per-node flag tracking)
                    self.ast.flags = RegexFlags(self.ast.flags.value | inline_flags.value)
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
                            "Unknown group modifier '(?" + chr(modifier) + "'",
                            self.pos - 1,
                        )
                    )
                )
        else:
            # Capturing group
            self.ast.group_count += 1
            group_index = self.ast.group_count

        var inner = self._parse_alternation()
        self._expect(ord(")"))

        if group_index == -1:
            # Non-capturing: just return the inner node directly
            return inner

        var node = ASTNode.group(inner, group_index)
        return self.ast.add_node(node^)

    def _parse_inline_flags(mut self) -> RegexFlags:
        """Parse inline flag characters (i, m, s) and return the flags."""
        var flags = RegexFlags()
        while not self._at_end():
            var ch = self._peek()
            if ch == ord("i"):
                flags = RegexFlags(flags.value | RegexFlags.IGNORECASE)
                self.pos += 1
            elif ch == ord("m"):
                flags = RegexFlags(flags.value | RegexFlags.MULTILINE)
                self.pos += 1
            elif ch == ord("s"):
                flags = RegexFlags(flags.value | RegexFlags.DOTALL)
                self.pos += 1
            else:
                break
        return flags^

    def _parse_group_name(mut self) raises -> String:
        """Parse a group name (letters, digits, underscores)."""
        var start = self.pos
        while not self._at_end() and self._peek() != ord(">"):
            var ch = self._peek()
            if not (
                (ch >= ord("a") and ch <= ord("z"))
                or (ch >= ord("A") and ch <= ord("Z"))
                or (ch >= ord("0") and ch <= ord("9"))
                or ch == ord("_")
            ):
                raise Error(
                    String.write(
                        RegexError("Invalid character in group name", self.pos)
                    )
                )
            self.pos += 1
        if self.pos == start:
            raise Error(
                String.write(
                    RegexError("Empty group name", self.pos)
                )
            )
        return String(self.pattern[byte=start : self.pos])

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
        if ch == ord("b"):
            var node = ASTNode.anchor(AnchorKind.WORD_BOUNDARY)
            return self.ast.add_node(node^)
        elif ch == ord("B"):
            var node = ASTNode.anchor(AnchorKind.NOT_WORD_BOUNDARY)
            return self.ast.add_node(node^)

        # Backreferences \1 through \9
        if ch >= ord("1") and ch <= ord("9"):
            var group = ch - ord("0")
            var node = ASTNode.backreference(group)
            return self.ast.add_node(node^)

        # Shorthand character classes
        if ch == ord("d") or ch == ord("D"):
            var cs = CharSet.digit()
            if ch == ord("D"):
                cs.negate()
            cs.build_bitmap()
            var cs_idx = self.ast.add_charset(cs^)
            var node = ASTNode.char_class(cs_idx, ch == ord("D"))
            return self.ast.add_node(node^)
        elif ch == ord("w") or ch == ord("W"):
            var cs = CharSet.word()
            if ch == ord("W"):
                cs.negate()
            cs.build_bitmap()
            var cs_idx = self.ast.add_charset(cs^)
            var node = ASTNode.char_class(cs_idx, ch == ord("W"))
            return self.ast.add_node(node^)
        elif ch == ord("s") or ch == ord("S"):
            var cs = CharSet.whitespace()
            if ch == ord("S"):
                cs.negate()
            cs.build_bitmap()
            var cs_idx = self.ast.add_charset(cs^)
            var node = ASTNode.char_class(cs_idx, ch == ord("S"))
            return self.ast.add_node(node^)

        # Literal character escapes
        if ch == ord("t"):
            var node = ASTNode.literal(UInt32(ord("\t")))
            return self.ast.add_node(node^)
        elif ch == ord("n"):
            var node = ASTNode.literal(UInt32(ord("\n")))
            return self.ast.add_node(node^)
        elif ch == ord("r"):
            var node = ASTNode.literal(UInt32(ord("\r")))
            return self.ast.add_node(node^)

        # Metacharacter escapes
        if (
            ch == ord("\\")
            or ch == ord(".")
            or ch == ord("*")
            or ch == ord("+")
            or ch == ord("?")
            or ch == ord("[")
            or ch == ord("]")
            or ch == ord("(")
            or ch == ord(")")
            or ch == ord("|")
            or ch == ord("{")
            or ch == ord("}")
            or ch == ord("^")
            or ch == ord("$")
        ):
            var node = ASTNode.literal(UInt32(ch))
            return self.ast.add_node(node^)

        raise Error(
            String.write(
                RegexError(
                    "Invalid escape sequence '\\" + chr(ch) + "'",
                    self.pos - 2,
                )
            )
        )

    def _parse_char_class(mut self) raises -> Int:
        """Parse a character class: [abc], [a-z], [^abc], etc."""
        self.pos += 1  # consume '['
        var negated = False
        if not self._at_end() and self._peek() == ord("^"):
            negated = True
            self.pos += 1

        var cs = CharSet()

        # Handle ']' as first character in class (literal)
        if not self._at_end() and self._peek() == ord("]"):
            cs.add_range(UInt32(ord("]")), UInt32(ord("]")))
            self.pos += 1

        while not self._at_end() and self._peek() != ord("]"):
            var ch = self._advance()
            if ch == ord("\\"):
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
                if esc == ord("d"):
                    self.pos += 1
                    cs.add_range(UInt32(ord("0")), UInt32(ord("9")))
                    continue
                elif esc == ord("D"):
                    # \D inside a class is complex — skip for now, just add ranges
                    self.pos += 1
                    cs.add_range(0, UInt32(ord("0")) - 1)
                    cs.add_range(UInt32(ord("9")) + 1, 127)
                    continue
                elif esc == ord("w"):
                    self.pos += 1
                    cs.add_range(UInt32(ord("a")), UInt32(ord("z")))
                    cs.add_range(UInt32(ord("A")), UInt32(ord("Z")))
                    cs.add_range(UInt32(ord("0")), UInt32(ord("9")))
                    cs.add_range(UInt32(ord("_")), UInt32(ord("_")))
                    continue
                elif esc == ord("W"):
                    # \W inside class — hard to represent, skip detailed handling
                    self.pos += 1
                    continue
                elif esc == ord("s"):
                    self.pos += 1
                    cs.add_range(UInt32(ord(" ")), UInt32(ord(" ")))
                    cs.add_range(UInt32(ord("\t")), UInt32(ord("\t")))
                    cs.add_range(UInt32(ord("\n")), UInt32(ord("\n")))
                    cs.add_range(UInt32(ord("\r")), UInt32(ord("\r")))
                    cs.add_range(0x0B, 0x0B)
                    cs.add_range(0x0C, 0x0C)
                    continue
                elif esc == ord("S"):
                    self.pos += 1
                    continue
                elif esc == ord("t"):
                    self.pos += 1
                    ch = ord("\t")
                elif esc == ord("n"):
                    self.pos += 1
                    ch = ord("\n")
                elif esc == ord("r"):
                    self.pos += 1
                    ch = ord("\r")
                else:
                    ch = self._advance()

            # Check for range: a-z
            if not self._at_end() and self._peek() == ord("-"):
                # Peek ahead to see if this is a range or a literal '-' at end
                if self.pos + 1 < len(self.pattern) and Int(self.pattern.as_bytes()[self.pos + 1]) != ord("]"):
                    self.pos += 1  # consume '-'
                    var hi_ch = self._advance()
                    if hi_ch == ord("\\"):
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
        var node = ASTNode.char_class(cs_idx, negated)
        return self.ast.add_node(node^)


def parse(pattern: String) raises -> AST:
    """Parse a regex pattern string into an AST."""
    var p = Parser(pattern)
    return p.parse()
