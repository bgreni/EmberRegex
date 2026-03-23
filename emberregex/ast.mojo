"""AST node types for parsed regex patterns.

Uses a flat-pool representation for cache friendliness. Each ASTNode
stores its kind as an integer tag, with kind-specific data in fields.
Children are referenced by index into the AST's node pool.
"""

from .charset import CharSet
from .flags import RegexFlags


struct ASTNodeKind(Copyable, Movable):
    """Constants for AST node types."""

    comptime LITERAL = 0
    comptime DOT = 1
    comptime CHAR_CLASS = 2
    comptime ALTERNATION = 3
    comptime CONCAT = 4
    comptime QUANTIFIER = 5
    # Future milestones:
    comptime GROUP = 6
    comptime ANCHOR = 7
    comptime LOOKAHEAD = 8
    comptime LOOKBEHIND = 9
    comptime BACKREFERENCE = 10


struct AnchorKind:
    """Constants for anchor types."""

    comptime BOL = 0  # Beginning of line/string (^)
    comptime EOL = 1  # End of line/string ($)
    comptime WORD_BOUNDARY = 2  # \b
    comptime NOT_WORD_BOUNDARY = 3  # \B
    comptime BOL_MULTILINE = 4  # ^ with MULTILINE — matches after \n too
    comptime EOL_MULTILINE = 5  # $ with MULTILINE — matches before \n too


struct ASTNode(Copyable, Movable):
    """A single node in the regex AST."""

    var kind: Int
    var char_value: UInt32  # For LITERAL
    var quantifier_min: Int  # For QUANTIFIER
    var quantifier_max: Int  # -1 = unbounded
    var greedy: Bool
    var group_index: Int  # For GROUP (-1 = non-capturing)
    var charset_index: Int  # Index into AST's charset pool (-1 = none)
    var children: List[Int]  # Indices into AST node pool
    var negated: Bool
    var anchor_type: Int  # For ANCHOR

    def __init__(out self, kind: Int):
        self.kind = kind
        self.char_value = 0
        self.quantifier_min = 0
        self.quantifier_max = 0
        self.greedy = True
        self.group_index = -1
        self.charset_index = -1
        self.children = List[Int]()
        self.negated = False
        self.anchor_type = -1

    @staticmethod
    def literal(ch: UInt32, out node: ASTNode):
        node = ASTNode(ASTNodeKind.LITERAL)
        node.char_value = ch

    @staticmethod
    def dot() -> ASTNode:
        return ASTNode(ASTNodeKind.DOT)

    @staticmethod
    def char_class(charset_idx: Int, negated: Bool, out node: ASTNode):
        node = ASTNode(ASTNodeKind.CHAR_CLASS)
        node.charset_index = charset_idx
        node.negated = negated

    @staticmethod
    def alternation(var children: List[Int], out node: ASTNode):
        node = ASTNode(ASTNodeKind.ALTERNATION)
        node.children = children^

    @staticmethod
    def concat(var children: List[Int], out node: ASTNode):
        node = ASTNode(ASTNodeKind.CONCAT)
        node.children = children^

    @staticmethod
    def quantifier(child: Int, min_rep: Int, max_rep: Int, greedy: Bool, out node: ASTNode):
        node = ASTNode(ASTNodeKind.QUANTIFIER)
        node.children = [child]
        node.quantifier_min = min_rep
        node.quantifier_max = max_rep
        node.greedy = greedy

    @staticmethod
    def group(child: Int, group_index: Int, out node: ASTNode):
        """Create a group node. group_index=-1 for non-capturing."""
        node = ASTNode(ASTNodeKind.GROUP)
        node.children = [child]
        node.group_index = group_index

    @staticmethod
    def anchor(anchor_type: Int, out node: ASTNode) :
        node = ASTNode(ASTNodeKind.ANCHOR)
        node.anchor_type = anchor_type

    @staticmethod
    def lookahead(child: Int, negated: Bool, out node: ASTNode):
        node = ASTNode(ASTNodeKind.LOOKAHEAD)
        node.children = [child]
        node.negated = negated

    @staticmethod
    def lookbehind(child: Int, negated: Bool, out node: ASTNode):
        node = ASTNode(ASTNodeKind.LOOKBEHIND)
        node.children = [child]
        node.negated = negated

    @staticmethod
    def backreference(group_index: Int, out node: ASTNode):
        node = ASTNode(ASTNodeKind.BACKREFERENCE)
        node.group_index = group_index


struct AST(Movable):
    """The complete AST for a parsed regex pattern.

    Owns a pool of ASTNode values and a pool of CharSet values.
    Nodes reference children and charsets by index.
    """

    var nodes: List[ASTNode]
    var charsets: List[CharSet]
    var root: Int
    var group_count: Int
    var group_names: Dict[String, Int]
    var flags: RegexFlags  # Inline flags extracted from the pattern (e.g. (?i), (?m), (?s))

    def __init__(out self):
        self.nodes = List[ASTNode]()
        self.charsets = List[CharSet]()
        self.root = -1
        self.group_count = 0
        self.group_names = Dict[String, Int]()
        self.flags = RegexFlags()

    def add_node(mut self, var node: ASTNode) -> Int:
        """Add a node to the pool and return its index."""
        var idx = len(self.nodes)
        self.nodes.append(node^)
        return idx

    def add_charset(mut self, var cs: CharSet) -> Int:
        """Add a charset to the pool and return its index."""
        var idx = len(self.charsets)
        self.charsets.append(cs^)
        return idx
