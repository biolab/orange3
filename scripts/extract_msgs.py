import os

import libcst as cst
from libcst.metadata import PositionProvider

class StringCollector(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, ranges):
        self.module = None
        self.function_stack = []
        self.collection = []
        self.ranges = ranges

    @property
    def outside(self):
        if not self.function_stack:
            return []
        return self.function_stack[-1].body.children

    def visit_Module(self, node):
        self.module = node

    def push_context(self, node):
        blacklist = [
            child.body[0].value
            for child in node.body.children
            if isinstance(child, cst.SimpleStatementLine)
                and len(child.body) == 1
                and isinstance(child.body[0], cst.Expr)
                and isinstance(child.body[0].value, cst.SimpleString)]
        self.function_stack.append((node, blacklist))

    def pop_context(self):
        self.function_stack.pop()

    def blacklisted(self, node):
        return self.function_stack and node in self.function_stack[-1][1]

    def visit_ClassDef(self, node: cst.ClassDef):
        self.push_context(node)

    def leave_ClassDef(self, _):
        self.pop_context()

    def visit_FunctionDef(self, node: cst.FunctionDef):
        self.push_context(node)

    def leave_FunctionDef(self, _):
        self.pop_context()

    def visit_FormattedString(self, node: cst.FormattedStringExpression):
        if not self.blacklisted(node):
            self.collection.append((self.ranges[node].start.line, "",
                                    self.module.code_for_node(node)[2:-1], ""))

    def visit_SimpleString(self, node: cst.SimpleString):
        s = self.module.code_for_node(node)[1:-1]
        if s and not self.blacklisted(node):
            self.collection.append((self.ranges[node].start.line, "", s, ""))


def extract(source, _, comment_tags, options=None):
    if os.path.basename(source.name).startswith("test_"):
        return
    source_tree = cst.metadata.MetadataWrapper(cst.parse_module(source.read()))
    collector = StringCollector(source_tree.resolve(cst.metadata.PositionProvider))
    source_tree.visit(collector)
    yield from iter(collector.collection)
