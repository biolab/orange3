import os
import sys

import yaml

import libcst as cst
from libcst.metadata import PositionProvider


class StringCollector(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self):
        self.module = None
        self.function_stack = []
        self.ranges = {}
        self.contexts = [{}]

    def open_module(self, tree, name):
        self.ranges = self.resolve(tree)
        self.module_name = name

    @property
    def outside(self):
        if not self.function_stack:
            return []
        return self.function_stack[-1].body.children

    def visit_Module(self, node):
        self.module = node
        self.push_context(node, self.module_name)

    def leave_Module(self, _):
        self.pop_context()

    def push_context(self, node, name=None):
        blacklist = [
            child.body[0].value
            for child in (node.body if isinstance(node, cst.Module)
                          else node.body.children)
            if isinstance(child, cst.SimpleStatementLine)
                and len(child.body) == 1
                and isinstance(child.body[0], cst.Expr)
                and isinstance(child.body[0].value, cst.SimpleString)]
        self.function_stack.append((node, blacklist, name))
        self.contexts.append({})

    def pop_context(self):
        node, _, name = self.function_stack.pop()
        context = self.contexts.pop()
        if context:
            self.contexts[-1][name or node.name.value] = context

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
            self.contexts[-1][self.module.code_for_node(node)[2:-1]] = ""

    def visit_SimpleString(self, node: cst.SimpleString):
        s = self.module.code_for_node(node)[1:-1]
        if s and not self.blacklisted(node):
            self.contexts[-1][s] = ""


class StringTranslator(cst.CSTTransformer):
    def __init__(self, contexts):
        self.module = None
        self.context_stack = [contexts]

    @property
    def context(self):
        return self.context_stack[-1]

    def open_module(self, node, name):
        self.module = node
        self.push_context(node, name)

    def push_context(self, node, name=None):
        self.context_stack.append(self.context.get(name or node.name.value, {}))

    def pop_context(self):
        self.context_stack.pop()

    def __visit(self, node):
        self.push_context(node)

    def __leave(self, _, updated_node):
        self.pop_context()
        return updated_node

    def __translate(self, node, updated_node, pref):
        original = self.module.code_for_node(node)[1 + len(pref):-1]
        translation = self.context.get(original)
        if not translation:
            return updated_node
        return cst.parse_expression(f'{pref}"{translation}"')

    visit_ClassDef = __visit
    visit_FunctionDef = __visit

    leave_ClassDef = __leave
    leave_FunctionDef = __leave

    def leave_FormattedString(self,
            node: cst.FormattedStringExpression,
            updated_node: cst.FormattedStringExpression):
        return self.__translate(node, updated_node, "f")

    def leave_SimpleString(
            self, node: cst.SimpleString, updated_node: cst.SimpleString):
        return self.__translate(node, updated_node, "")


def collect_messages(outf):
    collector = StringCollector()
    for root, _, files in os.walk("Orange/widgets"):
        for name in files:
            if name.startswith("test_") or os.path.splitext(name)[1] != ".py":
                continue
            fullname = os.path.join(root, name)
            print(f"parsing {fullname}")
            with open(fullname) as f:
                tree = cst.metadata.MetadataWrapper(cst.parse_module(f.read()))
                collector.open_module(tree, fullname)
                tree.visit(collector)

    with open(outf, "wt") as f:
        f.write(yaml.dump(collector.contexts[0], indent=4))


def translate(translations):
    translator = StringTranslator(translations)
    for root, _, files in os.walk("Orange/widgets"):
        for name in files:
            fullname = os.path.join(root, name)
            # Provisional: only translates owtable.py into owtable2.py
            if fullname != "Orange/widgets/data/owtable.py":
                continue
            with open(fullname) as f:
                tree = cst.parse_module(f.read())
            translator.open_module(tree, fullname)
            translated = tree.visit(translator)
            with open(fullname[:-3] + "2.py", "wt") as f:
                f.write(tree.code_for_node(translated))


# Extract messages:
# messages = collect_messages(sys.argv[1])

# Translate:
# messages = yaml.load(open(sys.argv[1]))
# translate(messages)
