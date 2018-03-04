import json

from tests.base import TestCase

from suplearn_clone_detection import ast


class AstTest(TestCase):
    ast_nodes = [{"type": "root", "children": [1, 2]},
                 {"type": "child1", "children": [3]},
                 {"type": "child2"},
                 {"type": "grand-child1"}]

    @classmethod
    def setUpClass(cls):
        with open(cls.fixture_path("asts.json")) as f:
            cls.asts = [json.loads(v) for v in f if v]

    def test_from_list_without_value(self):
        list_ast = [{"type": "foo"}]
        root = ast.from_list(list_ast)
        self.assertEqual(root.type, "foo")
        self.assertIsNone(root.value)
        self.assertEqual(len(root.children), 0)

    def test_from_list_with_value(self):
        list_ast = [{"type": "foo", "value": "bar"}]
        root = ast.from_list(list_ast)
        self.assertEqual(root.type, "foo")
        self.assertEqual(root.value, "bar")
        self.assertEqual(len(root.children), 0)

    def test_from_list_recursive(self):
        list_ast = [{"type": "foo", "value": "bar", "children": [1]},
                    {"type": "baz"}]
        root = ast.from_list(list_ast)
        self.assertEqual(root.type, "foo")
        self.assertEqual(root.value, "bar")
        self.assertEqual(len(root.children), 1)
        child = root.children[0]
        self.assertEqual(child.type, "baz")

    def test_from_list_complex(self):
        list_ast = self.asts[0]
        root = ast.from_list(list_ast)
        self.assertEqual(root.type, "CompilationUnit")

    def test_bfs(self):
        root = ast.from_list(self.ast_nodes)
        bfs_types = [node["type"] for node in root.bfs()]
        expected = ["root", "child1", "child2", "grand-child1"]
        self.assertEqual(expected, bfs_types)

    def test_dfs(self):
        root = ast.from_list(self.ast_nodes)
        dfs_types = [node["type"] for node in root.dfs()]
        expected = ["root", "child1", "grand-child1", "child2"]
        self.assertEqual(expected, dfs_types)

    def test_dfs_reverse(self):
        root = ast.from_list(self.ast_nodes)
        dfs_types = [node["type"] for node in root.dfs(reverse=True)]
        expected = ["root", "child2", "child1", "grand-child1"]
        self.assertEqual(expected, dfs_types)

    def _load_list_ast(self):
        with open(self.fixture_path("asts.json")) as f:
            return json.loads(next(f))
