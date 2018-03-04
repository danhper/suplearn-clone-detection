class Node:
    def __init__(self, node_id, node_type, value=None):
        self.node_id = node_id
        self.type = node_type
        self.value = value
        self.children = []

    def get(self, key):
        return getattr(self, key)

    def __getitem__(self, key):
        return self.get(key)

    def dfs(self, reverse=False):
        children = reversed(self.children) if reverse else self.children
        yield self
        for child in children:
            yield from child.dfs(reverse)

    def bfs(self):
        queue = [self]
        while queue:
            node = queue.pop()
            yield node
            for child in node.children:
                queue.insert(0, child)

def from_list(list_ast):
    def create_node(index):
        node_info = list_ast[index]
        node = Node(node_info.get("id"), node_info["type"], node_info.get("value"))
        for child_index in node_info.get("children", []):
            node.children.append(create_node(child_index))
        return node
    return create_node(0)
