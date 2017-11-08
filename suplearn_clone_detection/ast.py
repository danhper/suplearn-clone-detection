class Node:
    def __init__(self, node_id, node_type, value=None):
        self.node_id = node_id
        self.type = node_type
        self.value = value
        self.children = []

    def dfs(self, reverse=False):
        children = reversed(self.children) if reverse else self.children
        yield self
        for child in children:
            yield from child.dfs(reverse)


def from_list(list_ast):
    def create_node(index):
        node_info = list_ast[index]
        node = Node(node_info.get("id"), node_info["type"], node_info.get("value"))
        for child_index in node_info.get("children", []):
            node.children.append(create_node(child_index))
        return node
    return create_node(0)
