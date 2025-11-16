# Data Processing README

Data processing Pipeline currently looks like:

Version 1:

1. Loading Data into pandas `DataFrame`
2. Based on code and language labels create tree-sitter `Tree` object
3. BFS through tree to create torch-geometric `Data` object and gather the following information about each node:
    - depth: at what depth is given node
    - num_children: how many children current node have
    - type_id: what is type of the given node (class, function, etc.)
    - is_named: whether the node is "named"
    - length: the length, how many characters does code block takes
