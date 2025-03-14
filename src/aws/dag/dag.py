"""
dag.py - Directed Acyclic Graph (DAG) Concepts and Examples

This module provides an in-depth explanation of Directed Acyclic Graphs (DAGs),
their characteristics, and applications in different domains, along with illustrative
examples and algorithms.

A Directed Acyclic Graph (DAG) is a graph structure consisting of nodes and directed
edges, with the following properties:
1. The graph is directed (edges have a direction).
2. The graph is acyclic (contains no cycles).

DAGs are commonly used in various fields, such as:
- Workflow management (e.g., Apache Airflow)
- Blockchain (e.g., IOTA)
- Compiler design (e.g., optimizing expression trees)
- Data processing pipelines
- Machine learning pipelines
- Git version control system

==================================================================
# Key Properties:
- Directed edges represent a hierarchical or dependency structure.
- Cycles (circular dependencies) are not allowed, ensuring clear execution order.
- Supports parallelism by allowing independent nodes to be executed concurrently.

==================================================================
# Applications of DAGs:
1. Workflow orchestration (e.g., Apache Airflow, Luigi).
2. Compiler optimization.
3. Blockchain systems (e.g., IOTA).
4. Task scheduling.
5. Version control systems (e.g., git commit history).
6. Machine learning and data pipelines.

==================================================================
# Algorithms for DAG Processing:
1. Topological Sorting: Orders the nodes in such a way that all dependencies
   are satisfied before processing a node.
2. Shortest-Path Algorithms: Used on weighted DAGs to find the shortest or
   longest path efficiently.

==================================================================
"""

class DAG:
    """
    A basic implementation of a Directed Acyclic Graph (DAG).
    
    Attributes:
        nodes (dict): A dictionary where keys are node names and values are lists
                      of dependent child nodes (directed edges).
    
    Methods:
        add_node(node): Adds a node to the graph.
        add_edge(parent, child): Adds a directed edge from `parent` to `child`.
        topological_sort(): Performs topological sort on the DAG.
        has_cycle(): Checks whether the graph contains a cycle (shouldn't happen in a DAG).
    """

    def __init__(self):
        """Initialize an empty DAG."""
        self.nodes = {}

    def add_node(self, node):
        """
        Add a node to the graph.
        
        :param node: Name of the node (any hashable value).
        """
        if node not in self.nodes:
            self.nodes[node] = []

    def add_edge(self, parent, child):
        """
        Add a directed edge from `parent` to `child`.

        :param parent: The parent node (dependency).
        :param child: The child node (dependent).
        """
        if parent not in self.nodes:
            self.add_node(parent)
        if child not in self.nodes:
            self.add_node(child)
        self.nodes[parent].append(child)

    def has_cycle_util(self, node, visited, stack):
        """
        Utility function to check for cycles in the graph using Depth-First Search (DFS).
        
        :param node: The current node being visited.
        :param visited: A set of visited nodes.
        :param stack: A set of nodes in the recursion stack.
        :return: True if a cycle is found, otherwise False.
        """
        visited.add(node)
        stack.add(node)
        for neighbor in self.nodes[node]:
            if neighbor not in visited:
                if self.has_cycle_util(neighbor, visited, stack):
                    return True
            elif neighbor in stack:
                return True
        stack.remove(node)
        return False

    def has_cycle(self):
        """
        Check whether the graph contains a cycle.
        
        :return: True if the graph contains a cycle, otherwise False.
        """
        visited = set()
        stack = set()
        for node in self.nodes:
            if node not in visited:
                if self.has_cycle_util(node, visited, stack):
                    return True
        return False

    def topological_sort_util(self, node, visited, stack):
        """
        Utility function for topological sorting using Depth-First Search (DFS).
        
        :param node: The current node being visited.
        :param visited: A set of visited nodes.
        :param stack: A stack to store the topological order.
        """
        visited.add(node)
        for neighbor in self.nodes[node]:
            if neighbor not in visited:
                self.topological_sort_util(neighbor, visited, stack)
        stack.append(node)

    def topological_sort(self):
        """
        Perform a topological sort on the DAG.

        :return: A list representing the topological order of nodes.
        :raises: Exception if the graph contains a cycle.
        """
        if self.has_cycle():
            raise Exception("The graph contains a cycle, so topological sorting is not possible.")

        visited = set()
        stack = []
        for node in self.nodes:
            if node not in visited:
                self.topological_sort_util(node, visited, stack)
        return stack[::-1]  # Reverse the stack to get the correct order


def dag_example():
    """
    Example demonstrating the use of the `DAG` class.
    
    A small Directed Acyclic Graph with the following structure:
    
        A → B → C
        ↓
        D → E

    The expected topological order (one possible result): [A, D, B, E, C].

    :return: None
    """
    # Create DAG
    dag = DAG()
    dag.add_edge("A", "B")
    dag.add_edge("A", "D")
    dag.add_edge("B", "C")
    dag.add_edge("D", "E")

    print("Nodes in the graph:", dag.nodes)

    # Check for cycles
    if dag.has_cycle():
        print("The graph has a cycle.")
    else:
        print("The graph is acyclic.")

    # Perform topological sort
    try:
        order = dag.topological_sort()
        print("Topological order of nodes:", order)
    except Exception as e:
        print(str(e))


if __name__ == "__main__":
    """
    Run the example when the script is executed as a standalone program.
    """
    dag_example()
