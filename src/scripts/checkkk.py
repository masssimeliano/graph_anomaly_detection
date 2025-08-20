import matplotlib.pyplot as plt
import networkx as nx

# Create a basic graph with 4 nodes
G = nx.Graph()

# Add nodes with multiple attributes
G.add_node(1, role="A", value=10)
G.add_node(2, role="A", value=12)
G.add_node(3, role="A", value=9)
G.add_node(4, role="B", value=300)

# Add edges (1-2, 1-3, 2-3) and (1-4)
G.add_edges_from([(1, 2), (1, 3), (2, 3), (1, 4)])

# Position nodes
pos = nx.spring_layout(G, seed=42)

plt.figure(figsize=(8, 8))

colors = ["red" if n == 4 else "skyblue" for n in G.nodes()]
# Draw nodes and edges
nx.draw(
    G,
    pos,
    with_labels=True,
    node_color=colors,
    node_size=1200,
    font_size=12,
    font_weight="bold",
)

# Draw attributes as separate text (column style)
offset_pos = {k: (v[0] + 0.015, v[1]) for k, v in pos.items()}
nx.draw_networkx_labels(
    G,
    offset_pos,
    labels={
        n: f"role: {d['role']}\nvalue: {d['value']}" for n, d in G.nodes(data=True)
    },
    font_size=9,
    font_color="black",
)

plt.title("Graph with Multiple Node Attributes Shown Beside Nodes", fontsize=14)
plt.axis("off")
plt.show()
