import matplotlib.pyplot as plt
import numpy as np

def draw_polygon(vertices):
    fig, ax = plt.subplots()
    polygon = plt.Polygon(vertices, closed=True, fill=None, edgecolor='r')
    ax.add_patch(polygon)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def fill_polygon(vertices):
    # Find the bounding box of the polygon
    min_x = min(vertices, key=lambda p: p[0])[0]
    max_x = max(vertices, key=lambda p: p[0])[0]
    min_y = min(vertices, key=lambda p: p[1])[1]
    max_y = max(vertices, key=lambda p: p[1])[1]

    # Create an edge table
    edge_table = []
    for i in range(len(vertices)):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % len(vertices)]
        if y1 != y2:  # Ignore horizontal edges
            if y1 > y2:
                x1, y1, x2, y2 = x2, y2, x1, y1
            edge_table.append([y1, y2, x1, (x2 - x1) / (y2 - y1)])

    # Sort edge table by the y-coordinate of the lower endpoint
    edge_table.sort()

    # Initialize the active edge table
    active_edge_table = []

    # Scanline fill
    for y in range(min_y, max_y + 1):
        # Add edges to the active edge table
        while edge_table and edge_table[0][0] == y:
            active_edge_table.append(edge_table.pop(0))

        # Remove edges from the active edge table
        active_edge_table = [edge for edge in active_edge_table if edge[1] != y]

        # Sort active edge table by x-coordinate
        active_edge_table.sort(key=lambda edge: edge[2])

        # Fill pixels between pairs of intersections
        for i in range(0, len(active_edge_table), 2):
            x_start = int(active_edge_table[i][2])
            x_end = int(active_edge_table[i + 1][2])
            for x in range(x_start, x_end):
                plt.plot(x+0.5, y-0.5, 'bo')

        # Update x-coordinates in the active edge table
        for edge in active_edge_table:
            edge[2] += edge[3]

    plt.xlim(min_x - 1, max_x + 1)
    plt.ylim(min_y - 1, max_y + 1)
    plt.gca().set_aspect('equal', adjustable='box')
    polygon = plt.Polygon(vertices, closed=True, fill=None, edgecolor='r')
    plt.gca().add_patch(polygon)
    plt.show()

# Define the vertices of the polygon
vertices = [(2, 1), (4, 5), (7, 8), (9, 4), (6, 2)]

# Draw the polygon
draw_polygon(vertices)

# Fill the polygon
fill_polygon(vertices)