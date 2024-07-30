import numpy as np
import matplotlib.pyplot as plt
import map

class astar_node():
    def __init__(self, x:int, y:int):
        self.x = x
        self.y = y
        self.g = float('inf')
        self.h = 0
        self.f = float('inf')
        self.parent = None
        # flag for open list and close list
        self.set_flag = 0

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def reset(self):
        self.g = float('inf')
        self.h = 0
        self.f = float('inf')
        self.parent = None
        self.set_flag = 0

class AStarPlanner():
    def __init__(self, grid_map:np.ndarray):
        self.grid_map = grid_map
        self.n_x = grid_map.shape[0]
        self.n_y = grid_map.shape[1]
        self.grid_nodes = [[None for _ in range(self.n_y)] for _ in range(self.n_x)]
        for i in range(self.n_x):
            for j in range(self.n_y):
                self.grid_nodes[i][j] = astar_node(i, j)

    def resetNodes(self):
        for i in range(self.n_x):
            for j in range(self.n_y):
                self.grid_nodes[i][j].reset()

    def _get_distance(self, node1:astar_node, node2:astar_node):
        return np.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)

    def _heuristic(self, node:astar_node, goal:astar_node):
        return self._get_distance(node, goal)
    
    def _is_valid_node(self, x:int, y:int):
        if x < 0 or x >= self.n_x or y < 0 or y >= self.n_y:
            return False
        if self.grid_map[x, y] == 1:
            return False
        return True

    def _get_neighbors(self, node:astar_node):
        neighbors = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                x = node.x + i
                y = node.y + j
                if self._is_valid_node(x, y):
                    distance = 1
                    if i != 0 and j != 0:
                        distance = np.sqrt(2)
                    neighbors.append((x, y, distance))
        return neighbors

    def plan(self, start:tuple, goal:tuple):
        # the = in python equal to the reference in c++
        start_node = self.grid_nodes[start[0]][start[1]]
        goal_node = self.grid_nodes[goal[0]][goal[1]]
        start_node.g = 0
        start_node.h = self._heuristic(start_node, goal_node)
        start_node.f = start_node.g + start_node.h
        open_list = []
        open_list.append(start_node)
        start_node.set_flag = 1

        while len(open_list) > 0:
            current_node = open_list.pop(0)
            current_node.set_flag = 2
            if current_node == goal_node:
                path = []
                while current_node is not None:
                    path.append((current_node.x, current_node.y))
                    current_node = current_node.parent
                return path[::-1]
            neighbors = self._get_neighbors(current_node)
            for neighbor in neighbors:
                x, y, distance = neighbor
                neighbor_node = self.grid_nodes[x][y]
                if neighbor_node.set_flag == 2:
                    continue
                tentative_g = current_node.g + distance
                if neighbor_node.set_flag == 0:
                    neighbor_node.g = tentative_g
                    neighbor_node.h = self._heuristic(neighbor_node, goal_node)
                    neighbor_node.f = neighbor_node.g + neighbor_node.h
                    neighbor_node.parent = current_node
                    heapq.heappush(open_list, (neighbor_node.f, neighbor_node))
                    neighbor_node.set_flag = 1
                elif tentative_g < neighbor_node.g:
                    neighbor_node.g = tentative_g
                    neighbor_node.f = neighbor_node.g + neighbor_node.h
                    neighbor_node.parent = current_node
                    heapq.heappush(open_list, (neighbor_node.f, neighbor_node))
                

        return None

    