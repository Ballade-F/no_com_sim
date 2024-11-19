import numpy as np
import matplotlib.pyplot as plt
import heapq
import time as TM

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
    
    def __lt__(self, other):
        return self.f < other.f

    def reset(self):
        self.g = float('inf')
        self.h = 0
        self.f = float('inf')
        self.parent = None
        self.set_flag = 0

class AStarPlanner():
    def __init__(self, grid_map:np.ndarray, resolution_x:float, resolution_y:float):
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
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
        return np.sqrt(((node1.x - node2.x)*self.resolution_x) ** 2 + ((node1.y - node2.y)*self.resolution_y) ** 2)
        # return np.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)

    def _heuristic(self, node:astar_node, goal:astar_node):
        return self._get_distance(node, goal)
        # return 0
    
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
                    # consider resolution
                    distance = np.sqrt(self.resolution_x**2+self.resolution_y**2)
                    if i !=0 and j == 0:
                        distance = self.resolution_x
                    elif i == 0 and j != 0:
                        distance = self.resolution_y
                    neighbors.append((x, y, distance))
        return neighbors
    
    def true2grid(self, point)->tuple:
        x_index = int(point[0]/self.resolution_x)
        y_index = int(point[1]/self.resolution_y)
        x_index = max(0, min(x_index, self.n_x-1))
        y_index = max(0, min(y_index, self.n_y-1))
        return (x_index, y_index)
    
    def grid2true(self, point)->tuple:
        x = (point[0] + 0.5) * self.resolution_x
        x = max(0, min(x, self.n_x*self.resolution_x))
        y = (point[1] + 0.5) * self.resolution_y
        y = max(0, min(y, self.n_y*self.resolution_y))
        return (x, y)

    #return the path from start to goal and the distance
    
    def plan(self, start:tuple, goal:tuple, reset_nodes=True, grid_mode=True, path_flag=True):
        '''
        @param start: the start point of the path
        @param goal: the goal point of the path
        @param reset_nodes: if reset the nodes g, h, f, parent, set_flag
        @param grid_mode: if the start and goal are in grid mode, if not, please input points in true coordinate
        @param path_flag: if return the path and distance, if not, only return the distance
        '''
        if reset_nodes:
            self.resetNodes()

        start_idx = None
        goal_idx = None
        if not grid_mode:
            start_idx = self.true2grid(start)
            goal_idx = self.true2grid(goal)
        else:
            start_idx = start
            goal_idx = goal

        # the = in python equal to the reference in c++
        start_node = self.grid_nodes[start_idx[0]][start_idx[1]]
        goal_node = self.grid_nodes[goal_idx[0]][goal_idx[1]]
        start_node.g = 0
        start_node.h = self._heuristic(start_node, goal_node)
        start_node.f = start_node.g + start_node.h
        open_list = []
        # heapq is a priority queue
        # push the start node into the open list
        heapq.heappush(open_list,  start_node)
        start_node.set_flag = 1

        while len(open_list) > 0:
            # pop the node with the smallest f value, which means put it into the close list
            #debug
            # print([(p.x,p.y,p.f) for p in open_list])
            current_node = heapq.heappop(open_list)
            # current_node = open_list.pop(0)
            # if the node is in the close list, skip
            if current_node.set_flag == 2:
                continue
            current_node.set_flag = 2
            # if the current node is the goal node, return the path and distance
            if current_node == goal_node:
                if path_flag == False:
                    return goal_node.g
                path = []
                while current_node is not None:
                    if grid_mode:
                        path.append((current_node.x, current_node.y))
                    else:
                        path.append(self.grid2true((current_node.x, current_node.y)))
                    current_node = current_node.parent
                return path[::-1], goal_node.g
            # get the neighbors of the current node
            neighbors = self._get_neighbors(current_node)
            for neighbor in neighbors:
                x, y, distance = neighbor
                neighbor_node = self.grid_nodes[x][y]
                # heappush cannot cover the same node, so we need to check if the neighbor node is in the close list
                if neighbor_node.set_flag == 2:
                    continue
                tentative_g = current_node.g + distance
                # if the neighbor node is not in the open list, put it into the open list
                if neighbor_node.set_flag == 0:
                    neighbor_node.g = tentative_g
                    neighbor_node.h = self._heuristic(neighbor_node, goal_node)
                    neighbor_node.f = neighbor_node.g + neighbor_node.h
                    neighbor_node.parent = current_node
                    heapq.heappush(open_list, neighbor_node)
                    neighbor_node.set_flag = 1
                # if the neighbor node is in the open list, update the g value
                elif tentative_g < neighbor_node.g:
                    
                    neighbor_node.g = tentative_g
                    neighbor_node.f = neighbor_node.g + neighbor_node.h
                    neighbor_node.parent = current_node
                    # heapq.heappush(open_list, neighbor_node)
                    heapq.heapify(open_list)
                
        if path_flag == False:
            return goal_node.g
        return [(start[0], start[1])], self.n_x*self.n_y*(self.resolution_x+self.resolution_y)

    
if __name__ == '__main__':
    grid_map = np.zeros((100, 100))
    grid_map[20:80, 20:80] = 1
    grid_map[10:40, 25:75] = 0
    grid_map[60:90, 25:75] = 0
    astar_planner = AStarPlanner(grid_map,0.1,0.1)

    time_1 = TM.time()
    path, dis = astar_planner.plan((1, 1), (95, 95),False)
    time_2 = TM.time()
    print('time: ', time_2 - time_1)

    img = 255-grid_map*255
    img = img.transpose()
    print(dis)
    # print(path)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.plot([x[0] for x in path], [x[1] for x in path], 'r')
    plt.show()