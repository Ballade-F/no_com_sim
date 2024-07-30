import numpy as np
import matplotlib.pyplot as plt
import heapq

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

    def reset(self):
        self.g = float('inf')
        self.h = 0
        self.f = float('inf')
        self.parent = None
        self.set_flag = 0

if __name__ == "__main__":
    # #test =
    # node0 = astar_node(1, 2)
    # node_test = node0
    # print(node0.g)
    # node_test.g = 1
    # print(node0.g)
    # a = 1
    # b = a
    # b = 5
    # print(a)

    #test heapq
    heap = []
    node_list = [astar_node(1, 2), astar_node(2, 3), astar_node(3, 4)]
    node_list[0].f = 1
    node_list[1].f = 2
    node_list[2].f = 3
    for node in node_list:
        heapq.heappush(heap, (node.f, node))
    print(heap)
    node_list[0].f = 4
    print(heap)
    heapq.heappush(heap, (node_list[0].f, node_list[0]))
    print(heap)