from time import sleep
import numpy as np
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

#test reshape
# a = np.array([[1, 2], [3, 4]])
# print(a)
# print(a.reshape(-1, 1))

# start_node = astar_node(0,0)
# goal_node = astar_node(10,10)
# start_node.g = 0
# start_node.h = 0
# start_node.f = 0
# open_list = []
# # heapq is a priority queue
# # push the start node into the open list
# heapq.heappush(open_list,  start_node)
# start_node.set_flag = 1
# start_node.f = 10
# print(open_list[0].f)
# heapq


def _task(a,b):
    print(a)
    sleep(2)
    print(b)
    return a+b


def test_multiprocess():
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = []
        for i in range(8):
            futures.append(executor.submit(_task, i, i+10))
        for future in futures:
            temp = future.result()
            print(temp)
    print('done')


import logging
import time

def test_logging():
    # logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(filename='/home/users/wzr/project/no_com_sim/log/time_{}.log'.format(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())),
                         level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.debug('This is a debug message')
    logging.info('This is an info message')
    logging.warning('This is a warning message')
    logging.error('This is an error message')
    logging.critical('This is a critical message')
 

if __name__ == '__main__':
    # test_multiprocess()
    test_logging()