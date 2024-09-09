import numpy as np
from scipy.optimize import linear_sum_assignment

class Hungarian:
    def __init__(self, cost_matrix):
        self.cost_matrix = cost_matrix
        self.row_ind, self.col_ind = linear_sum_assignment(self.cost_matrix)
        self.total_cost = self.cost_matrix[self.row_ind, self.col_ind].sum()
        self.row_ind = self.row_ind.tolist()
        self.col_ind = self.col_ind.tolist()

    def get_row_ind(self):
        return self.row_ind

    def get_col_ind(self):
        return self.col_ind

    def get_total_cost(self):
        return self.total_cost
    
if __name__ == '__main__':
    cost_matrix = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
    hungarian = Hungarian(cost_matrix)
    print(hungarian.get_row_ind())
    print(hungarian.get_col_ind())
    print(hungarian.get_total_cost())
    cost_matrix = np.array([[4, 1, 3], [2, 0, 0]])
    hungarian = Hungarian(cost_matrix)
    print(hungarian.get_row_ind())
    print(hungarian.get_col_ind())
    print(hungarian.get_total_cost())
    cost_matrix = np.array([[4, 1], [2, 0], [0, 2]])
    hungarian = Hungarian(cost_matrix)
    print(hungarian.get_row_ind())
    print(hungarian.get_col_ind())
    print(hungarian.get_total_cost())