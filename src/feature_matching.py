# This module is not used as the dtw that is implemented in here is very slow
import numpy as np

def normaldtw(template, target, window = 0):
    template_len, target_len = len(template), len(target)
    w = np.max([window, abs(template_len - target_len)])

    dtw_matrix = np.full((template_len + 1, target_len + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, template_len + 1):
        for j in range(np.max([1, i - w]), np.min([target_len, i + w]) + 1):
            dtw_matrix[i, j] = 0
    
    for i in range(1, template_len + 1):
        for j in range(np.max([1, i - w]), np.min([target_len, i + w]) + 1):
            dtw_matrix[i, j] = abs(template[i - 1] - target[j - 1]) + np.min([dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]])

    dtw_matrix = np.delete(dtw_matrix, 0, 0)
    dtw_matrix = np.delete(dtw_matrix, 0, 1)

    distance, path = dtw_distance(dtw_matrix)
    return distance, path, dtw_matrix

def dtw_distance(matrix):
    distance = matrix[-1, -1]
    current_point = (len(matrix) - 1, len(matrix[0]) - 1)
    path = np.array([distance])

    while(current_point != (0, 0)):
        position1 = (current_point[0] - 1, current_point[1] - 1)
        position2 = (current_point[0] - 1, current_point[1])
        position3 = (current_point[0], current_point[1] - 1)
        surrounding = {position1 : matrix[position1[0], position1[1]], position2 : matrix[position2[0], 
            position2[1]], position3 : matrix[position3[0], position3[1]]}

        point = min(surrounding.items(), key=lambda x: x[1])
        path = np.append(path, point[1])
        distance += point[1]
        current_point = point[0]
    
    return distance / len(path), path