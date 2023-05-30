import numpy as np


def homogenous_transform(
    query_array: np.ndarray,
    translation_vec: np.ndarray,
    rotational_theta: float,
) -> np.ndarray:
    
    is_single_pt = (len(query_array.shape) == 1)
    if is_single_pt:
        query_array = np.expand_dims(query_array, axis=0)

    T_mat = np.array([
        [np.cos(rotational_theta), -np.sin(rotational_theta), translation_vec[0]],
        [np.sin(rotational_theta), np.cos(rotational_theta), translation_vec[1]],
        [0, 0, 1],
    ])

    result_array = []
    for pt in query_array:
        query_pt = np.append(pt, 1).reshape((3, 1))
        transfered_pt = np.matmul(T_mat, query_pt)
        transfered_pt = transfered_pt.reshape((3,))
        result_array.append(transfered_pt[0:2])
    result_array = np.array(result_array)

    if is_single_pt:
        return result_array[0]
    else:
        return result_array

def rotate_2d(
    query_array: np.ndarray,
    theta: float,
) -> np.ndarray:

    is_single_pt = (len(query_array.shape) == 1)
    if is_single_pt:
        query_array = np.expand_dims(query_array, axis=0)
    
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    result_array = []
    for pt in query_array:
        transfered_pt = np.matmul(rotation_matrix, pt)
        result_array.append(transfered_pt)
    result_array = np.array(result_array)
    
    if is_single_pt:
        return result_array[0]
    else:
        return result_array

def translate_2d(
    query_array: np.ndarray,
    vector: float,
) -> np.ndarray:

    is_single_pt = (len(query_array.shape) == 1)
    if is_single_pt:
        query_array = np.expand_dims(query_array, axis=0)

    result_array = []
    for pt in query_array:
        transfered_pt = vector + pt
        result_array.append(transfered_pt)
    result_array = np.array(result_array)
    
    if is_single_pt:
        return result_array[0]
    else:
        return result_array
    
def get_sub_idxs(
    capacity: list,
    idx: int
) -> list:
    idx_ls = []
    for cap in reversed(capacity):
        result = np.divmod(idx, cap)
        idx_ls.append(result[1])
        idx = result[0]
    idx_ls.reverse()
    return idx_ls

def find_nearest_2d(
    point_list: np.ndarray,
    query_point: np.ndarray
) -> tuple:
    distance = [np.linalg.norm(point - query_point) for point in point_list]
    distance = np.array(distance)
    idx = distance.argmin()
    return idx, distance[idx]

def frenet_to_cartesian(
    reference_line: np.ndarray, # in cartesian
    headings_vec: np.ndarray, # in cartesian
    s_vec: np.ndarray, # in frenet
    query_pose_array: np.ndarray, # in frenet
) -> np.ndarray:
    
    reference_line_sl = np.zeros_like(reference_line)
    reference_line_sl[:,0] = s_vec
    
    converted_pose_array = []
    for query_pose in query_pose_array:
        s = query_pose[0]
        l = query_pose[1]
        heading_sl = query_pose[2]

        idx, _ = find_nearest_2d(reference_line_sl, np.array([s, l]))
        x_r = reference_line[idx][0]
        y_r = reference_line[idx][1]
        heading_r = headings_vec[idx]

        x = x_r - l * np.sin(heading_r)
        y = y_r + l * np.cos(heading_r)
        heading = heading_r + heading_sl

        converted_pose_array.append([x, y, heading])
    converted_pose_array = np.array(converted_pose_array)
    return converted_pose_array
