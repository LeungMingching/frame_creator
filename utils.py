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