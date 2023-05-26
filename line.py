import numpy as np
import matplotlib.pyplot as plt


class Line:

    def __init__(self,
        waypoints: np.ndarray,
        handle: np.ndarray
    ) -> None:
        self.waypoints = waypoints
        self.handle = handle
    
    @classmethod
    def arc(cls,
        length: float,
        kappa: float,
        step: float
    ):
        kappa = 1e-12 if kappa == 0.0 else kappa
        sign = np.sign(kappa)

        radius = np.abs(1 / kappa)
        num_pt = int(np.ceil(length / step))
        angular_range = kappa * length
        idxs = np.linspace(0, angular_range, num_pt)
        points = []
        for i in idxs:
            points.append([radius * np.cos(i), radius * np.sin(i)])
        points = np.array(points)

        handle = np.array([radius, 0.0, sign*np.pi/2])

        return cls(points, handle)
    
    def transform(self,
        translation_vec: np.ndarray,
        rotate_theta: float
    ) -> None:

        T_mat = np.array([
            [np.cos(rotate_theta), -np.sin(rotate_theta), translation_vec[0]],
            [np.sin(rotate_theta), np.cos(rotate_theta), translation_vec[1]],
            [0, 0, 1],
        ])

        result_array = []
        for pt in self.waypoints:
            query_pt = np.append(pt, 1).reshape((3, 1))
            transfered_pt = np.matmul(T_mat, query_pt)
            transfered_pt = transfered_pt.reshape((3,))
            result_array.append(transfered_pt[0:2])
        result_array = np.array(result_array)

        self.waypoints = result_array
    
if __name__ == '__main__':

    l = Line.arc(length=100,
        kappa=-0.001,
        step=0.5
    )

    handle = l.handle
    points = l.waypoints
    print(f'handle: {handle}')
    # plt.xlim(-100, 100)
    # plt.ylim(-100, 100)
    plt.scatter(points[:,0], points[:,1])
    plt.show()