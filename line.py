import numpy as np
import matplotlib.pyplot as plt

from utils import homogenous_transform


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
    
    def transform_to(self,
        target_handle: np.ndarray
    ) -> None:

        

        pass
    
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