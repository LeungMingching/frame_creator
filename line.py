import numpy as np
import matplotlib.pyplot as plt

from utils import *


class Line:

    def __init__(self,
        waypoints: np.ndarray,
        handle: np.ndarray
    ) -> None:
        self.waypoints = waypoints
        self.handle = handle

        self.s_vec = self.calculate_s_vec()
        self.heading_vec = self.calculate_heading()
    
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

        # to origin (translate -> rotate)
        self.waypoints = translate_2d(self.waypoints, -self.handle[0:2])
        self.waypoints = rotate_2d(self.waypoints, -self.handle[2])

        # to target (rotate -> translate)
        self.waypoints = rotate_2d(self.waypoints, target_handle[2])
        self.waypoints = translate_2d(self.waypoints, target_handle[0:2])

        self.handle = target_handle

        # update heading
        self.heading_vec = self.calculate_heading()

    def calculate_s_vec(self):
        s_vec = [0]
        for idx in range(1, len(self.waypoints)):
            dis = np.linalg.norm(self.waypoints[idx] - self.waypoints[idx-1])
            s_vec.append(s_vec[-1] + dis)
        return np.array(s_vec)
    
    def calculate_heading(self):
        grad = np.gradient(self.waypoints, axis=0)
        dx = grad[:, 0]
        dy = grad[:, 1]
        heading = np.arctan2(dy, dx)
        return heading
    
if __name__ == '__main__':

    l = Line.arc(length=100,
        kappa=-0.01,
        step=0.5
    )

    handle = l.handle
    points = l.waypoints
    print(f'handle: {handle}')
    plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    plt.scatter(points[:,0], points[:,1], c='g')

    new_handle = np.array([18,30,2*np.pi/6])
    l.transform_to(new_handle)
    handle = l.handle
    points = l.waypoints
    print(f'handle: {handle}')
    print(f'points: {points}')
    plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    plt.scatter(points[:,0], points[:,1], c='r')
    plt.show()