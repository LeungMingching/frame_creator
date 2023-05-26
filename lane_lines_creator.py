import numpy as np
import matplotlib.pyplot as plt

from utils import *


class LaneLinesCreator:

    def __init__(self,
        num_range: tuple, # (start, end, step)
        length_range: tuple,
        kappa_range: tuple
    ) -> None:

        self.num_range = num_range
        self.length_range = length_range
        self.kappa_range = kappa_range

    def create_arc(self,
        start_point: list,
        start_heading: float,
        length: float,
        kappa: float,
        step: float
    ) -> list[list]:
        
        kappa = 1e-12 if kappa == 0.0 else kappa
        
        # get the circle at (0, 0)
        radius = 1 / kappa
        angular_range = kappa * length
        num_pt = int(np.ceil(length / step))
        idxs = np.linspace(0, angular_range, num_pt)
        origin_pt = []
        for i in idxs:
            origin_pt.append([radius * np.cos(i), radius * np.sin(i)])
        origin_pt = np.array(origin_pt)

        # homogenous transform
        # TODO: check rotation & translation
        # transfered_array = origin_pt
        # transfered_array = homogenous_transform(origin_pt, np.array([0, radius]), -np.pi/2)
        # # transfered_array[:,1] = -transfered_array[:,1]
        # transfered_array = homogenous_transform(transfered_array, np.array(start_point), start_heading)

        translation_vec = np.array(start_point) - np.array([radius, 0])
        print(f'translation_vec: {translation_vec}')
        theta = np.pi/2 + start_heading
        transfered_array = homogenous_transform(origin_pt, translation_vec, theta)

        # plt.xlim(-100, 100)
        # plt.ylim(-100, 100)
        plt.scatter(transfered_array[:,0], transfered_array[:,1])
        plt.show()

        return transfered_array

if __name__ == '__main__':

    lane_line_creator = LaneLinesCreator(
        num_range = (0, 6, 1),
        length_range = (0, 100, 0.5),
        kappa_range = (0, 2, 0.1)
    )
    lane_line_creator.create_arc([20, 30], np.pi*1/6, 100, 0.01, 0.5)
