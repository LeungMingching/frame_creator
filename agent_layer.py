import numpy as np
from agent import Agent


class AgentLayer:

    def __init__(self,
        distribution_mask: np.ndarray
    ) -> None:
        assert len(distribution_mask.shape) == 2

        self.distribution_mask = distribution_mask

    def update_frenet_grid_range(self,
        s_min: float,
        s_max: float,
        l_min: float,
        l_max: float
    ) -> None:
        s_num = self.distribution_mask.shape[1]
        l_num = self.distribution_mask.shape[0]

        s_grid = np.linspace(s_min, s_max, s_num)
        l_grid = np.linspace(l_min, l_max, l_num)

        print(s_grid)
        print(l_grid)


if __name__ == '__main__':
    distribution_mask = np.zeros((5, 5))
    distribution_mask[3][0] = 2 # ego car
    distribution_mask[4][1] = 1 # left ahead agent
    distribution_mask[2][2] = 1 # right ahead agent
    al = AgentLayer(
        distribution_mask=distribution_mask
    )
    al.update_frenet_grid_range(0, 100, 0, 12)