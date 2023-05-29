import numpy as np
import matplotlib.pyplot as plt
from line import Line


class LineLayer:
    def __init__(self,
        num_line: int,
        lane_width:float,
        length_list: np.ndarray,
        kappa_list: np.ndarray,
        step: float = 0.5
    ) -> None:
    
        assert len(length_list) == num_line
        assert len(kappa_list) == num_line
        
        # frenet_range = (s_min. s_max, l_min, l_max)
        self.frenet_range = (0, max(length_list), 0, lane_width * (num_line - 1))
        self.waypoints_ls = []
        self.s_vec_ls = []
        self.heading_ls = []
        for idx in range(num_line):
            line = Line.arc(
                length=length_list[idx],
                kappa=kappa_list[idx],
                step=step
            )

            handle = np.array([0, lane_width*idx, 0])
            line.transform_to(handle)

            self.waypoints_ls.append(line.waypoints)
            self.s_vec_ls.append(line.s_vec)
            self.heading_ls.append(line.heading_vec)


if __name__ == '__main__':
    num_line = 3
    ll = LineLayer(
        num_line=num_line,
        lane_width=3,
        length_list=100 * np.ones((num_line,)),
        kappa_list=-0.001 * np.ones((num_line,)),
        step=0.5
    )

    print(ll.frenet_range)

    for l in ll.waypoints_ls:
        plt.scatter(l[:,0], l[:,1], c='g')
    plt.show()