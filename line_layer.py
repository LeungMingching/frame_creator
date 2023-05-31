import numpy as np
from line import Line


class LineLayer:
    
    def __init__(self) -> None:

        self.frenet_range = None
        self.waypoints_ls = None
        self.s_vec_ls = None
        self.heading_ls = None
    
    def create_lines(self,
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
    import matplotlib.pyplot as plt
    
    num_line = 3
    ll = LineLayer()
    ll.create_lines(
        num_line=num_line,
        lane_width=3,
        # length_list=100 * np.ones((num_line,)),
        length_list=[50, 200, 100],
        # kappa_list=-0.001 * np.ones((num_line,)),
        kappa_list=[-0.001, 0.0, 0.005],
        step=0.5
    )

    print(ll.frenet_range)

    for l in ll.waypoints_ls:
        plt.scatter(l[:,0], l[:,1], c='g')
    plt.show()