import numpy as np
from utils import *


class Agent:

    def __init__(self,
        path_sl: np.ndarray, # (n, 2)
        heading_sl: np.ndarray, # (n,)
        velocity_sl: np.ndarray,  # (n, 2)
        acceleration_sl: np.ndarray,  # (n, 2)
        t_step: float,
        t_horizon: float,
        dimention: tuple
    ) -> None:
        self.t_step = t_step
        self.t_horizon = t_horizon
        self.dimention = dimention

        # frenet 
        self.path_sl = path_sl
        self.heading_sl = heading_sl
        self.velocity_sl = velocity_sl
        self.acceleration_sl = acceleration_sl

        # cartesian
        self.path = None
        self.heading = None
        self.velocity = None
        self.acceleration = None

    @classmethod
    def init_agent(cls,
        init_pose_sl: np.ndarray,
        init_velocity_sl: float,
        init_acceleration_sl: np.ndarray,
        t_step: float = 0.1,
        t_horizon: float = 3,
        dimention: tuple = (1.81, 1.35, 4.39)
    ): # no trajectory supported
        path = np.array([init_pose_sl[0:2]])
        heading = np.array([init_pose_sl[2]])
        velocity = np.array([init_velocity_sl * np.cos(heading), init_velocity_sl * np.sin(heading)]).reshape((1,2))
        acceleration = np.array([init_acceleration_sl])

        return cls(path, heading, velocity, acceleration, t_step, t_horizon, dimention)
        
    
    @classmethod
    def aligned_uniformly_accelerate_agent(cls,
        init_pose_sl: np.ndarray,
        init_velocity_sl: float,
        init_acceleration_sl: np.ndarray,
        t_step: float = 0.1,
        t_horizon: float = 3,
        dimention: tuple = (3, 2, 2)
    ):
        num_frame = int(t_horizon / t_step + 1)
        s0, l0 = init_pose_sl[0], init_pose_sl[1]
        v0 = init_velocity_sl
        a0 = np.linalg.norm(init_acceleration_sl)

        t_array = np.linspace(0, t_horizon, num_frame)
        a_array = [a0]
        v_array = [v0]
        s_array = [s0]
        for t in t_array[1:]:
            a = a0
            v = v0 + a * t
            s = s0 + v0 * t + 0.5 * a * t**2

            a_array.append(a)
            v_array.append(v)
            s_array.append(s)
        a_array = np.array(a_array)
        v_array = np.array(v_array)
        s_array = np.array(s_array)

        path = l0 * np.ones((num_frame, 2))
        path[:,0] = s_array
        heading = np.zeros((num_frame,)) # heading is aligned to ref_heading
        velocity = np.zeros((num_frame, 2))
        velocity[:,0] = v_array
        acceleration = np.zeros((num_frame, 2))
        acceleration[:,0] = a_array

        return cls(path, heading, velocity, acceleration, t_step, t_horizon, dimention)
    
    def update_cartesian(self,
        reference_line_xy: np.ndarray,
        headings_xy: np.ndarray,
        s_vec: np.ndarray
    ) -> None:
        num_pose = len(self.path_sl)
        
        reference_line_sl = np.zeros_like(reference_line_xy)
        reference_line_sl[:,0] = s_vec
        
        self.path = []
        self.heading = []
        self.velocity = []
        self.acceleration = []
        for i_pose in range(num_pose):
            s = self.path_sl[i_pose][0]
            l = self.path_sl[i_pose][1]
            heading_sl = self.heading_sl[i_pose]

            idx, _ = find_nearest_2d(reference_line_sl, np.array([s, l]))
            x_r = reference_line_xy[idx][0]
            y_r = reference_line_xy[idx][1]
            heading_r = headings_xy[idx]

            x = x_r - l * np.sin(heading_r)
            y = y_r + l * np.cos(heading_r)
            heading = heading_r + heading_sl

            v = rotate_2d(self.velocity_sl[i_pose], heading_r)
            a = rotate_2d(self.acceleration_sl[i_pose], heading_r)

            self.path.append([x, y])
            self.heading.append(heading)
            self.velocity.append(v)
            self.acceleration.append(a)
        
        self.path = np.array(self.path)
        self.heading = np.array(self.heading)
        self.velocity = np.array(self.velocity)
        self.acceleration = np.array(self.acceleration)


if __name__ == '__main__':
    a = Agent.aligned_uniformly_accelerate_agent(
        init_pose_sl=[3, -10, 5*np.pi /6],
        init_velocity_sl=10,
        init_acceleration_sl=[200.0, 0.0]
    )

    ref_line = np.zeros((101, 2))
    ref_line[:,0] = np.linspace(0, 100, 101)
    ref_line[:,1] = np.linspace(0, 100, 101)
    # ref_line[:,1] = np.zeros((101,))
    headings_r = 1*np.pi /4 * np.ones((101,))
    s_vec = np.array([i*np.sqrt(2) for i in range(101)])
    # s_vec = np.array([i for i in range(101)])

    print(f'position_sl: {a.path_sl[0]}')
    print(f'heading_sl: {a.heading_sl[0]}')
    print(f'velocity_sl: {a.velocity_sl[0]}')
    print(f'acceleration_sl: {a.acceleration_sl[0]}')


    a.update_cartesian(
        ref_line, headings_r, s_vec
    )
    print(f'position: {a.path[0]}')
    print(f'path: {a.path}')
    print(f'heading: {a.heading[0]}')
    print(f'velocity: {a.velocity[0]}')
    print(f'acceleration: {a.acceleration[0]}')

    import matplotlib.pyplot as plt

    plt.plot(ref_line[:,0], ref_line[:,1])
    plt.quiver(*a.path[0], *a.velocity[0])
    plt.scatter(a.path[:,0], a.path[:,1], c='g')
    plt.show()