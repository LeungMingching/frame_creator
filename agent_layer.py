import numpy as np
from agent import Agent


class AgentLayer:

    def __init__(self) -> None:

        self.reference_line_xy = None
        self.headings_xy = None
        self.s_vec = None
        self.is_base_reference_line_set = False
        
        self.num_agent = None
        self.agent_location_array = None
        self.agent_heading_array = None
        self.agent_velocity_array = None
        self.agent_acceleration_array = None

    def set_base_reference_line(self,
        reference_line_xy: np.ndarray,
        headings_xy: np.ndarray,
        s_vec: np.ndarray
    ) -> None:
        self.reference_line_xy = reference_line_xy
        self.headings_xy = headings_xy
        self.s_vec = s_vec
        self.is_base_reference_line_set = True
    
    def get_frenet_location_from_mask(self,
        distribution_mask: np.ndarray,
        frenet_range: np.ndarray
    ) -> np.ndarray:
        assert len(distribution_mask.shape) == 2
        distribution_mask = np.flip(distribution_mask, axis=0)

        assert len(frenet_range) == 4
        s_min, s_max, l_min, l_max = (*frenet_range,)
        s_num = distribution_mask.shape[1]
        l_num = distribution_mask.shape[0]
        s_grid = np.linspace(s_min, s_max, s_num)
        l_grid = np.linspace(l_min, l_max, l_num)

        # agent
        l_idx_agent, s_idx_agent = np.where(distribution_mask == 1)
        s_agent = s_grid[s_idx_agent]
        l_agent = l_grid[l_idx_agent]
        agent_loc_sl = np.column_stack((s_agent,l_agent))

        # ego
        l_idx_ego, s_idx_ego = np.where(distribution_mask > 1)
        s_ego = s_grid[s_idx_ego]
        l_ego = l_grid[l_idx_ego]
        ego_loc_sl = np.column_stack((s_ego, l_ego))

        all_agent_loc_sl = np.concatenate((ego_loc_sl, agent_loc_sl), axis=0)
        return all_agent_loc_sl
    
    def create_agents(self,
        location_array, # (n, 2)
        heading_array, # (n,)
        velocity_array, # (n,)
        acceleration_array, # (n, 2)
        ego_type=Agent.init_agent,
        other_type=Agent.aligned_uniformly_accelerate_agent
    ):
        self.num_agent = len(location_array)

        assert len(heading_array) == self.num_agent
        assert len(velocity_array) == self.num_agent
        assert len(acceleration_array) == self.num_agent

        self.agent_location_array = []
        self.agent_heading_array = []
        self.agent_velocity_array = []
        self.agent_acceleration_array = []
        for agent_id in range(self.num_agent):
            
            init_pose = np.append(location_array[agent_id], heading_array[agent_id])
            agent_type = ego_type if agent_id == 0 else other_type
            
            agent = agent_type(
                init_pose,
                velocity_array[agent_id],
                acceleration_array[agent_id]
            )
            # FIXME: pass in the cloest reference line
            if self.is_base_reference_line_set:
                agent.update_cartesian(
                    self.reference_line_xy, self.headings_xy, self.s_vec
                )
            else:
                print('Please set reference line before accessing cartesian info.')
                return
            
            self.agent_location_array.append(agent.path[0])
            self.agent_heading_array.append(agent.heading[0])
            self.agent_velocity_array.append(agent.velocity[0])
            self.agent_acceleration_array.append(agent.acceleration[0])

        self.agent_location_array = np.array(self.agent_location_array)
        self.agent_heading_array = np.array(self.agent_heading_array)
        self.agent_velocity_array = np.array(self.agent_velocity_array)
        self.agent_acceleration_array = np.array(self.agent_acceleration_array)

        

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    distribution_mask = np.array([
        [0, 0, 0, 0, 0,],
        [0, 0, 1, 0, 0,],
        [2, 0, 0, 0, 0,],
        [0, 1, 0, 0, 0,],
        [0, 0, 0, 0, 0 ]
    ])

    ref_line = np.zeros((101, 2))
    ref_line[:,0] = np.linspace(0, 100, 101)
    ref_line[:,1] = np.linspace(0, 100, 101)
    # ref_line[:,1] = np.zeros((101,))
    headings_r = 1*np.pi /4 * np.ones((101,))
    s_vec = np.array([i*np.sqrt(2) for i in range(101)])
    # s_vec = np.array([i for i in range(101)])

    al = AgentLayer()
    al.set_base_reference_line(
        ref_line, headings_r, s_vec
    )

    agents_loc_sl = al.get_frenet_location_from_mask(distribution_mask, (0, 100, 0, 12))
    heading_array = np.ones((len(agents_loc_sl))) * np.pi * 1/6
    velocity_array = np.ones((len(agents_loc_sl))) * 3.0
    acceleration_array = np.ones_like(agents_loc_sl)

    al.create_agents(
        agents_loc_sl, heading_array, velocity_array, acceleration_array,
        ego_type=Agent.init_agent, other_type=Agent.aligned_uniformly_accelerate_agent
    )

    plt.plot(ref_line[:,0], ref_line[:,1])
    for idx in range(al.num_agent):
        marker_color = 'r' if idx == 0 else 'g'
        plt.quiver(*al.agent_location_array[idx], *al.agent_velocity_array[idx], color=marker_color)
    plt.show()