import numpy as np
import matplotlib.pyplot as plt

from line_layer import LineLayer
from agent_layer import AgentLayer
from agent import Agent
from navi_layer import NaviLayer


class Frame:

    def __init__(self) -> None:

        self.reference_line_layer = LineLayer()
        self.agent_layer = AgentLayer()
        self.navi_layer = NaviLayer()

    def create_frame(self, 
        config: dict
    ):
        line_layer_config = config['line_layer']
        agent_layer_config = config['agent_layer']

        self.reference_line_layer.create_lines(**line_layer_config)

        self.agent_layer.set_base_reference_line(
            self.reference_line_layer.waypoints_array[0],
            self.reference_line_layer.heading_array[0],
            self.reference_line_layer.s_vec_array[0]
        )

        agents_loc_sl = self.agent_layer.get_frenet_location_from_mask(
            agent_layer_config['distribution_mask'],
            self.reference_line_layer.frenet_range
        )
        self.agent_layer.create_agents(
            location_array = agents_loc_sl,
            heading_array = agent_layer_config['heading_array'],
            velocity_array = agent_layer_config['velocity_array'],
            acceleration_array = agent_layer_config['acceleration_array'],
            ego_type=Agent.init_agent,
            other_type=Agent.aligned_uniformly_accelerate_agent
        )

        self.navi_layer.create_navi()

    def draw(self) -> None:
        waypoints_array = self.reference_line_layer.waypoints_array
        agent_location_array = self.agent_layer.agent_location_array
        agent_velocity_array = self.agent_layer.agent_velocity_array

        # reference lines
        for waypts in waypoints_array:
            x = waypts[:,0]
            y = waypts[:,1]
            plt.plot(x, y, color='g')
        
        # agents
        for idx in range(len(agent_location_array)):
            marker_color = 'r' if idx == 0 else 'b'
            plt.quiver(
                *agent_location_array[idx],
                *agent_velocity_array[idx],
                color=marker_color
            )
        
        max_limit = max(waypoints_array.flatten()) + 30
        min_limit = min(waypoints_array.flatten()) - 30
        plt.xlim(min_limit, max_limit)
        plt.ylim(min_limit, max_limit)
        plt.show()


if __name__ == '__main__':

    config = {
        'line_layer': {
            'num_line': 5,
            'lane_width': 10,
            'length_list': np.ones((5,)) * 200,
            'kappa_list': np.ones((5,)) * 0.008,
            'step': 0.5
        },
        'agent_layer': {
            'distribution_mask': np.array([
                [0, 1, 0, 0, 1,],
                [0, 0, 0, 0, 0,],
                [1, 0, 2, 0, 0,],
                [0, 1, 0, 0, 1,],
                [0, 0, 0, 0, 0 ]
            ]),
            'heading_array': np.ones((6,)) * np.pi * 1/6, # (n,)
            'velocity_array': np.ones((6,)) * 6.0, # (n,)
            'acceleration_array': np.ones((6, 2)) * 0.0 # (n, 2)
        }
    }

    frame = Frame()
    frame.create_frame(config)
    frame.draw()
