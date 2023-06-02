import os
import json
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

        ax = plt.subplot(1, 1, 1)
        # reference lines
        for waypts in waypoints_array:
            x = waypts[:,0]
            y = waypts[:,1]
            ax.plot(x, y, color='g')
        
        # agents
        for idx in range(len(agent_location_array)):
            marker_color = 'r' if idx == 0 else 'b'
            ax.quiver(
                *agent_location_array[idx],
                *agent_velocity_array[idx],
                color=marker_color
            )
        
        ax.set_aspect('equal', 'datalim')
        plt.show()

    def export_to_json(self, save_dir):
        timestamp = np.random.rand() * 1e3

        navi = {
            'command': self.navi_layer.navi_command,
            'execute_distance': self.navi_layer.execute_distance
        }

        ego_status = {
            'position': self.agent_layer.agent_location_array[0].tolist(),
            'heading': self.agent_layer.agent_heading_array[0],
            'velocity': self.agent_layer.agent_velocity_array[0].tolist(),
            'acceleration': self.agent_layer.agent_acceleration_array[0].tolist()
        }

        agent_ls = []
        for i in range(1, self.agent_layer.num_agent):
            agent = {
                'pose': {
                    'BEV': {
                        'position': None,
                        'heading': None,
                        'velocity': None,
                        'acceleration': None
                    },
                    'UTM': {
                        'position': self.agent_layer.agent_location_array[i].tolist(),
                        'heading': self.agent_layer.agent_heading_array[i],
                        'velocity': self.agent_layer.agent_velocity_array[i].tolist(),
                        'acceleration': self.agent_layer.agent_acceleration_array[i].tolist()
                    }
                },
                'is_movable': True,
                'dimension': {
                    "width": 1.81,
                    "height": 1.35,
                    "length": 4.39
                },
                'future_poses': []
            }
            agent_ls.append(agent)

        ref_line_ls = []
        for i in range(0, self.reference_line_layer.num_line):
            ref_line = {
                'lane_attribute': 1111, 
                'passable_type': 1111,
                'waypoint': {
                    'UTM': self.reference_line_layer.waypoints_array[i].tolist(),
                    'BEV': None
                }
            }
            ref_line_ls.append(ref_line)

        frame = {
            'timestamp': timestamp,
            'navi': navi,
            'ego_status': ego_status,
            'agents': agent_ls,
            'reference_lines': ref_line_ls,
            'lane_lines': []
        }

        # save
        file_name = os.path.join(save_dir, f'{int(timestamp)}_frame.json')
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(frame, f, ensure_ascii=False, indent=4)
        print(f'json file saved at {file_name}')


if __name__ == '__main__':

    config = {
        'line_layer': {
            'num_line': 5,
            'lane_width': 10,
            'length_list': np.ones((5,)) * 200,
            'kappa_list': np.ones((5,)) * 0.002,
            'step': 0.5
        },
        'agent_layer': {
            'distribution_mask': np.array([
                [0, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0],
                [1, 2, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0]
            ]),
            'heading_array': np.ones((6,)) * np.pi * 0, # (n,)
            'velocity_array': np.ones((6,)) * 6.0, # (n,)
            'acceleration_array': np.ones((6, 2)) * 0.0 # (n, 2)
        }
    }

    frame = Frame()
    frame.create_frame(config)
    frame.draw()
    frame.export_to_json('./data')
