import os
import time
import json
import copy
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
        navi_layer_config = config['navi_layer']

        self.reference_line_layer.create_lines(**line_layer_config)

        self.agent_layer.set_base_reference_line(
            self.reference_line_layer.waypoints_array[0],
            self.reference_line_layer.heading_array[0],
            self.reference_line_layer.s_vec_array[0]
        )

        agent_frenet_range = self._clip_frenet_range(
            config['agent_layer']['agent_frenet_range'],
            self.reference_line_layer.frenet_range)
        agents_loc_sl = self.agent_layer.get_frenet_location_from_mask(
            agent_layer_config['distribution_mask'],
            agent_frenet_range
        )
        self.agent_layer.create_agents(
            location_array = agents_loc_sl,
            heading_array = agent_layer_config['heading_array'],
            velocity_array = agent_layer_config['velocity_array'],
            acceleration_array = agent_layer_config['acceleration_array'],
            ego_type=Agent.init_agent,
            other_type=Agent.aligned_uniformly_accelerate_agent
        )

        self.navi_layer.create_navi(
            navi_command=navi_layer_config['command'],
            execute_distance=navi_layer_config['execute_distance'],
        )

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

    def get_frame_dict(self):
        timestamp = np.mod(np.round(time.time()*1e3), 1e9) *1e3

        navi = {
            'command': self.navi_layer.navi_command,
            'execute_distance': np.round(self.navi_layer.execute_distance, decimals=4).item()
        }

        try:
            velocity = np.round(np.linalg.norm(self.agent_layer.agent_velocity_array[0]), decimals=4)
            ego_status = {
                'position': np.round(self.agent_layer.agent_location_array[0], decimals=4).tolist(),
                'heading': np.round(self.agent_layer.agent_heading_array[0], decimals=4).item(),
                'velocity': [velocity.item(), 0],
                'acceleration': np.round(self.agent_layer.agent_acceleration_array[0], decimals=4).tolist()
            }
        except:
            print('No ego status found!')
            ego_status = None

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
                        'position': np.round(self.agent_layer.agent_location_array[i], decimals=4).tolist(),
                        'heading': np.round(self.agent_layer.agent_heading_array[i], decimals=4).item(),
                        'velocity': np.round(self.agent_layer.agent_velocity_array[i], decimals=4).tolist(),
                        'acceleration': np.round(self.agent_layer.agent_acceleration_array[i], decimals=4).tolist()
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
                    'UTM': np.round(np.array(self.reference_line_layer.waypoints_array[i].tolist()), decimals=4).tolist(),
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

        return copy.deepcopy(frame)
    
    def  _clip_frenet_range(self, agent_range, reference_line_range):
        s_agent_range = agent_range[0:2]
        l_agent_range = agent_range[2:4]
        cliped_s_range = np.clip(s_agent_range, *reference_line_range[0:2])
        cliped_l_range = np.clip(l_agent_range, *reference_line_range[2:4])
        cliped_range = np.concatenate((cliped_s_range, cliped_l_range))
        return cliped_range


if __name__ == '__main__':

    config = {
        'navi_layer': {
            'command': np.random.randint(0, 4),
            'execute_distance': np.random.rand() * 300
        },
        'line_layer': {
            'num_line': 5,
            'lane_width': 5,
            'outer_length': 200,
            'outer_radius': -200,
            'step': 0.5
        },
        'agent_layer': {
            'agent_frenet_range': np.array([50, 100, 2, 4]),
            'distribution_mask': np.array([
                [0, 0, 0, 0, 0, 1],
                [0, 0, 1, 0, 0, 0],
                [1, 2, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ]),
            'heading_array': np.ones((6,)) * np.pi * 0, # (n,)
            'velocity_array': np.ones((6,)) * 6.0, # (n,)
            'acceleration_array': np.ones((6, 2)) * 0.0 # (n, 2)
        }
    }

    frame = Frame()
    frame.create_frame(config)
    frame.draw()
    frame_dict = frame.get_frame_dict()

    # save
    timestamp = frame_dict['timestamp']
    file_name = os.path.join('./data', f'{timestamp}_frame.json')
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(frame_dict, f, ensure_ascii=False, indent=4)
    print(f'json file saved at {file_name}')
