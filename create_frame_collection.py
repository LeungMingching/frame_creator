import os
import json
from tqdm import tqdm
import numpy as np
from frame import Frame

# navi
navi_cmd_grid = np.linspace(0, 3, 4)

# reference line
num_ref_line = np.linspace(1, 5, num=5, dtype=int)
outer_length = np.array([200])
outer_radius = np.union1d(
    np.linspace(400, 1000, 6),
    -np.linspace(400, 1000, 6)
)
outer_radius = np.sort(np.append(outer_radius, 1e9))

# ego
ego_heading = np.linspace(-80/180 * np.pi, -80/180 * np.pi, 5)
ego_velocity = np.linspace(10, 20, 5)
ego_acceleration = np.zeros((1,2))

# objects


frame = Frame()

def generate_squential_collection(
    navi_cmd_size: int,
    max_num_ref_line: int,
    outer_length_range: tuple, # (min, max, num)
    outer_kappa_range: tuple, # (min, max, num)
    ego_heading_range: tuple, # (min, max, num)
    ego_velocity_range: tuple, # (min, max, num)
    ego_acceleration_range: tuple, # (min, max, num)
):
    frame_collection = []
    
    navi_cmd_grid = np.linspace(0, navi_cmd_size-1, navi_cmd_size)

    num_ref_line_grid = np.linspace(1, max_num_ref_line, num=max_num_ref_line, dtype=int)
    outer_length_grid = np.linspace(outer_length_range[0], outer_length_range[1], num=outer_length_range[2])
    outer_kappa_grid = np.linspace(outer_kappa_range[0], outer_kappa_range[1], num=outer_kappa_range[2])
    outer_radius_grid = 1.0 / outer_kappa_grid
    ego_heading_grid = np.linspace(ego_heading_range[0], ego_heading_range[1], num=ego_heading_range[2])
    ego_velocity_grid = np.linspace(ego_velocity_range[0], ego_velocity_range[1], num=ego_velocity_range[2])
    ego_acceleration_grid = np.linspace(ego_acceleration_range[0], ego_acceleration_range[1], num=ego_acceleration_range[2])
    
    for navi_cmd in tqdm(navi_cmd_grid, desc='navi_cmd'):

        for num_ref_line in tqdm(num_ref_line_grid, desc='num_ref_line'):
            for outer_length in outer_length_grid:
                for outer_radius in outer_radius_grid:

                    l_dim = int(2*num_ref_line + 1)
                    s_dim = int(np.floor(outer_length/4.0) + 1)

                    for ego_l_idx in range(l_dim):
                        distribution_mask = np.zeros((l_dim, s_dim))
                        distribution_mask[ego_l_idx][0] = 2

                        for ego_heading in ego_heading_grid:
                            ego_heading_array = np.array([ego_heading])

                            for ego_velocity in ego_velocity_grid:
                                ego_velocity_array = np.array([ego_velocity])

                                ego_acceleration_array = np.zeros((1,2))

                                config = {
                                    'navi_layer': {
                                        'command': navi_cmd,
                                        'execute_distance': np.random.rand() * 300
                                    },
                                    'line_layer': {
                                        'num_line': num_ref_line,
                                        'lane_width': 3.5,
                                        'outer_length': outer_length,
                                        'outer_radius': outer_radius,
                                        'step': 0.5
                                    },
                                    'agent_layer': {
                                        'distribution_mask': distribution_mask,
                                        'heading_array': ego_heading_array, # (n,)
                                        'velocity_array': ego_velocity_array, # (n,)
                                        'acceleration_array': ego_acceleration_array # (n, 2)
                                    }
                                }

                                frame.create_frame(config)
                                # frame.draw()
                                frame_dict = frame.get_frame_dict()
                                frame_collection.append(frame_dict)

    return frame_collection


frame_collection = generate_squential_collection(
    navi_cmd_size=4,
    max_num_ref_line=5,
    outer_length_range=(150, 250, 5), # (min, max, num)
    outer_kappa_range=(-0.0025, 0.0025, 8), # (min, max, num)
    ego_heading_range=(-80/180 * np.pi, -80/180 * np.pi, 5), # (min, max, num)
    ego_velocity_range=(0, 20, 5), # (min, max, num)
    ego_acceleration_range=(0, 0, 0), # (min, max, num)
)

# save
save_dir = './data'
file_name = os.path.join('./data', 'rand.json')
with open(file_name, 'w', encoding='utf-8') as f:
    json.dump(frame_collection, f, ensure_ascii=False, indent=4)
print(f'json file saved at {file_name}')