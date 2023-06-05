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


frame_collection = []
frame = Frame()

# for navi_cmd in tqdm(navi_cmd_grid, desc='Navi Layer: '):
navi_cmd = np.random.randint(0, 4)

for num_line in tqdm(num_ref_line, desc='Reference Line Layer: '):
    # for outer_l in outer_length:
    outer_l = 150 + 100*np.random.rand()
    for outer_r in outer_radius:

        l_dim = int(2*num_line + 1)
        s_dim = int(np.floor(outer_l/4.0) + 1)
        
        for ego_l_idx in range(l_dim):
            distribution_mask = np.zeros((l_dim, s_dim))
            distribution_mask[ego_l_idx][0] = 2

            # for ego_head in ego_heading:
            sigma = np.pi * 80 / 180 / 3
            mu = 0.0
            ego_head = sigma * np.random.randn() + mu
            ego_head_array = np.array([ego_head])

            for ego_v in ego_velocity:
                ego_v_array = np.array([ego_v])

                config = {
                    'navi_layer': {
                        'command': navi_cmd,
                        'execute_distance': np.random.rand() * 300
                    },
                    'line_layer': {
                        'num_line': num_line,
                        'lane_width': 3.5,
                        'outer_length': outer_l,
                        'outer_radius': outer_r,
                        'step': 0.5
                    },
                    'agent_layer': {
                        'distribution_mask': distribution_mask,
                        'heading_array': ego_head_array, # (n,)
                        'velocity_array': ego_v_array, # (n,)
                        'acceleration_array': ego_acceleration # (n, 2)
                    }
                }

                frame.create_frame(config)
                # frame.draw()
                frame_dict = frame.get_frame_dict()
                frame_collection.append(frame_dict)


# save
save_dir = './data'
file_name = os.path.join('./data', 'rand.json')
with open(file_name, 'w', encoding='utf-8') as f:
    json.dump(frame_collection, f, ensure_ascii=False, indent=4)
print(f'json file saved at {file_name}')