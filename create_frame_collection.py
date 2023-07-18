import os
import json
from tqdm import tqdm
import numpy as np
from frame import Frame


def generate_squential_meta(
    navi_cmd_size: int,
    max_num_ref_line: int,
    outer_length_range: tuple, # (min, max, num)
    outer_kappa_range: tuple, # (min, max, num)
    ego_heading_range: tuple, # (min, max, num)
    ego_velocity_range: tuple, # (min, max, num)
    ego_acceleration_range: tuple, # (min, max, num)
):
    meta_collection = []
    
    navi_cmd_grid = np.linspace(0, navi_cmd_size-1, navi_cmd_size)

    num_ref_line_grid = np.linspace(1, max_num_ref_line, num=max_num_ref_line, dtype=int)
    outer_length_grid = np.linspace(outer_length_range[0], outer_length_range[1], num=outer_length_range[2])
    outer_kappa_grid = np.linspace(outer_kappa_range[0], outer_kappa_range[1], num=outer_kappa_range[2])
    outer_radius_grid = 1.0 / outer_kappa_grid
    ego_heading_grid = np.linspace(ego_heading_range[0], ego_heading_range[1], num=ego_heading_range[2])
    ego_velocity_grid = np.linspace(ego_velocity_range[0], ego_velocity_range[1], num=ego_velocity_range[2])
    ego_acceleration_grid = np.linspace(ego_acceleration_range[0], ego_acceleration_range[1], num=ego_acceleration_range[2])
    
    for navi_cmd in navi_cmd_grid:

        for num_ref_line in num_ref_line_grid:
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
                                meta_collection.append(config)

    return meta_collection

def generate_random_meta(
    num_frame: int,
    navi_cmd_size: int,
    max_num_ref_line: int,
    outer_length_range: tuple, # (min, max)
    outer_kappa_range: tuple, # (min, max)
    agent_frenet_range: tuple, # (s_min, s_max, l_min, l_max)
    distribution_density_range: tuple, # (min, max)
    heading_range: tuple, # (min, max)
    velocity_range: tuple, # (min, max)
    acceleration_range: tuple # (min, max)
):
    meta_collection = []
    
    for _ in range(num_frame):
    
        # navi
        navi_cmd = np.random.choice(np.arange(navi_cmd_size))

        # reference line
        num_ref_line = np.random.choice(np.arange(1, max_num_ref_line + 1))
        outer_length = np.random.rand() * (outer_length_range[1] - outer_length_range[0]) + outer_length_range[0]
        outer_kappa = np.random.rand() * (outer_kappa_range[1] - outer_kappa_range[0]) + outer_kappa_range[0]
        outer_radius = 1 / (outer_kappa + 1e-9)

        # distribution
        l_dim = num_ref_line
        s_length = agent_frenet_range[1] - agent_frenet_range[0]
        s_dim = int(np.floor(s_length/5.0) + 1)
        p_threshold = np.random.rand() * (distribution_density_range[1] - distribution_density_range[0]) + distribution_density_range[0]
        
        distribution_mask = np.random.rand(l_dim, s_dim)
        distribution_mask = (distribution_mask < p_threshold).astype(float)
        ego_l_idx = np.random.choice(np.arange(l_dim))
        distribution_mask[ego_l_idx][0] = 2

        num_all_agent = (distribution_mask > 0).sum()

        # agents
        mu = (heading_range[1] + heading_range[0]) / 2.0
        simga = (heading_range[1] - heading_range[0]) / 6.0
        heading_array = np.random.randn(num_all_agent,) * simga + mu
        velocity_array = np.random.rand(num_all_agent,) * (velocity_range[1] - velocity_range[0]) + velocity_range[0]
        acceleration_array = np.zeros((num_all_agent,2))
        
        config = {
            'navi_layer': {
                'command': int(navi_cmd),
                'execute_distance': float(np.random.rand() * 300)
            },
            'line_layer': {
                'num_line': int(num_ref_line),
                'lane_width': 3.5,
                'outer_length': float(outer_length),
                'outer_radius': float(outer_radius),
                'step': 0.5
            },
            'agent_layer': {
                'agent_frenet_range': agent_frenet_range,
                'distribution_mask': distribution_mask,
                'heading_array': heading_array, # (n,)
                'velocity_array': velocity_array, # (n,)
                'acceleration_array': acceleration_array # (n, 2)
            }
        }
        meta_collection.append(config)

    return meta_collection

def generate_sample_overtake_meta(
    num_frame: int,
    navi_cmd_size: int,
    max_num_ref_line: int,
    outer_length_range: tuple, # (min, max)
    outer_kappa_range: tuple, # (min, max)
    agent_frenet_range: tuple, # (s_min, s_max, l_min, l_max)
    num_agent_range: tuple, # (min, max)
    heading_range: tuple, # (min, max)
    velocity_range: tuple, # (min, max)
    acceleration_range: tuple # (min, max)
):
    meta_collection = []

    for _ in range(num_frame):
        # navi
        navi_cmd = int(0)

        # reference line
        num_ref_line = np.random.choice(np.arange(1, max_num_ref_line + 1))
        outer_length = np.random.rand() * (outer_length_range[1] - outer_length_range[0]) + outer_length_range[0]
        outer_kappa = np.random.rand() * (outer_kappa_range[1] - outer_kappa_range[0]) + outer_kappa_range[0]
        outer_radius = 1 / (outer_kappa + 1e-9)

        # distribution
        l_dim = num_ref_line
        s_length = agent_frenet_range[1] - agent_frenet_range[0]
        s_dim = int(np.floor(s_length/15.0) + 1)

        distribution_mask = np.zeros((l_dim, s_dim))
        l_idx = np.random.choice(np.arange(l_dim))
        distribution_mask[l_idx][0] = 2

        num_agent = np.random.randint(num_agent_range[0], num_agent_range[1])
        ag_s_idxs = np.random.randint(1, s_dim, size=num_agent)
        distribution_mask[l_idx][ag_s_idxs] = 1

        num_all_agent = (distribution_mask > 0).sum()

        mu = (heading_range[1] + heading_range[0]) / 2.0
        simga = (heading_range[1] - heading_range[0]) / 6.0
        heading_array = np.zeros((num_all_agent,))
        heading_array[0] = np.random.randn() * simga + mu
        velocity_array = np.zeros((num_all_agent,))
        velocity_array[0] = np.random.rand() * (velocity_range[1] - velocity_range[0]) + velocity_range[0]
        acceleration_array = np.zeros((num_all_agent,2))

        config = {
            'navi_layer': {
                'command': int(navi_cmd),
                'execute_distance': float(np.random.rand() * 300)
            },
            'line_layer': {
                'num_line': int(num_ref_line),
                'lane_width': 3.5,
                'outer_length': float(outer_length),
                'outer_radius': float(outer_radius),
                'step': 0.5
            },
            'agent_layer': {
                'agent_frenet_range': agent_frenet_range,
                'distribution_mask': distribution_mask,
                'heading_array': heading_array, # (n,)
                'velocity_array': velocity_array, # (n,)
                'acceleration_array': acceleration_array # (n, 2)
            }
        }
        meta_collection.append(config)

    return meta_collection


def export_frame_to_json(save_dir, frame_per_file, meta_collection):
    assert len(meta_collection) != 0

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    frame = Frame()
    frame_collection = []
    print(f'Total number of frame: {len(meta_collection)}')
    for idx, config in enumerate(tqdm(meta_collection, desc='Frames: ')):
        frame.create_frame(config)
        # frame.draw()
        frame_dict = frame.get_frame_dict()
        frame_collection.append(frame_dict)

        if (idx+1) % frame_per_file == 0:
            start_timestamp = frame_collection[0]['timestamp']
            file_name = f'{int(start_timestamp)}_frames.json'
            with open(os.path.join(save_dir, file_name), 'w', encoding='utf-8') as f:
                json.dump(frame_collection, f, ensure_ascii=False, indent=4)
            print(f'json file saved at {file_name}')

            frame_collection = []


# meta_collection = generate_squential_meta(
#     navi_cmd_size=4,
#     max_num_ref_line=5,
#     outer_length_range=(150, 250, 5), # (min, max, num)
#     outer_kappa_range=(-0.0025, 0.0025, 8), # (min, max, num)
#     ego_heading_range=(-80/180 * np.pi, 80/180 * np.pi, 5), # (min, max, num)
#     ego_velocity_range=(0, 20, 5), # (min, max, num)
#     ego_acceleration_range=(0, 0, 0), # (min, max, num)
# )

# meta_collection = generate_random_meta(
#     num_frame=5001,
#     navi_cmd_size=3,
#     max_num_ref_line=5,
#     outer_length_range=(600, 600), # (min, max)
#     outer_kappa_range=(-0.0025, 0.0025), # (min, max)
#     agent_frenet_range=(30, 300, 0, 50), # (s_min, s_max, l_min, l_max)
#     distribution_density_range=(0.0, 0.0), # (min, max)
#     heading_range=(-60/180 * np.pi, 60/180 * np.pi), # (min, max)
#     velocity_range=(0, 0), # (min, max)
#     acceleration_range=(0, 0) # (min, max)
# )

meta_collection = generate_sample_overtake_meta(
    num_frame=5001,
    navi_cmd_size=3,
    max_num_ref_line=5,
    outer_length_range=(600, 600), # (min, max)
    outer_kappa_range=(-0.0025, 0.0025), # (min, max)
    agent_frenet_range=(30, 300, 0, 50), # (s_min, s_max, l_min, l_max)
    num_agent_range=(0, 3), # (min, max)
    heading_range=(-60/180 * np.pi, 60/180 * np.pi), # (min, max)
    velocity_range=(0, 20), # (min, max)
    acceleration_range=(0, 0) # (min, max)
)

save_dir = './data/0718/lk_ag'
export_frame_to_json(save_dir, 500, meta_collection)
