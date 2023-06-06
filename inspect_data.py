import json
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


with open('data/1686032134882000_frames.json') as f:
    frame_collection = json.load(f)

plt.ion()
fig, ax = plt.subplots(1, 1)
ax.set_aspect('equal', 'datalim')

# for batch_idx, data in enumerate(dataloader):
for frame in tqdm(frame_collection):
    ax.cla()
    # navi
    navi_cmd = frame['navi']['command']
    execute_distance = frame['navi']['execute_distance']
    ax.text(.01, .99,
            f'navi_cmd: {navi_cmd} \nexecute_distance: {execute_distance:.2f}',
            ha='left', va='top', transform = ax.transAxes)

    # reference lines
    for ref_line in frame['reference_lines']:
        waypts = ref_line['waypoint']['UTM']
        waypts = np.array(waypts)
        x = waypts[:,0]
        y = waypts[:,1]
        ax.plot(x, y, color='g')
    
    # ego
    ego_position = np.array(frame['ego_status']['position'])
    ego_heading = np.array(frame['ego_status']['heading'])
    ego_velocity = np.array(frame['ego_status']['velocity'])

    ax.text(*ego_position, f'({ego_velocity[0].item():.2f}, {ego_velocity[1].item():.2f})')
    ax.quiver(*ego_position, np.cos(ego_heading), np.sin(ego_heading), color='r')

    # agents
    for ag in frame['agents']:
        position = np.array(ag['pose']['UTM']['position'])
        heading = np.array(ag['pose']['UTM']['heading'])
        velocity = np.array(ag['pose']['UTM']['velocity'])

        ax.text(*position, f'({velocity[0].item():.2f}, {velocity[1].item():.2f})')
        ax.quiver(*position, *velocity, color='b')
    
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.8)
plt.ioff()