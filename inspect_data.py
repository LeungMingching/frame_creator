import json
import time
import numpy as np
import matplotlib.pyplot as plt


with open('data/rand.json') as f:
    frame_collection = json.load(f)

plt.ion()
fig, ax = plt.subplots(1, 1)

# for batch_idx, data in enumerate(dataloader):
for frame in frame_collection:
    ax.cla()
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

    ax.quiver(*ego_position, *ego_velocity, color='r')

    # agents
    for ag in frame['agents']:
        position = np.array(ag['pose']['UTM']['position'])
        heading = np.array(ag['pose']['UTM']['heading'])
        velocity = np.array(ag['pose']['UTM']['velocity'])

        ax.quiver(*position, *velocity, color='b')
    
    ax.set_aspect('equal', 'datalim')
    fig.canvas.draw()
    fig.canvas.flush_events()
    # time.sleep(0.01)
plt.ioff()