# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt
from matplotlib import cm

from read_metadata import read_metadata


# %%
get_ipython().magic('store -r vid_name')
metadata = read_metadata(vid_name)

video_out_file = metadata.folder_output + vid_name + '_tracks.mp4'
os.path.join(metadata.folder_images, '*.jpg')


# %%
#import re
#def atoi(text):
#    return int(text) if text.isdigit() else text

#def natural_keys(text):
#    '''
#    alist.sort(key=natural_keys) sorts in human order
#    http://nedbatchelder.com/blog/200712/human_sorting.html
#    (See Toothy's implementation in the comments)
#    '''
#    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


frame_files = sorted(glob.glob(os.path.join(metadata.folder_images, '*.jpg')))
#frame_files.sort(key=natural_keys)

###print("{} frame files found.".format(len(frame_files)))
tracks_file = metadata.folder_output + 'tracks_' + vid_name + '.npy'
tracks = np.load(tracks_file, allow_pickle=True) 


# %%
im = cv2.imread(frame_files[0])

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(video_out_file, fourcc, 23.976, (im.shape[1], im.shape[0]))


# %%
import pandas as pd

def calc_distance(a, b):
    dx = a[1] - b[1]
    dy = a[1] - b[1]
    return np.sqrt(np.sum(dx**2 + dy**2))

def get_color(f):
    """return 3d hsv color based on f from 0 to 1"""
    c = (np.array(cm.hsv(f)[:3]) * 255)
    return (int(c[0]), int(c[1]), int(c[2]))

frame_ind = 1

steps = 50 # 90
step_inc = 1 # 5
circle_radius = 5 # 20

with_tail = True

colors = np.random.uniform(size=len(tracks))
colors = [get_color(c) for c in colors]

all_tracking = pd.DataFrame({'frame': [], 'id': [], 'x': [], 'y':[],'class':[]})

#all_tracking = []
all_distance = []
#ax = plt.gca()
for frame_ind in range(0, len(frame_files), 1):

    im = cv2.imread(frame_files[frame_ind])
    ###if frame_ind % 250 == 0:
       ## print(frame_ind)


    for track_ind, track in enumerate(tracks):
        rel_frame = frame_ind - track['first_frame']
        if rel_frame >= 0:
            if track['last_frame'] >= frame_ind:
                    center_pos = track['track'][rel_frame]
                    cv2.circle(im, (int(center_pos[0]), int(center_pos[1])), circle_radius, colors                                    [track_ind], 2)
                    cv2.putText(im, str(track_ind), (int(center_pos[0]), int(center_pos[1])), 0, 1, colors[track_ind],2)

                    to_append = [frame_ind,track_ind,int(center_pos[0]), int(center_pos[1]),track['class']]
                    a_series = pd.Series(to_append, index = all_tracking.columns)
                    all_tracking = all_tracking.append(a_series, ignore_index=True)

                    # ax.annotate(str(track_ind),(int(center_pos[0]), int(center_pos[1])))
                    if with_tail:
                        for inc in range(step_inc, steps, step_inc):
                            tail_ind = rel_frame-inc
                            if tail_ind >= 0:
                                pos = track['track'][tail_ind]
                                distance = calc_distance(pos, center_pos)
                                if distance < circle_radius:
                                    continue
                                cv2.circle(im, (int(pos[0]), int(pos[1])), 2, colors[track_ind], -1)
                            else:
                                break
                
    video_writer.write(im)
video_writer.release()
all_tracking.to_csv(metadata.folder_output + 'tracks_' + vid_name + '.csv', index=False)


# %%
files_in_directory = os.listdir(metadata.folder_images)
filtered_files = [file for file in files_in_directory if file.endswith(".jpg")]
for file in filtered_files:
    path_to_file = os.path.join(metadata.folder_images, file)
    os.remove(path_to_file)


