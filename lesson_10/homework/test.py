import time
import cv2
import numpy as np
from matplotlib import pyplot as plt

vid = cv2.VideoCapture('road_1min.mp4')
start_time = time.time()
frames = []
n_frames = 0

while vid.isOpened():
    ret, frame = vid.read()
    if ret == True:
        n_frames += 1
        if n_frames % 20 == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    else:
        break
vid.release()

frames = np.array(frames)
result = np.median(frames, axis=0)
print('Elapsed time: ', time.time() - start_time)

plt.imshow(result.astype(np.uint8))

plt.show()