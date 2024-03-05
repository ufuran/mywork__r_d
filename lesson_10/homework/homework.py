import cv2
import numpy as np
from matplotlib import pyplot as plt

tracker = cv2.legacy.TrackerCSRT().create()
# tracker = cv2.legacy.TrackerKCF().create()
# vid = cv2.VideoCapture('road_10sec.mp4')
vid = cv2.VideoCapture('road_1min.mp4')

# tick_f = 40
# x = 76
# y = 538
# w = 130
# h = 150

tick_f = 13
x = 360
y = 570
w = 120
h = 120

for i in range(tick_f):
    ret, frame = vid.read()

copy_frame = frame.copy()

cv2.rectangle(copy_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
plt.imshow(cv2.cvtColor(copy_frame, cv2.COLOR_BGR2RGB))
plt.show()



bbox = (x, y, w, h)
ok = tracker.init(frame, bbox)
while vid.isOpened():
    ret, frame = vid.read()
    if not ret:
        break
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
    ok, bbox = tracker.update(img)
    print(ok, bbox)

    x1, y1 = int(bbox[0]), int(bbox[1])
    width, height = int(bbox[2]), int(bbox[3])

    cv2.rectangle(img, (x1, y1), (x1+width, y1+height), (0, 255, 0), 2)
    cv2.imshow('frame', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cv2.waitKey(1)


'''
Do you see any differences? If so, what are they?
Does one tracker perform better than the other? In what way?

kfc tracker быстрее, но менее точный, чем csrt. так же kfc трекер не работает с масштабированием, в отличии от csrt трекера. 
на более слабом делезе возможно логичнее использовать kfc трекер, так как он требует меньше ресурсов. но все зависит от задачи :)

для себя я понял что kfc tracker более оптимальный если нам надо что-то real time отслеживать, 
а csrt если нам надо что-то более точное отслеживать. опять такие впирается в ресурсы которые доступны и задачу которую надо решить :)
'''