import cv2
import numpy as np
TrDict = {'csrt': cv2.TrackerCSRT_create,
         'kcf' : cv2.TrackerKCF_create,
         'boosting' : cv2.TrackerBoosting_create,
         'mil': cv2.TrackerMIL_create,
         'tld': cv2.TrackerTLD_create,
         'medianflow': cv2.TrackerMedianFlow_create,
         'mosse':cv2.TrackerMOSSE_create
          }
trackers = cv2.MultiTracker_create()
v = cv2.VideoCapture('mot.mp4')
#v = cv2.VideoCapture(0)
ret, frame = v.read()
k = 1
for i in range(k):
    cv2.imshow('Frame',frame)
    bbi = cv2.selectROI('Frame',frame)
    #print(bbi)

    tracker_i = TrDict['csrt']()
    trackers.add(tracker_i,frame,bbi)
frameNumber = 2
baseDir = './TrackingResults'
while True:
    ret, frame = v.read()
    if not ret:
        break
    (success, boxes) = trackers.update(frame)
    #np.savetxt(baseDir + '/frame_' + str(frameNumber) + '.txt', boxes, fmt='%f')
    #frameNumber += 1
    output = []
    id = 1
    for box in boxes:
        (x, y, w, h) = [int(a) for a in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        output.append([x + w/2, y + h/2, id])
        id += 1
    # the output is now an array with centroid coordinates and box id
    np.savetxt(baseDir + '/frame_' + str(frameNumber) + '.txt', output, fmt='%f')
    frameNumber += 1
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
v.release()
cv2.destroyAllWindows()