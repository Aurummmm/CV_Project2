import cv2
import numpy as np
cv2.__version__
cap = cv2.VideoCapture("video2.mp4")
trackers = cv2.legacy.MultiTracker_create()
while True:
    _, frame = cap.read()
    if frame is None:
        break
    frame = cv2.resize(frame, (600, int(frame.shape[0] * 600 / frame.shape[1])), cv2.INTER_AREA)
    (success, boxes) = trackers.update(frame)
    for box in boxes:
        (x1, y1, w, h) = box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (0, 255, 0))
    cv2.imshow("video", frame)
    key = cv2.waitKey(30)
    if key == ord('s'):
        box = cv2.selectROI("video", frame, True, False)
        #trackers.add(cv2.legacy.TrackerKCF_create(), frame, box)
        #trackers.add(cv2.legacy.TrackerBoosting_create(), frame, box)
        trackers.add(cv2.legacy.TrackerMOSSE_create(), frame, box)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
