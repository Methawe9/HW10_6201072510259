import numpy as np
import cv2

cap = cv2.VideoCapture('grandcentral.mp4')
feature_params = dict( maxCorners = 900,
                    qualityLevel = 0.03,
                    minDistance = 7,
                    blockSize = 50 )
lk_params = dict( winSize  = (21,21),
                maxLevel = 3,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
colors = np.random.randint(0, 125, (900, 3)) 
ret, old_frame = cap.read()
premask = np.zeros(old_frame.shape).astype(old_frame.dtype)
myROI = [(720,476), (530,25 ), (169, 25), (0,476)]  # (x, y)
cv2.fillPoly(premask, [np.array(myROI)], (255,255,255))
old_frame = cv2.bitwise_and(old_frame, premask)
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params) 
mask = np.zeros_like(old_frame)

while cap.isOpened() :
    ret, frameori = cap.read()
    ret, frame = cap.read()

    premask = np.zeros(frame.shape).astype(frame.dtype)
    myROI = [(720,476), (530,25 ), (169, 25), (0,476)]  # (x, y)
    cv2.fillPoly(premask, [np.array(myROI)], (255,255,255))
    frame = cv2.bitwise_and(frame, premask)

    if ret :
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel().astype(int)
            c, d = old.ravel().astype(int)
            mask = cv2.line(mask, (a, b), (c, d), (0,0,125), 1)
            frame = cv2.circle(frame, (a,b), 5, (125,0,0), -1)
        compare_img = cv2.hconcat([frame, mask])
        disp_img = cv2.addWeighted(mask,0.9,frame,0.7,0)
        disp_img = cv2.addWeighted(disp_img,1,frameori,0.6,0)
        cv2.imshow('frame', disp_img)
        key = cv2.waitKey(27) & 0xFF
        if key == 27 or key == ord('q') :
            break
        elif key == ord('c') : # clear mask
            mask = np.zeros_like(old_frame)
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        else :
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
    else :
        break
cap.release()
cv2.destroyAllWindows()
