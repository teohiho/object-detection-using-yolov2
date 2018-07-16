# python flow --model cfg/yolo-voc-1.cfg --load bin/yolo-voc.weights --train --annotation annotations3 --dataset images3 --gpu 0.8

import cv2
from darkflow.net.build import TFNet
from kalman.sort import Sort
import time
import csv
import numpy as np
from math import sqrt
import imutils


#test2.mp4
corners1 = (400, 360)
corners2 = (350, 720)
corners4 = (680, 360)
corners3 = (1000, 720)


# train3.mp4
# corners1 = (470, 450)
# corners2 = (400, 500)
# corners3 = (950, 500)
# corners4 = (950, 450)

def draw_ROI(frame):
    cv2.line(frame, corners1, corners2, (255,255,0), 2)
    cv2.line(frame, corners1, corners4, (255,255,0), 2)
    cv2.line(frame, corners2, corners3, (255,255,0), 2)
    cv2.line(frame, corners3, corners4, (255,255,0), 2)

def point_center(tl, br):
    return ((tl[0] + br[0])/2, (tl[1]+br[1])/2 - 0.25*(br[1]-tl[1]))

def sign(corners1, corners2, x, y):
    return (corners2[1] - corners1[1])*(x - corners1[0]) - (corners2[0]- corners1[0])*(y - corners1[1])

def distance(point1, point2):
    return sqrt((point2[0]-point1[0])*(point2[0]-point1[0]) + (point2[1]-point1[1])*(point2[1]-point1[1]))


# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

color = np.random.randint(0,255,(100,3))


# read video
capture = cv2.VideoCapture('data/video/test1.mp4')
ret, old_frame = capture.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = np.array([[[0, 0]]], dtype = np.float32)
mask = np.zeros_like(old_frame)
number_frame = 0
object = 0

option = {
    'model': "cfg/yolo-voc-1.cfg",
    'load': 8000,
    'threshold': 0.3,
    'gpu': 0.25
}
tfnet = TFNet(option)


while (capture.isOpened()):
    stime = time.time()
    ret, frame = capture.read()
    results = []
    if ret:
        if number_frame % 1 == 0:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # get update the vector! 
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            good_new = p1
            good_old = p0
            number_frame += 1

            # draw the ROI 
            cv2.line(frame, corners1, corners2, (255,255,0), 2)
            cv2.line(frame, corners1, corners4, (255,255,0), 2)
            cv2.line(frame, corners2, corners3, (255,255,0), 2)
            cv2.line(frame, corners3, corners4, (255,255,0), 2)

            # predict and return object is detected
            results = tfnet.return_predict(frame)
            for result in results:
                tl = (result['topleft']['x'], result['topleft']['y'])
                br = (result['bottomright']['x'], result['bottomright']['y'])
                label = result['label']
                point_flow = point_center(tl, br)

                # check the point_flow in ROI!
                check = sign(corners1, corners4, point_flow[0], point_flow[1])< 0 and sign(corners1, corners2, point_flow[0], point_flow[1]) > 0 and sign(corners3, corners4, point_flow[0], point_flow[1]) > 0 and point_flow[1] < 550
                if check:
                    chk = False
                    for i in p1:
                        # distance between point_flow and all point in p1
                        print(distance(point_flow, (i[0][0], i[0][1])))
                        if distance(point_flow, (i[0][0], i[0][1])) < 50:
                            chk = True
                    if not chk:
                        p2 = np.array([[[point_flow[0], point_flow[1]]]], dtype=np.float32)
                        p1 = np.append(p1, p2, axis=0)
                        object+=1
                
                size_of_p1 = len(p1)
                index_cur = 0
                while index_cur < size_of_p1:
                    if p1[index_cur][0][1] > 600:
                        p1 = np.delete(p1, index_cur, axis=0)
                        size_of_p1-=1
                    else: index_cur+=1

            # draw the point object is dectected
            for i,(new,old) in enumerate(zip(good_new,good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
                mask = cv2.line(mask, (a,b), (c,d), color[i].tolist(), 2)
                frame = cv2.circle(frame, (a,b), 5, color[i].tolist(), -1)
            cv2.putText(frame,str(object), (int(frame.shape[0]/2) ,int(frame.shape[1]/2)), cv2.FONT_HERSHEY_SIMPLEX, 2, color=(200,50,75), thickness=3)
            img = cv2.add(frame, mask)
            print('FPS {:.1f}'.format(1 / (time.time() - stime)))
            cv2.imshow("frame", img)

            # Update old_fray and p0 for next flow
            old_gray = frame_gray.copy()
            p0 = p1.reshape(-1,1,2)

            if len(p0) > 80:
                p0 = np.array([[[0,0]]], dtype=np.float32)
            # for two second reset mask
            if number_frame%60 == 0:
                mask = np.zeros_like(old_frame)
        # press "q" to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        capture.release()
        cv2.destroyAllWindows()
        break
