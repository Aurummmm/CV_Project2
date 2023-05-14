import numpy as np
import cv2
import platform
import sys
import time

# hsv tennis ball yellow
upper = np.array([180, 30, 255])
lower = np.array([0, 0, 221])

# hsv cochonnet red-orange ish
# upper = np.array([150, 105, 255])
# lower = np.array([0, 0, 0,])


state = np.matrix('0.0;0.0;0.0;0.0')  # x, y, xd, yd,

# P and Q matrices for EKF
P = np.matrix('10.0,0.0,0.0,0.0; \
				0.0,10.0,0.0,0.0; \
					0.0,0.0,10.0,0.0; \
						0.0,0.0,0.0,10.0')

Q = np.matrix('2.0,0.0,0.0,0.0; \
				0.0,2.0,0.0,0.0; \
					0.0,0.0,2.0,0.0; \
						0.0,0.0,0.0,2.0')

measurement = np.matrix('0;0')

debug_print = False
data = np.loadtxt('video/basketball/groundtruth.txt', dtype=np.float32, delimiter=',')


def find_ball(frame0,frame1,i,box):
    #bbox = np.array([[int(data[i][0])], [int(data[i][1])], [int(data[i][2])], [int(data[i][3])]])
    # 定义目标区域的范围
    x1, y1 = int(box[0]), int(box[1])  # 区域左上角坐标
    x2, y2 = int(x1 + box[3]), int(y1 + box[3])  # 区域右下角坐标
    target_image=frame0
    current_image=frame1
    # 提取目标区域的像素值
    target_region = target_image[y1:y2, x1:x2]

    # 将目标区域转换为灰度图像
    gray_target = cv2.cvtColor(target_region, cv2.COLOR_BGR2GRAY)

    # 计算目标区域的直方图
    target_hist = cv2.calcHist([gray_target], [0], None, [256], [0, 256])

    # 将当前图像转换为灰度图像
    gray_current = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)

    # 在当前图像中寻找与目标区域直方图相似的区域
    result = cv2.matchTemplate(gray_current, gray_target, cv2.TM_CCOEFF_NORMED)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc
    x1, y1 = int(max_loc[0]), int(max_loc[1])
    return [x1,y1,box[2],box[3]]

def run_EKF_model(state, P, Q, dt):
    # model is
    # X(0) = X(0) + X(2)*dt
    # X(1) = X(1) + X(3)*dt
    # X(2) = X(2)
    # X(3) = X(3)
    # it has no input, so Ju = 0

    state[0] = state[0] + dt * state[2]
    state[1] = state[1] + dt * state[3]
    state[2] = state[2]
    state[3] = state[3]

    # covariance matrix gets updated through
    # P = J*P*trans(J) + Q
    # where J = [1, 0, dt, 0;
    #			 0, 1, 0, dt;
    #			 0, 0, 1, 0;
    #			 0, 0, 0, 1]

    J = np.matrix('1.0,0.0,0.0,0.0;\
				   0.0,1.0,0.0,0.0;\
				   0.0,0.0,1.0,0.0;\
				   0.0,0.0,0.0,1.0')
    J[0, 2] = dt
    J[1, 3] = dt

    P = J * P * (J.transpose()) + Q

    return state, P, J


def run_EKF_measurement(state, measurement, P):
    # Observation is (x,y) = (X(0), X(1))
    # so H is very simple...
    # H = [1, 0, 0, 0, 0, 0;
    # 		0, 1, 0, 0, 0, 0]

    # sigma_x and sigma_y are pretty decent...
    # R = [sigma_x, 0;
    #		0, sigma_y]

    H = np.matrix('1.0,0.0,0.0,0.0; \
				   0.0,1.0,0.0,0.0')

    R = np.matrix('5.0,0.0;\
					0.0,5.0')

    z = measurement - H * state
    HPH = H * P * (H.transpose())
    S = HPH + R
    invS = np.linalg.inv(S)
    K = P * (H.transpose()) * np.linalg.inv(S)

    state = state + K * z
    P = P - P * (H.transpose()) * np.linalg.inv(S) * H * P

    if (debug_print == True):
        print('running new measurement')
        print('norm P is ')
        print
        np.linalg.norm(P)

        print('z is ')
        print
        z.transpose()

        print('HPH is ')
        print
        HPH
        print('S is ')
        print
        S

        print('invS is ')
        print
        invS

        print('PH is ')
        print
        P * (H.transpose())

        print('K ')
        print
        K

    return state, P


np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

# print basic info
print('python ' + platform.python_version())
print('opencv ' + cv2.__version__)
print('numpy ' + np.version.version)

# open camera
video_path = "video2.mp4"
cap = cv2.VideoCapture(video_path)
#cap = cv2.VideoCapture(0)
'''# find a way to fix camera settings...
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
print(cap.get(cv2.CAP_PROP_BRIGHTNESS))
print
cap.get(cv2.CAP_PROP_CONTRAST)'''
# cap.set(cv2.CAP_PROP_BRIGHTNESS, 20)
# cap.set(cv2.CAP_PROP_CONTRAST, 0.1)


prev_time = time.time()
i = 0
ret, frame = cap.read()
box = cv2.selectROI("video", frame, True, False)
bbox = box
last_frame=frame
while (True):
    now_time = time.time()
    dt = now_time - prev_time

    # run the model every 0.01 s
    if (dt > 0.01):
        prev_time = now_time

        state, P, J = run_EKF_model(state, P, Q, dt)


    # read camera
    ret, frame = cap.read()
    if ret == True:
        # process
        #bbox = find_ball(frame,i)
        if bbox[0]<0:
            bbox[0]=0
        if bbox[1]<0:
            bbox[1]=0
        bbox = find_ball(last_frame,frame,0,bbox)
        x=measurement[0] = bbox[0]+bbox[2]/2
        y=measurement[1] = bbox[1]+bbox[3]/2

        cimg=frame
        if (measurement[0] != 0) and (measurement[1] != 0):
            state, P = run_EKF_measurement(state, measurement, P)
        bbox[0]=int(state[0]-bbox[2]/2)
        bbox[1]=int(state[1]-bbox[3]/2)
        if (x != 0):
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (255, 0, 0))


        cv2.imshow('video', cimg)
        i+=1
        last_frame=frame
        key = cv2.waitKey(30)
    # close
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# clean up
cap.release()
cv2.destroyAllWindows()




