from got10k.trackers import Tracker
import glob, cv2, torch
import numpy as np
from os.path import realpath, dirname, join
import time

state = np.matrix('0.0;0.0;0.0;0.0')  # x, y, xd, yd,
global P,Q
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
data = np.loadtxt('test_data/val/GOT-10k_Val_000095/groundtruth.txt', dtype=np.float32, delimiter=',')


def find_top_values(matrix, num_values):
    # 使用 argpartition 函数找到最大的 num_values 个值的索引
    flattened_indices = np.argpartition(matrix.flatten(), -num_values)[-num_values:]

    # 将一维索引转换为二维坐标
    indices = np.unravel_index(flattened_indices, matrix.shape)

    # 获取对应的值
    values = matrix[indices]

    return values, indices
def find_area(frame0,frame1,i,box):
    x1, y1 = int(box[0]), int(box[1])  # 区域左上角坐标
    x2, y2 = int(x1 + box[3]), int(y1 + box[3])  # 区域右下角坐标
    target_image=frame0
    current_image=frame1
    # 提取目标区域的像素值
    target_region = target_image[y1:y2, x1:x2]
    # 将目标区域转换为灰度图像
    gray_target = cv2.cvtColor(target_region, cv2.COLOR_BGR2GRAY)
    # 将当前图像转换为灰度图像
    gray_current = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
    # 在当前图像中寻找与目标区域直方图相似的区域
    result = cv2.matchTemplate(gray_current, gray_target, cv2.TM_CCOEFF_NORMED)
    v,i=find_top_values(result,10)
    index=np.array(i)
    points=index.transpose()
    target_point=np.array([[int(box[0]),int(box[1])]])
    target_point=np.int64(target_point)
    distances = np.linalg.norm(points - target_point, axis=1)  # 计算目标点与每个点之间的距离
    nearest_index = np.argmin(distances)  # 找到最小距离对应的索引
    nearest_point = points[nearest_index]  # 获取最近点的坐标
    max_loc=[nearest_point[1],nearest_point[0]]
    x1, y1 = int(max_loc[0]), int(max_loc[1])
    return [[x1], [y1], box[2], box[3]]


def run_EKF_model(state, P, Q, dt):

    state[0] = state[0] + dt * state[2]
    state[1] = state[1] + dt * state[3]
    state[2] = state[2]
    state[3] = state[3]

    J = np.matrix('1.0,0.0,0.0,0.0;\
				   0.0,1.0,0.0,0.0;\
				   0.0,0.0,1.0,0.0;\
				   0.0,0.0,0.0,1.0')
    J[0, 2] = dt
    J[1, 3] = dt

    P = J * P * (J.transpose()) + Q

    return state, P, J


def run_EKF_measurement(state, measurement, P):

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


global i,last_frame,state1,bbox,prev_time

i = 0
state1 = np.matrix('0.0;0.0;0.0;0.0')
class EKF_Tracker(Tracker):
    def __init__(self):
        super(EKF_Tracker, self).__init__(
            name='EKF_Tracker',  # tracker name
            is_deterministic=True  # stochastic (False) or deterministic (True)
        )


    def init(self, image, box):
        global bbox,last_frame,state1,prev_time
        img_pil2cv = np.array(image)
        img_pil2cv = cv2.cvtColor(img_pil2cv, cv2.COLOR_RGB2BGR)
        # HxWxC 读取图像
        self.image = img_pil2cv
        last_frame = self.image
        state1[0] = box[0] + box[2] / 2
        state1[1] = box[1] + box[3] / 2
        self.box = box
        prev_time = time.time()



    def update(self, image):
        global last_frame, state1, P,prev_time
        now_time = time.time()
        dt = now_time - prev_time

        # run the model every 0.01 s
        if (dt > 0.01):
            prev_time = now_time

            state, P, J = run_EKF_model(state1, P, Q, dt)

        img_pil2cv = np.array(image)
        img_pil2cv = cv2.cvtColor(img_pil2cv, cv2.COLOR_RGB2BGR)
        # HxWxC 读取图像
        image = img_pil2cv
        bbox=self.box
        if bbox[0]<0:
            bbox[0]=0
        if bbox[1]<0:
            bbox[1]=0

        frame=image
        bbox = find_area(last_frame, frame, 0, bbox)
        x = measurement[0] = bbox[0] + bbox[2] / 2
        y = measurement[1] = bbox[1] + bbox[3] / 2

        cimg = frame
        if (measurement[0] != 0) and (measurement[1] != 0):
            state1, P = run_EKF_measurement(state1, measurement, P)
        bbox[0] = int(state1[0] - bbox[2] / 2)
        bbox[1] = int(state1[1] - bbox[3] / 2)


        self.box=bbox
        return self.box

from got10k.experiments import ExperimentGOT10k,ExperimentVOT

# ... tracker definition ...

# instantiate a tracker
tracker = EKF_Tracker()

# setup experiment (validation subset)
experiment = ExperimentGOT10k(
    root_dir='test_data',    # GOT-10k's root directory
    subset='val',               # 'train' | 'val' | 'test'
    result_dir='results_GOT10k_EKF',       # where to store tracking results
    report_dir='reports_GOT10k_EKF'        # where to store evaluation reports
)
experiment.run(tracker, visualize=True)

# report tracking performance
experiment.report([tracker.name])