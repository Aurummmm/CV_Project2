from got10k.trackers import Tracker
import glob, cv2, torch
import numpy as np
from os.path import realpath, dirname, join

global trackers

class IdentityTracker(Tracker):
    def __init__(self):
        super(IdentityTracker, self).__init__(
            name='IdentityTracker',  # tracker name
            is_deterministic=True  # stochastic (False) or deterministic (True)
        )

    def init(self, image, box):
        self.box = box
        global trackers
        trackers = cv2.legacy.MultiTracker_create()
        # image and init box
        img_pil2cv = np.array(image)
        img_pil2cv = cv2.cvtColor(img_pil2cv, cv2.COLOR_RGB2BGR)
        # HxWxC 读取图像
        self.image = img_pil2cv
        frame=self.image
        trackers.add(cv2.legacy.TrackerMOSSE_create(), frame, self.box)


    def update(self, image):
        # image and init box
        img_pil2cv = np.array(image)
        img_pil2cv = cv2.cvtColor(img_pil2cv, cv2.COLOR_RGB2BGR)
        # HxWxC 读取图像
        self.image = img_pil2cv
        frame = self.image
        global trackers
        #frame = cv2.resize(frame, (600, int(frame.shape[0] * 600 / frame.shape[1])), cv2.INTER_AREA)
        (success, boxes) = trackers.update(frame)
        self.box=boxes[0]
        return self.box

from got10k.experiments import ExperimentGOT10k,ExperimentVOT

# ... tracker definition ...

# instantiate a tracker
tracker = IdentityTracker()

# setup experiment (validation subset)
experiment = ExperimentGOT10k(
    root_dir='test_data',    # GOT-10k's root directory
    subset='val',               # 'train' | 'val' | 'test'
    result_dir='results_GOT10k',       # where to store tracking results
    report_dir='reports_GOT10k'        # where to store evaluation reports
)
#experiment.run(tracker, visualize=True)

# report tracking performance
experiment.report(['MOSSE','EKF_Tracker','DaSiamPRN'])
