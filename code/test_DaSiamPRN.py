from got10k.trackers import Tracker
import glob, cv2, torch
import numpy as np
from os.path import realpath, dirname, join

from net import SiamRPNvot,SiamRPNBIG
from run_SiamRPN import SiamRPN_init, SiamRPN_track
from utils import get_axis_aligned_bbox, cxy_wh_2_rect
global state
class IdentityTracker(Tracker):
    def __init__(self):
        super(IdentityTracker, self).__init__(
            name='IdentityTracker',  # tracker name
            is_deterministic=True  # stochastic (False) or deterministic (True)
        )

    def init(self, image, box):
        self.box = box
        # load net
        # 初始化网络模型
        net = SiamRPNBIG()
        # 网络模型参数读取
        net.load_state_dict(torch.load(join(realpath(dirname(__file__)), 'SiamRPNBIG.model')))
        # 将其放在GPU上运行
        net.eval().cuda()
        # image and init box
        img_pil2cv = np.array(image)
        img_pil2cv = cv2.cvtColor(img_pil2cv, cv2.COLOR_RGB2BGR)
        # HxWxC 读取图像
        self.image=img_pil2cv

        x, y, w, h = box


        # tracker init
        # 目标位置转化 将目标的中心位置标记为target_pos 尺度大小标记为target_sz
        target_pos = np.array([x + w / 2, y + h / 2])
        target_sz = np.array([w, h])

        # 初始化网络
        global state
        state = SiamRPN_init(self.image, target_pos, target_sz, net)


    def update(self, image):
        # 读取图片
        img_pil2cv = np.array(image)
        img_pil2cv = cv2.cvtColor(img_pil2cv, cv2.COLOR_RGB2BGR)
        # HxWxC 读取图像
        image = img_pil2cv
        im = image
        tic = cv2.getTickCount()
        # track
        global state
        state = SiamRPN_track(state, im)

        res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
        res = [int(l) for l in res]
        # 跟踪框绘制
        #cv2.rectangle(im, (res[0], res[1]), (res[0] + res[2], res[1] + res[3]), (0, 255, 255), 3)
        # 显示跟踪结果
        #cv2.imshow('SiamRPN', im)
        self.box=np.array(res)
        #self.state=state
        return self.box

from got10k.experiments import ExperimentGOT10k,ExperimentVOT

# ... tracker definition ...

# instantiate a tracker
tracker = IdentityTracker()
from got10k.experiments import ExperimentGOT10k

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
experiment.run(tracker, visualize=True)

# report tracking performance
experiment.report([tracker.name])

