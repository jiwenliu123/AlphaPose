# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Haoyi Zhu,Hao-Shu Fang
# -----------------------------------------------------

"""Script for single-image demo."""
import argparse#argparse 是 Python 内置的一个用于命令项选项与参数解析的模块，通过在程序中定义好我们需要的参数，argparse 将会从 sys.argv 中解析出这些参数，并自动生成帮助和使用信息
import torch#引用torch模块
import os#引入对文件操作的模块
import platform#引入platform模块对系统的信息进行操作 
import sys
import math#引入math函数，提供所需数学方法
import time#引入对时间处理的模块

import cv2#调用OpenCV
import numpy as np#引入数据计算库

from alphapose.utils.transforms import get_func_heatmap_to_coord#获取功能热图到坐标
from alphapose.utils.pPose_nms import pose_nms
from alphapose.utils.presets import SimpleTransform#引入简单变换的模块
from alphapose.utils.transforms import flip, flip_heatmap#引入反转热图的模块
from alphapose.models import builder
from alphapose.utils.config import update_config#引入更新配置的模块
from detector.apis import get_detector#获取探测器
from alphapose.utils.vis import getTime#引入获取时间的模块

"""----------------------------- Demo options -----------------------------"""
parser = argparse.ArgumentParser(description='AlphaPose Single-Image Demo')#创建一个解析对象 方法参数须知：一般我们只选择用description
parser.add_argument('--cfg', type=str, required=True,
                    help='experiment configure file name')#向该对象中添加你要关注的命令行参数和选项

parser.add_argument('--checkpoint', type=str, required=True,
                    help='checkpoint file name')#required: 表示这个参数是否一定需要设置
parser.add_argument('--detector', dest='detector',
                    help='detector name', default="yolo")#type：参数类型
parser.add_argument('--image', dest='inputimg',
                    help='image-name', default="")#default：没有设置值情况下的默认参数
parser.add_argument('--save_img', default=False, action='store_true',
                    help='save result as image')#把结果按照图片的形式存储
parser.add_argument('--vis', default=False, action='store_true',
                    help='visualize image')#将图片可视化
parser.add_argument('--showbox', default=False, action='store_true',
                    help='visualize human bbox')#将人物方框可视化
parser.add_argument('--profile', default=False, action='store_true',
                    help='add speed profiling at screen output')#在屏幕输出处添加速度分析
parser.add_argument('--format', type=str,
                    help='save in the format of cmu or coco or openpose, option: coco/cmu/open')#以cmu或coco或openpose格式保存
parser.add_argument('--min_box_area', type=int, default=0,
                    help='min box area to filter out')#要过滤掉的最小盒子面积
parser.add_argument('--eval', dest='eval', default=False, action='store_true',
                    help='save the result json as coco format, using image index(int) instead of image name(str)')#将结果json保存为coco格式，使用图像索引（int）而不是图像名称（str）
parser.add_argument('--gpus', type=str, dest='gpus', default="0",
                    help='choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')#按索引选择要使用的cuda设备，并输入逗号以使用多GPU
parser.add_argument('--flip', default=False, action='store_true',
                    help='enable flip testing')#启用翻转测试
parser.add_argument('--debug', default=False, action='store_true',
                    help='print detail information')#打印出细节信息
parser.add_argument('--vis_fast', dest='vis_fast',
                    help='use fast rendering', action='store_true', default=False)#使用快速渲染
"""----------------------------- Tracking options -----------------------------"""
parser.add_argument('--pose_flow', dest='pose_flow',
                    help='track humans in video with PoseFlow', action='store_true', default=False)#用PoseFlow在视频中跟踪人类
parser.add_argument('--pose_track', dest='pose_track',
                    help='track humans in video with reid', action='store_true', default=False)

args = parser.parse_args()#进行解析
cfg = update_config(args.cfg)#更新配置

args.gpus = [int(args.gpus[0])] if torch.cuda.device_count() >= 1 else [-1]
args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
args.tracking = args.pose_track or args.pose_flow or args.detector=='tracker'

class DetectionLoader():#检测装载机
    def __init__(self, detector, cfg, opt):
        self.cfg = cfg
        self.opt = opt
        self.device = opt.device
        self.detector = detector

        self._input_size = cfg.DATA_PRESET.IMAGE_SIZE#将输入大小设置为预设图片大小
        self._output_size = cfg.DATA_PRESET.HEATMAP_SIZE#将输出大小设置为热图大小

        self._sigma = cfg.DATA_PRESET.SIGMA

        pose_dataset = builder.retrieve_dataset(self.cfg.DATASET.TRAIN)#检索数据集
        if cfg.DATA_PRESET.TYPE == 'simple':
            self.transformation = SimpleTransform(
                pose_dataset, scale_factor=0,
                input_size=self._input_size,
                output_size=self._output_size,
                rot=0, sigma=self._sigma,
                train=False, add_dpg=False, gpu_device=self.device)#数据转换

        self.image = (None, None, None, None)
        self.det = (None, None, None, None, None, None, None)
        self.pose = (None, None, None, None, None, None, None)

    def process(self, im_name, image):
       #开始对目标检测的图像进行预处理
        self.image_preprocess(im_name, image)
       #开始在图像中检测人类
        self.image_detection()
        #开始对裁剪的人体图像进行后处理以进行姿势估计
        self.image_postprocess()
        return self

    def image_preprocess(self, im_name, image):#图像预处理
        #预期的图像形状，如（1,3，h，w）或（3，h，w）
        img = self.detector.image_preprocess(image)
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        #如果图像形状（3，h，w），则在前面添加一个尺寸用于批处理
        if img.dim() == 3:
            img = img.unsqueeze(0)
        orig_img = image # scipy.misc.imread(im_name_k, mode='RGB') is depreciated
        im_dim = orig_img.shape[1], orig_img.shape[0]

        im_name = os.path.basename(im_name)

        with torch.no_grad():
            im_dim = torch.FloatTensor(im_dim).repeat(1, 2)

        self.image = (img, orig_img, im_name, im_dim)

    def image_detection(self):#图像质量检测
        imgs, orig_imgs, im_names, im_dim_list = self.image
        if imgs is None:
            self.det = (None, None, None, None, None, None, None)
            return

        with torch.no_grad():#上下文管理器
            dets = self.detector.images_detection(imgs, im_dim_list)
            if isinstance(dets, int) or dets.shape[0] == 0:
                self.det = (orig_imgs, im_names, None, None, None, None, None)
                return
            if isinstance(dets, np.ndarray):#判断np.ndarray是否为dets型
                dets = torch.from_numpy(dets)
            dets = dets.cpu()
            boxes = dets[:, 1:5]#boxes截取第一到第五个片段
            scores = dets[:, 5:6]
            ids = torch.zeros(scores.shape)

        boxes = boxes[dets[:, 0] == 0]
        if isinstance(boxes, int) or boxes.shape[0] == 0:
            self.det = (orig_imgs, im_names, None, None, None, None, None)
            return
        inps = torch.zeros(boxes.size(0), 3, *self._input_size)
        cropped_boxes = torch.zeros(boxes.size(0), 4)

        self.det = (orig_imgs, im_names, boxes, scores[dets[:, 0] == 0], ids[dets[:, 0] == 0], inps, cropped_boxes)

    def image_postprocess(self):#对图像进行后期处理
        with torch.no_grad():
            (orig_img, im_name, boxes, scores, ids, inps, cropped_boxes) = self.det
            if orig_img is None:
                self.pose = (None, None, None, None, None, None, None)
                return
            if boxes is None or boxes.nelement() == 0:
                self.pose = (None, orig_img, im_name, boxes, scores, ids, None)
                return

            for i, box in enumerate(boxes):
                inps[i], cropped_box = self.transformation.test_transform(orig_img, box)
                cropped_boxes[i] = torch.FloatTensor(cropped_box)

            self.pose = (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes)

    def read(self):
        return self.pose


class DataWriter():
    def __init__(self, cfg, opt):
        self.cfg = cfg
        self.opt = opt

        self.eval_joints = list(range(cfg.DATA_PRESET.NUM_JOINTS))
        self.heatmap_to_coord = get_func_heatmap_to_coord(cfg)
        self.item = (None, None, None, None, None, None, None)

    def start(self):
        # start to read pose estimation results
        return self.update()

    def update(self):
        norm_type = self.cfg.LOSS.get('NORM_TYPE', None)
        hm_size = self.cfg.DATA_PRESET.HEATMAP_SIZE

        # get item
        (boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name) = self.item
        if orig_img is None:
            return None
        # image channel RGB->BGR
        orig_img = np.array(orig_img, dtype=np.uint8)[:, :, ::-1]
        self.orig_img = orig_img
        if boxes is None or len(boxes) == 0:
            return None
        else:
            # location prediction (n, kp, 2) | score prediction (n, kp, 1)
            assert hm_data.dim() == 4
            if hm_data.size()[1] == 136:
                self.eval_joints = [*range(0,136)]
            elif hm_data.size()[1] == 26:
                self.eval_joints = [*range(0,26)]
            pose_coords = []
            pose_scores = []

            for i in range(hm_data.shape[0]):
                bbox = cropped_boxes[i].tolist()
                pose_coord, pose_score = self.heatmap_to_coord(hm_data[i][self.eval_joints], bbox, hm_shape=hm_size, norm_type=norm_type)
                pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
                pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))
            preds_img = torch.cat(pose_coords)
            preds_scores = torch.cat(pose_scores)

            boxes, scores, ids, preds_img, preds_scores, pick_ids = \
                pose_nms(boxes, scores, ids, preds_img, preds_scores, self.opt.min_box_area)

            _result = []
            for k in range(len(scores)):
                _result.append(
                    {
                        'keypoints':preds_img[k],
                        'kp_score':preds_scores[k],
                        'proposal_score': torch.mean(preds_scores[k]) + scores[k] + 1.25 * max(preds_scores[k]),
                        'idx':ids[k],
                        'bbox':[boxes[k][0], boxes[k][1], boxes[k][2]-boxes[k][0],boxes[k][3]-boxes[k][1]] 
                    }
                )

            result = {
                'imgname': im_name,
                'result': _result
            }

            if hm_data.size()[1] == 49:
                from alphapose.utils.vis import vis_frame_dense as vis_frame
            elif self.opt.vis_fast:
                from alphapose.utils.vis import vis_frame_fast as vis_frame
            else:
                from alphapose.utils.vis import vis_frame
            self.vis_frame = vis_frame

        return result

    def save(self, boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name):
        self.item = (boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name)

class SingleImageAlphaPose():
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg

        # Load pose model
        self.pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

        print(f'Loading pose model from {args.checkpoint}...')
        self.pose_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
        self.pose_dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)

        self.pose_model.to(args.device)
        self.pose_model.eval()
        
        self.det_loader = DetectionLoader(get_detector(self.args), self.cfg, self.args)

    def process(self, im_name, image):
        # Init data writer
        self.writer = DataWriter(self.cfg, self.args)

        runtime_profile = {
            'dt': [],
            'pt': [],
            'pn': []
        }
        pose = None
        try:
            start_time = getTime()
            with torch.no_grad():
                (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = self.det_loader.process(im_name, image).read()
                if orig_img is None:
                    raise Exception("no image is given")
                if boxes is None or boxes.nelement() == 0:
                    if self.args.profile:
                        ckpt_time, det_time = getTime(start_time)
                        runtime_profile['dt'].append(det_time)
                    self.writer.save(None, None, None, None, None, orig_img, im_name)
                    if self.args.profile:
                        ckpt_time, pose_time = getTime(ckpt_time)
                        runtime_profile['pt'].append(pose_time)
                    pose = self.writer.start()
                    if self.args.profile:
                        ckpt_time, post_time = getTime(ckpt_time)
                        runtime_profile['pn'].append(post_time)
                else:
                    if self.args.profile:
                        ckpt_time, det_time = getTime(start_time)
                        runtime_profile['dt'].append(det_time)
                    # Pose Estimation
                    inps = inps.to(self.args.device)
                    if self.args.flip:
                        inps = torch.cat((inps, flip(inps)))
                    hm = self.pose_model(inps)
                    if self.args.flip:
                        hm_flip = flip_heatmap(hm[int(len(hm) / 2):], self.pose_dataset.joint_pairs, shift=True)
                        hm = (hm[0:int(len(hm) / 2)] + hm_flip) / 2
                    if self.args.profile:
                        ckpt_time, pose_time = getTime(ckpt_time)
                        runtime_profile['pt'].append(pose_time)
                    hm = hm.cpu()
                    self.writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, im_name)
                    pose = self.writer.start()
                    if self.args.profile:
                        ckpt_time, post_time = getTime(ckpt_time)
                        runtime_profile['pn'].append(post_time)

            if self.args.profile:
                print(
                    'det time: {dt:.4f} | pose time: {pt:.4f} | post processing: {pn:.4f}'.format(
                        dt=np.mean(runtime_profile['dt']), pt=np.mean(runtime_profile['pt']), pn=np.mean(runtime_profile['pn']))
                )
            print('===========================> Finish Model Running.')
        except Exception as e:
            print(repr(e))
            print('An error as above occurs when processing the images, please check it')
            pass
        except KeyboardInterrupt:
            print('===========================> Finish Model Running.')

        return pose

    def getImg(self):
        return self.writer.orig_img

    def vis(self, image, pose):
        if pose is not None:
            image = self.writer.vis_frame(image, pose, self.writer.opt)
        return image

    def writeJson(self, final_result, outputpath, form='coco', for_eval=False):
        from alphapose.utils.pPose_nms import write_json
        write_json(final_result, outputpath, form=form, for_eval=for_eval)
        print("Results have been written to json.")

def example():
    outputpath = "examples/res/"
    if not os.path.exists(outputpath + '/vis'):
        os.mkdir(outputpath + '/vis')

    demo = SingleImageAlphaPose(args, cfg)
    im_name = args.inputimg    # the path to the target image
    image = cv2.cvtColor(cv2.imread(im_name), cv2.COLOR_BGR2RGB)
    pose = demo.process(im_name, image)
    img = demo.getImg()     # or you can just use: img = cv2.imread(image)
    img = demo.vis(img, pose)   # visulize the pose result
    cv2.imwrite(os.path.join(outputpath, 'vis', os.path.basename(im_name)), img)
    
    # if you want to vis the img:
    # cv2.imshow("AlphaPose Demo", img)
    # cv2.waitKey(30)

    # write the result to json:
    result = [pose]
    demo.writeJson(result, outputpath, form=args.format, for_eval=args.eval)

if __name__ == "__main__":
    example()
