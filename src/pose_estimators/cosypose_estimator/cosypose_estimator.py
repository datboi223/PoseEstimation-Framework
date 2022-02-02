###############################################################################
##  (C) MIT License (https://github.com/ylabbe/cosypose/blob/master/LICENSE) ##
##  Code to evaluate images on the cosypose program for 6D-Pose-Estimation   ##
##  This code is composed from many files from the lib-folder of cosypose    ##
##                                                                           ##
##  Author: gezp (https://github.com/gezp) [Composition of code snippets]    ##
##                                                               ##
###############################################################################

# Partially taken from here (functionality/execution of cosypose)
# https://github.com/ylabbe/cosypose/issues/17 <- Look at the code
# New additions: putting everything inside a class and some other new methods to
#                make it usable over ROS.

import os
import sys
import time
import signal
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches # for geometric shapes
from PIL import Image
import numpy as np
from copy import deepcopy
from pathlib import Path
import yaml
import torch
import threading
lock = threading.Lock()

# ROS Imports
import roslib, rospy
from std_msgs.msg import String, Header
from sensor_msgs.msg import Image, CameraInfo
import message_filters
from cv_bridge import CvBridge, CvBridgeError
import tf
import geometry_msgs.msg
from geometry_msgs.msg import Pose, PoseStamped, PoseArray


from scipy.spatial.transform import Rotation as Rot


# importing the cosypose-"backend"
# sys.path.append(os.environ['COSYPOSE_HOME'])
# print('COSYPOSE_HOME: ', os.environ['COSYPOSE_HOME'])


# TODO: fix the dependencies/imports
from cosypose.datasets.datasets_cfg import make_scene_dataset, make_object_dataset

# Pose estimator
from cosypose.lib3d.rigid_mesh_database import MeshDataBase
from cosypose.training.pose_models_cfg import create_model_refiner, create_model_coarse
from cosypose.training.pose_models_cfg import check_update_config as check_update_config_pose
from cosypose.rendering.bullet_batch_renderer import BulletBatchRenderer
from cosypose.integrated.pose_predictor import CoarseRefinePosePredictor
from cosypose.integrated.multiview_predictor import MultiviewScenePredictor
from cosypose.datasets.wrappers.multiview_wrapper import MultiViewWrapper

# Detection
from cosypose.training.detector_models_cfg import create_model_detector
from cosypose.training.detector_models_cfg import check_update_config as check_update_config_detector
from cosypose.integrated.detector import Detector

from cosypose.evaluation.pred_runner.bop_predictions import BopPredictionRunner

from cosypose.utils.distributed import get_tmp_dir, get_rank
from cosypose.utils.distributed import init_distributed_mode

from cosypose.config import EXP_DIR, RESULTS_DIR

# From Notebook
from cosypose.rendering.bullet_scene_renderer import BulletSceneRenderer
from cosypose.visualization.singleview import render_prediction_wrt_camera
from cosypose.visualization.plotter import Plotter
from bokeh.io import export_png
from bokeh.plotting import gridplot

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# include the home-path to access the base-class
sys.path.insert(0, os.environ['EST_HOME'])
import run_pose_estimation as pe


def show_bbox_prediction(img, bbox):
    print('bbox = ', bbox.shape)
    bbox_np = np.array(bbox.detach().cpu()).reshape(-1, 2, 2)
    print('bbox(np) = ', bbox_np.shape)

    plt.imshow(img)
    ax = plt.gca()

    num_boxes = len(bbox_np)
    for i in range(num_boxes):
        pos = bbox_np[i]
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])

        dx = xmax - xmin
        dy = ymax - ymin
        rect = patches.Rectangle(xy=(xmin, ymin), width=dx, height=dy,
                                 linewidth=2,
                                 edgecolor='red',
                                 fill=False)
    ax.add_patch(rect)
    del ax
    plt.pause(0.001)

def generate_pose_array(poses, header):

    poses_np = poses.cpu().detach().numpy()
    ps = PoseArray()
    ps.header.frame_id = "/camera_link"

    for i, pose_np in enumerate(poses_np):
        print(i, ' - \n', pose_np)
        # p = PoseStamped()
        p = Pose()
        # p.header = header
        t = pose_np[:3, -1]
        r = Rot.from_matrix(pose_np[:3, :3])
        q = r.as_quat()  # Rotation as quaternion

        p.position.x = t[0]
        p.position.y = t[1]
        p.position.z = t[2]
        p.orientation.x = q[0]
        p.orientation.y = q[1]
        p.orientation.z = q[2]
        p.orientation.w = q[3]

        # p.pose.position.x = t[0]
        # p.pose.position.y = t[1]
        # p.pose.position.z = t[2]
        # p.pose.orientation.x = q[0]
        # p.pose.orientation.y = q[1]
        # p.pose.orientation.z = q[2]
        # p.pose.orientation.w = q[3]
        print(p)
        ps.poses.append(p)

    return ps

# TODO: adaption of the code to be used in class instead of separate functions
class Cosypose(pe.PoseEstimator):
    def __init__(self, parameters):
        super().__init__(parameters)
        self.im = None
        self.camera_info = None
        self.model_type = self.parameters['model_type']
        self.detector, self.pose_predictor = self.getModel(self.model_type)

        self.bridge = CvBridge()
        self.renderer = BulletSceneRenderer(urdf_ds='ycbv')

        # Init. of publisher
        # self.pose_publisher = tf.TransformBroadcaster()
        self.object_class_pub = rospy.Publisher('/object_classes', String, queue_size=1)
        self.pose_array_pub = rospy.Publisher('/object_poses', PoseArray, queue_size=1)

        # initialization of relevant subscribers (only) for cosypose
        self.rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image, queue_size=10)  # 10
        self.camera_sub = message_filters.Subscriber('/camera/color/camera_info', CameraInfo, queue_size=10)  # 10
        queue_size = 5
        slop_seconds = 0.1  # 0.1
        sync = message_filters.ApproximateTimeSynchronizer([self.rgb_sub,
                                                            self.camera_sub],
                                                           queue_size, slop_seconds)
        sync.registerCallback(self.callback)
        print('Waiting for Messages')

        self.tf_pub = None

    # def initialize_subscriber(self):
    #


    # def initialize_publisher(self):
    #     ''' for Publishing the estimated transformation '''
    #     pass

    def preprocess(self, data: dict, parameters: dict):
        ''' Preprocessing of the data gotten (if needed) '''
        return data

    def callback(self, rgb_msg, camera_msg):
        # get the rgb-data
        try:
            cv_rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, 'rgb8')
        except CvBridgeError as e:
            print(e)

        # get the Camera-Info-Message
        try:
            camera_message = camera_msg
        except Exception as e:
            print(e)

        with lock:
            self.im = cv_rgb_image.copy()
            self.camera_info = camera_message

    # TODO: used in __init__
    def load_detector(self, run_id):
        ''' TODO: replace as much as possible with the preloaded parameters '''

        run_dir = EXP_DIR / run_id
        cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
        cfg = check_update_config_detector(cfg)
        label_to_category_id = cfg.label_to_category_id
        model = create_model_detector(cfg, len(label_to_category_id))
        ckpt = torch.load(run_dir / 'checkpoint.pth.tar')
        ckpt = ckpt['state_dict']
        model.load_state_dict(ckpt)
        model = model.cuda().eval()
        model.cfg = cfg
        model.config = cfg
        model = Detector(model)
        return model

    # TODO: used in __init__
    def load_pose_models(self, coarse_run_id, refiner_run_id=None, n_workers=8):
        ''' TODO: replace as much as possible with the preloaded parameters '''

        run_dir = EXP_DIR / coarse_run_id
        cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
        cfg = check_update_config_pose(cfg)
        # object_ds = BOPObjectDataset(BOP_DS_DIR / 'tless/models_cad')
        object_ds = make_object_dataset(cfg.object_ds_name)
        mesh_db = MeshDataBase.from_object_ds(object_ds)
        renderer = BulletBatchRenderer(object_set=cfg.urdf_ds_name,
                                       n_workers=n_workers)
        mesh_db_batched = mesh_db.batched().cuda()

        def load_model(run_id):
            if run_id is None:
                return
            run_dir = EXP_DIR / run_id
            cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
            cfg = check_update_config_pose(cfg)
            if cfg.train_refiner:
                model = create_model_refiner(cfg, renderer=renderer, mesh_db=mesh_db_batched)
            else:
                model = create_model_coarse(cfg, renderer=renderer, mesh_db=mesh_db_batched)
            ckpt = torch.load(run_dir / 'checkpoint.pth.tar')
            ckpt = ckpt['state_dict']
            model.load_state_dict(ckpt)
            model = model.cuda().eval()
            model.cfg = cfg
            model.config = cfg
            return model

        coarse_model = load_model(coarse_run_id)
        refiner_model = load_model(refiner_run_id)
        model = CoarseRefinePosePredictor(coarse_model=coarse_model, refiner_model=refiner_model)
        return model, mesh_db

    # TODO: used in __init__
    def getModel(self, model_type):
        # load models
        if model_type == 'tless':
            detector_run_id = self.parameters['run_ids']['detector'] # 'detector-bop-tless-pbr--873074'
            coarse_run_id = self.parameters['run_ids']['coarse']     # 'coarse-bop-tless-pbr--506801'
            refiner_run_id = self.parameters['run_ids']['refiner']   # 'refiner-bop-tless-pbr--233420'
            detector = self.load_detector(detector_run_id)
            pose_predictor, mesh_db = self.load_pose_models(coarse_run_id=coarse_run_id,
                                                            refiner_run_id=refiner_run_id,
                                                            n_workers=4)
            return detector, pose_predictor

        if model_type == 'ycb':
            detector_run_id = self.parameters['run_ids']['detector'] # 'detector-bop-ycbv-pbr--970850'
            coarse_run_id = self.parameters['run_ids']['coarse']     # 'coarse-bop-ycbv-pbr--724183'
            refiner_run_id = self.parameters['run_ids']['refiner']   # 'refiner-bop-ycbv-pbr--604090'
            detector = self.load_detector(detector_run_id)
            pose_predictor, mesh_db = self.load_pose_models(coarse_run_id=coarse_run_id,
                                                            refiner_run_id=refiner_run_id,
                                                            n_workers=1)
            return detector, pose_predictor

    # TODO: used in evaluate()
    def inference(self, detector, pose_predictor, image, camera_k):
        # [1,540,720,3]->[1,3,540,720]
        images = torch.from_numpy(image).cuda().float().unsqueeze_(0)
        images = images.permute(0, 3, 1, 2) / 255
        # [1,3,3]
        cameras_k = torch.from_numpy(camera_k).cuda().float().unsqueeze_(0)
        # 2D detector
        # print("start detect object.")
        box_detections = detector.get_detections(images=images,
                                                 one_instance_per_class=False,
                                                 detection_th=0.8,
                                                 output_masks=False,
                                                 mask_th=0.9)
        print('#Predictions: ', len(box_detections))
        # pose estimation
        if len(box_detections) == 0:
            return None, None
        # print("start estimate pose.")
        final_preds, all_preds = pose_predictor.get_predictions(images,
                                                                cameras_k,
                                                                detections=box_detections,
                                                                n_coarse_iterations=4,
                                                                n_refiner_iterations=4)
        # result: this_batch_detections, final_preds
        return final_preds.cpu(), box_detections

    def evaluate(self):
        try:
            with lock:
                if self.im is None or self.camera_info is None:  # No data received
                    return
                im_color = self.im.copy()
                camera_info = self.camera_info

            H, W, _ = im_color.shape
            input_dim = (W, H)
            camera_K = np.array(camera_info.K).reshape(3, 3)
            print('im_color = ', type(im_color), im_color.shape)
            print('camera_K = ', type(camera_K), camera_K.shape)

            start = time.time()
            pred, detections = self.inference(detector=self.detector,
                                              pose_predictor=self.pose_predictor,
                                              image=im_color, camera_k=camera_K)
            end = time.time()
            print('Inference Time: {:.3f} s'.format(end - start))
            # print('pred = ', pred)
            # print('detections = ', detections)

            # end current iteration, if there was nothing detected (pred and/or detections == None)
            if pred is None or detections is None:
                return

            cam = dict(
                resolution=input_dim,
                K=camera_K,
                TWC=np.eye(4)
            )
            pred_rendered = render_prediction_wrt_camera(self.renderer, pred, cam)

            # Print the predictions
            print('n = ', self.n)
            print("num of pred:", len(pred))
            for i in range(len(pred)):
                print("object ", i, ":", pred.infos.iloc[i].label, "------\n  pose:",
                      pred.poses[i].numpy(), "\n  detection score:", pred.infos.iloc[i].score)

            # print('Pred\n', pred, type(pred))
            # print('Detections\n', detections.infos['label'], type(detections.infos['label']))
            msg_header = Header()
            msg_header.stamp = rospy.Time.now()
            msg_header.frame_id = 'measured/camera_link'
            poses = generate_pose_array(pred.poses, msg_header)
            poses.header = msg_header
            self.pose_array_pub.publish(poses)

            pred_classes = detections.infos['label'].tolist()
            pred_classes_str = ','.join(pred_classes)
            self.object_class_pub.publish(pred_classes_str)
            # print('pred_classes_str = ', pred_classes_str)

            figures = dict()
            plotter = Plotter()

            figures['input_im'] = plotter.plot_image(im_color)
            img_det = plotter.plot_image(im_color)
            figures['detections'] = plotter.plot_maskrcnn_bboxes(img_det, detections)
            figures['pred_rendered'] = plotter.plot_image(pred_rendered)
            figures['pred_overlay'] = plotter.plot_overlay(im_color, pred_rendered)
            fig_array = [figures['input_im'], figures['detections'], figures['pred_rendered'], figures['pred_overlay']]
            # fig_array = [figures['input_im'], figures['detections']]

            # continue

            res = gridplot(fig_array, ncols=2)
            print(type(res))
            out_png = os.path.join(os.environ['HOME'], 'Desktop', 'out', str(self.n) + '_result.png')
            try:
                export_png(res, filename=out_png)
                pass
            except Exception as e:
                print('Error: ', e)
            # show_bbox_prediction(im_color, detections.tensors['bboxes'])
            # if self.n > 10:
            #     raise Exception('ITS TIME TO STOP!')

            self.n += 1

        except KeyboardInterrupt:
            print('Ending it all!')
            raise Exception('ITS TIME TO STOP!')


