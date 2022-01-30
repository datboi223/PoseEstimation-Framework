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
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
import message_filters
from cv_bridge import CvBridge, CvBridgeError
import tf
import geometry_msgs.msg


# importing the cosypose-"backend"
sys.path.append(os.environ['COSYPOSE_HOME'])
print('COSYPOSE_HOME: ', os.environ['COSYPOSE_HOME'])



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

# Code By: Pavel Krsek
# taken from: http://people.ciirc.cvut.cz/~krsek/Teaching/NPGR001_ARO/Euler_angles.pdf
def rot2euler(R):
    eps = 1e-7
    if (abs(R[2, 2]) - 1) > eps:
        theta =  np.arccos(R[2, 2])
        phi = np.arctan2(R[1, 2], R[0, 2])
        psi = np.arctan2(R[2, 1], -R[2, 0])
        theta2 = 2 * np.pi - np.arccos(R[2, 2])
        phi2 = np.arctan2(-R[1, 2], -R[0, 2])
        psi2 = np.arctan2(-R[2, 2], R[2, 0])
        eul = np.array([[phi, theta, psi], [phi2, theta2, psi2]])
    else:
        if R[2, 2] > 0:
            theta = 0
            phi = np.arctan2(R[1, 0], R[0, 0])
            eul = np.array([[phi, theta, 0.0], [phi, theta, 0.0]])
        else:
            theta = np.pi
            phi = np.arctan2(-R[1, 0], -R[0, 0])
            eul = np.array([[phi, theta, 0.0], [phi, theta, 0.0]])
        return eul


def generate_tf_msg(t, R, child_frame, frame):
    euler_angles = rot2euler(R)[0, :]
    angle1, angle2, angle3 = euler_angles
    quat = tf.transformations.quaternion_from_euler(angle1, angle2, angle3)

    tf_msg = geometry_msgs.msg.TransformStamped()

    tf_msg.header.frame_id = '/world'
    tf_msg.header.stamp = rospy.Time.now()
    tf_msg.child_frame_id = 'pose_frame'
    tf_msg.transform.translation.x = t[0]
    tf_msg.transform.translation.y = t[0]
    tf_msg.transform.translation.z = t[0]

    tf_msg.transform.rotation.x = quat[0]
    tf_msg.transform.rotation.y = quat[1]
    tf_msg.transform.rotation.z = quat[2]
    tf_msg.transform.rotation.w = quat[3]

    tfm = tf.msg.tfMessage([tf_msg])

    return tfm

    # self.pose_publisher.sendTransform(translation=t,
    #                                   rotation=tf.transformations.quaternion_from_euler(angle1, angle2, angle3),
    #                                   time=rospy.Time.now(),
    #                                   child='pose_frame',
    #                                   parent='/world')


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
        self.pose_publisher = tf.TransformBroadcaster()

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
                                                            n_workers=4)
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

        pose = pred.poses[0].numpy()
        R = pose[:3, :3].astype(np.float32)
        t = pose[:3, -1].astype(np.float32).tolist()
        t_ = tuple(t)

        euler = rot2euler(R)[0]
        angle1 = float(euler[0])
        angle2 = float(euler[1])
        angle3 = float(euler[2])
        print('Euler: \n', euler)

        # TODO: use multiple tf-publisher for multiple classes
        # as seen in other approaches (by Nvidia)
        self.pose_publisher.sendTransform(translation=(t_[0], t_[1], t_[2]),
                                          rotation=tf.transformations.quaternion_from_euler(angle1, angle2, angle3),
                                          time=rospy.Time.now(),
                                          child='pose_frame',
                                          parent='camera_link')

        figures = dict()
        plotter = Plotter()

        figures['input_im'] = plotter.plot_image(im_color)
        img_det = plotter.plot_image(im_color)
        figures['detections'] = plotter.plot_maskrcnn_bboxes(img_det, detections)
        print('--->>>', type(figures['input_im']))
        print('npp', np.array(img_det))
        print('IN_IMG = ', np.array(figures['input_im']))
        figures['pred_rendered'] = plotter.plot_image(pred_rendered)
        figures['pred_overlay'] = plotter.plot_overlay(im_color, pred_rendered)
        fig_array = [figures['input_im'], figures['detections'], figures['pred_rendered'], figures['pred_overlay']]
        # fig_array = [figures['input_im'], figures['detections']]

        # continue

        res = gridplot(fig_array, ncols=2)
        print(type(res))
        out_png = os.path.join(os.environ['HOME'], 'Desktop', 'out', str(self.n) + '_result.png')
        export_png(res, filename=out_png)

        # show_bbox_prediction(im_color, detections.tensors['bboxes'])

        self.n += 1