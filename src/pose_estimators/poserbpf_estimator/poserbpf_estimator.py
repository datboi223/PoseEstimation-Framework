#!/usr/bin/env python

# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

# Original from PoseRBPF: ros/start_posecnn_ros.py
# Adapted by: datboi223

import os
import sys
import json
# include the home-path to access the base-class
sys.path.insert(0, os.environ['EST_HOME'])
# from run_pose_estimation import PoseEstimator
import run_pose_estimation as pe

# TODO:
# import the importing the PoseRBPF-"backend"
sys.path.insert(0, os.environ['POSERBPF_HOME'])
print('POSERBPF_HOME = ', sys.path[0])


import rospy
import tf
import message_filters
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from pprint import pprint
import threading
import sys
import posecnn_cuda
import _init_paths
import networks

# from Queue import Queue
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from sensor_msgs.msg import Image as ROS_Image
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import JointState
from transforms3d.quaternions import mat2quat, quat2mat, qmult, qinverse
from transforms3d.euler import quat2euler, mat2euler, euler2quat

from utils.render_utils import render_image_detection
from scipy.optimize import minimize
from geometry_msgs.msg import PoseStamped, PoseArray
from visualization_msgs.msg import MarkerArray, Marker
from rospy.numpy_msg import numpy_msg
import matplotlib.pyplot as plt
# import config.config_posecnn as config_posecnn
# print(config_posecnn)
# from config.config_posecnn import cfg
import config.config_posecnn
# from config.config import cfg_from_file as cfg_from_file_

print(config.config_posecnn.cfg)
# from config_posecnn import cfg, cfg_from_file # cfg_from_file # important for PoseCNN
# import config.config_posecnn.cfg_from_file as cfg_from_file_posecnn
from datasets.factory import get_dataset
from utils.nms import nms
from utils.cython_bbox import bbox_overlaps
from utils.blob import pad_im


from pose_rbpf.pose_rbpf import *
from pose_rbpf.sdf_multiple_optimizer import sdf_multiple_optimizer
import scipy.io
from scipy.spatial import distance_matrix as scipy_distance_matrix
import random

lock = threading.Lock()

from random import shuffle
import tf.transformations as tra
import time

def load_config(file):
    if not os.path.exists(file): # to be removed in final version (for testing)
        print('Config does not exist: ', file)
        return dict()
    else: # load data
        with open(file, 'r') as json_file:
            data = json.load(json_file)
            assert(type(data) == dict)
        return data


def ros_qt_to_rt(rot, trans):
    qt = np.zeros((4,), dtype=np.float32)
    qt[0] = rot[3]
    qt[1] = rot[0]
    qt[2] = rot[1]
    qt[3] = rot[2]
    obj_T = np.eye(4)
    obj_T[:3, :3] = quat2mat(qt)
    obj_T[:3, 3] = trans

    return obj_T

def get_relative_pose_from_tf(listener, source_frame, target_frame):
    first_time = True
    while True:
        try:
            stamp = rospy.Time.now()
            init_trans, init_rot = listener.lookupTransform(target_frame, source_frame, rospy.Time(0))
            break
        except Exception as e:
            if first_time:
                print(str(e))
                first_time = False
            continue
    return ros_qt_to_rt(init_rot, init_trans), stamp

def allocentric2egocentric(qt, T):
    ''' Fuction from se3.py of `PoseCNN-Python` '''
    dx = np.arctan2(T[0], -T[2])
    dy = np.arctan2(T[1], -T[2])
    quat = euler2quat(-dy, -dx, 0, axes='sxyz')
    quat = qmult(quat, qt)
    return quat

def compute_poses(out_pose, out_quaternion, rois):
    '''Computing the WHOLE POSE from PoseCNN'''
    num = rois.shape[0]
    poses = out_pose.copy()
    for j in range(num):
        cls = int(rois[j, 1])
        if cls >= 0:
            qt = out_quaternion[j, 4 * cls:4 * cls + 4]
            qt = qt / np.linalg.norm(qt)
            # allocentric to egocentric
            poses[j, 4] *= poses[j, 6]
            poses[j, 5] *= poses[j, 6]
            T = poses[j, 4:]
            poses[j, :4] = allocentric2egocentric(qt, T)
    print('poses = ', poses.shape, '\n', poses)
    return poses


'''PoseRBPF similar to ImageListener from PoseRBPF'''

class Poserbpf(pe.PoseEstimator):
    def __init__(self, parameters):
        parameter_path = os.path.join(os.path.dirname(__file__), parameters)
        parameters = load_config(parameter_path)
        super().__init__(parameters)
        # parameters similar to args (contains also cfg)

        ### PoseCNN ###
        posecnn_parameters = parameters['posecnn']
        poserbpf_parameters = parameters['poserbpf']
        # self.file_path = os.path.abspath(__file__)
        self.file_path = os.path.abspath(os.path.dirname(__file__))
        print('self.file_path = ', self.file_path)

        # self.posecnn = None
        self.posecnn_suffix = None
        self.posecnn_prefix = None
        self.posecnn_cfg = None
        self.posecnn_posecnn_network = None
        self.posecnn_listener = None

        self.initialize_poscnn(posecnn_parameters)

        self.cv_bridge = CvBridge()

        self.im = None
        self.depth = None
        self.rgb_frame_id = None
        self.rgb_frame_stamp = None

        suffix = '_%02d' % (self.posecnn_cfg.instance_id)
        prefix = '%02d_' % (self.posecnn_cfg.instance_id)
        self.posecnn_suffix = suffix
        self.posecnn_prefix = prefix


    def initialize_poscnn(self, posecnn_parameters):
        # load config
        for k, v  in posecnn_parameters.items():
            print('{} -> {}'.format(k, v))

        print('Check Paths: ')

        if posecnn_parameters['cfg_file'] is not None:
            cfg_path = os.path.join(self.file_path, posecnn_parameters['cfg_file'])
            # cfg_from_file(cfg_path)
            config.config_posecnn.cfg_from_file(cfg_path)
        print('Using config:')
        pprint(config.config_posecnn.cfg)
        self.posecnn_cfg = config.config_posecnn.cfg # make config part of the class itself

        if not posecnn_parameters['randomize']:
            # fix the random seeds (numpy and caffe) for reproducibility
            np.random.seed(self.posecnn_cfg.RNG_SEED)

        # device
        self.posecnn_cfg.gpu_id = 0
        self.posecnn_cfg.device = torch.device('cuda:{:d}'.format(self.posecnn_cfg.gpu_id))
        self.posecnn_cfg.instance_id = posecnn_parameters['instance_id']

        # dataset
        self.posecnn_cfg.MODE = 'TEST'
        self.posecnn_dataset = get_dataset(posecnn_parameters['dataset_name'])

        # prepare network
        if posecnn_parameters['pretrained']:
            pretrained_path = os.path.join(os.path.dirname(__file__), posecnn_parameters['pretrained'])
            network_data = torch.load(pretrained_path)
            print("=> using pre-trained network '{}'".format(posecnn_parameters['pretrained']))
        else:
            network_data = None
            print("no pretrained network specified")
            sys.exit()

        self.posecnn_network = networks.__dict__[posecnn_parameters['network_name']](self.posecnn_dataset.num_classes, self.posecnn_cfg.TRAIN.NUM_UNITS, network_data).cuda(device=self.posecnn_cfg.device)
        self.posecnn_network = torch.nn.DataParallel(self.posecnn_network, device_ids=[0]).cuda(device=self.posecnn_cfg.device)
        cudnn.benchmark = True
        self.posecnn_network.eval()

        self.posecnn_net = self.posecnn_network

        # initialize
        self.posecnn_listener = tf.TransformListener()
        self.posecnn_br = tf.TransformBroadcaster()
        self.posecnn_label_pub = rospy.Publisher('posecnn_label', ROS_Image, queue_size=10)
        self.posecnn_rgb_pub = rospy.Publisher('posecnn_rgb', ROS_Image, queue_size=10)
        self.posecnn_depth_pub = rospy.Publisher('posecnn_depth', ROS_Image, queue_size=10)
        self.posecnn_pub = rospy.Publisher('posecnn_detection', ROS_Image, queue_size=10)
        self.test_pub = rospy.Publisher('/test_string', String, queue_size=10)

        # create pose publisher for each known object class
        self.pubs = []
        for i in range(1, self.posecnn_dataset.num_classes):
            if self.posecnn_dataset.classes[i][3] == '_':
                cls = self.posecnn_dataset.classes[i][4:]
            else:
                cls = self.posecnn_dataset.classes[i]
            self.pubs.append(rospy.Publisher('/objects/prior_pose/' + cls, PoseStamped, queue_size=10))

        
        print('***PoseCNN ready, waiting for camera images***')
        if self.posecnn_cfg.TEST.ROS_CAMERA == 'D415':
            # use RealSense D415
            # self.base_frame = 'measured/base_link'
            self.base_frame = 'camera_link'
            rgb_sub = message_filters.Subscriber('/camera/color/image_raw', ROS_Image, queue_size=10)
            depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', ROS_Image, queue_size=10)
            msg = rospy.wait_for_message('/camera/color/camera_info', CameraInfo)
            # self.camera_frame = 'measured/camera_color_optical_frame'
            self.camera_frame = 'camera_color_optical_frame'
            self.target_frame = self.base_frame
            self.viz_pub = rospy.Publisher('/obj/mask_estimates/realsense', MarkerArray, queue_size=1)
        elif self.posecnn_cfg.TEST.ROS_CAMERA == 'Azure':
            # use RealSense Azure
            self.base_frame = 'measured/base_link'
            rgb_sub = message_filters.Subscriber('/k4a/rgb/image_raw', ROS_Image, queue_size=10)
            depth_sub = message_filters.Subscriber('/k4a/depth_to_rgb/image_raw', ROS_Image, queue_size=10)
            msg = rospy.wait_for_message('/k4a/rgb/camera_info', CameraInfo)
            self.camera_frame = 'rgb_camera_link'
            self.target_frame = self.base_frame
            self.viz_pub = rospy.Publisher('/obj/mask_estimates/azure', MarkerArray, queue_size=1)
        else:
            # use kinect
            self.base_frame = '%s_rgb_optical_frame' % (self.posecnn_cfg.TEST.ROS_CAMERA)
            rgb_sub = message_filters.Subscriber('/%s/rgb/image_color' % (self.posecnn_cfg.TEST.ROS_CAMERA), ROS_Image, queue_size=10)
            depth_sub = message_filters.Subscriber('/%s/depth_registered/image' % (self.posecnn_cfg.TEST.ROS_CAMERA), ROS_Image, queue_size=10)
            msg = rospy.wait_for_message('/%s/rgb/camera_info' % (self.posecnn_cfg.TEST.ROS_CAMERA), CameraInfo)
            self.camera_frame = '%s_rgb_optical_frame' % (self.posecnn_cfg.TEST.ROS_CAMERA)
            self.target_frame = self.base_frame
            self.viz_pub = rospy.Publisher('/obj/mask_estimates/%s' % (self.posecnn_cfg.TEST.ROS_CAMERA), MarkerArray, queue_size=1)

        # camera to base transformation
        self.Tbc_now = np.eye(4, dtype=np.float32)

        # update camera intrinsics (one times)
        K = np.array(msg.K).reshape(3, 3)
        self.posecnn_dataset._intrinsic_matrix = K
        print(self.posecnn_dataset._intrinsic_matrix)

        # wait for the rgbd-data
        queue_size = 1
        slop_seconds = 0.1
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size, slop_seconds)
        ts.registerCallback(self.callback)

        ## use fake label blob
        num_classes = self.posecnn_dataset.num_classes
        height = self.posecnn_cfg.TRAIN.SYN_HEIGHT
        width = self.posecnn_cfg.TRAIN.SYN_WIDTH
        label_blob = np.zeros((1, num_classes, height, width), dtype=np.float32)
        pose_blob = np.zeros((1, num_classes, 9), dtype=np.float32)
        gt_boxes = np.zeros((1, num_classes, 5), dtype=np.float32)

        ## construct the meta data

        # metadata = [K, K^-1]
        Kinv = np.linalg.pinv(K)
        meta_data_blob = np.zeros((1, 18), dtype=np.float32)
        meta_data_blob[0, 0:9] = K.flatten()
        meta_data_blob[0, 9:18] = Kinv.flatten()

        self.label_blob = torch.from_numpy(label_blob).cuda()
        self.meta_data_blob = torch.from_numpy(meta_data_blob).cuda()
        self.extents_blob = torch.from_numpy(self.posecnn_dataset._extents).cuda()
        self.gt_boxes_blob = torch.from_numpy(gt_boxes).cuda()
        self.poses_blob = torch.from_numpy(pose_blob).cuda()
        self.points_blob = torch.from_numpy(self.posecnn_dataset._point_blob).cuda()
        self.symmetry_blob = torch.from_numpy(self.posecnn_dataset._symmetry).cuda()



    def initialize_subscriber(self):
        pass

    def initialize_publisher(self):
        pass

    def preprocess(self, data: dict, parameters: dict):
        ''' Preprocessing of the data gotten (if needed) '''
        return data

    def preprocess(self, data: dict, parameters: dict):
        ''' Preprocessing of the data gotten (if needed) '''
        return data

    def callback(self, rgb, depth):
        # self.Tbc_now, self.Tbc_stamp = get_relative_pose_from_tf(self.posecnn.listener,
        #                                                          self.camera_frame, self.base_frame)
        if depth.encoding == '32FC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth)
        elif depth.encoding == '16UC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth).copy().astype(np.float32)
            depth_cv /= 1000.0
        else:
            rospy.logerr_throttle(
                1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(
                    depth.encoding))
            return
        im = self.cv_bridge.imgmsg_to_cv2(rgb, 'bgr8')

        # rescale image if necessary
        if self.posecnn_cfg.TEST.SCALES_BASE[0] != 1:
            im_scale = self.posecnn_cfg.TEST.SCALES_BASE[0]
            im = pad_im(cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR), 16)
            depth_cv = pad_im(cv2.resize(depth_cv, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_NEAREST), 16)

        with lock:
            self.im = im.copy()
            self.depth = depth_cv.copy()
            self.rgb_frame_id = rgb.header.frame_id
            self.rgb_frame_stamp = rgb.header.stamp

    def evaluate(self):
        # posecnn-part
        print('PoseRBPF -> Evaluate: ' + str(self.n))

        with lock:
            if self.im is None:
                return

            im_color = self.im.copy()
            depth_cv = self.depth.copy()
            rgb_frame_id = self.rgb_frame_id
            rgb_frame_stamp = self.rgb_frame_stamp
            # input_Tbc = self.Tbc_now.copy()
            # input_Tbc_stamp = self.Tbc_stamp

        print('===========================================')
        
        # compute image blob
        im = im_color.astype(np.float32, copy=True)
        im -= self.posecnn_cfg.PIXEL_MEANS
        height = im.shape[0]
        width = im.shape[1]
        im = np.transpose(im / 255.0, (2, 0, 1))
        im = im[np.newaxis, :, :, :]
        inputs = torch.from_numpy(im).cuda()

        # tranform to gpu
        # Tbc = torch.from_numpy(input_Tbc).cuda().float()

        # backproject depth
        depth = torch.from_numpy(depth_cv).cuda()
        fx = self.posecnn_dataset._intrinsic_matrix[0, 0]
        fy = self.posecnn_dataset._intrinsic_matrix[1, 1]
        px = self.posecnn_dataset._intrinsic_matrix[0, 2]
        py = self.posecnn_dataset._intrinsic_matrix[1, 2]
        im_pcloud = posecnn_cuda.backproject_forward(fx, fy, px, py, depth)[0]

        # compare the depth
        depth_meas_roi = im_pcloud[:, :, 2]
        mask_depth_meas = depth_meas_roi > 0
        mask_depth_valid = torch.isfinite(depth_meas_roi)

        # forward
        # self.posecnn_cfg.TRAIN.POSE_REG = False

        print('self.posecnn_cfg.TRAIN.POSE_REG = ', self.posecnn_cfg.TRAIN.POSE_REG)

        if self.posecnn_cfg.TRAIN.POSE_REG:
            print('Pose + Quat.')
            out_label, out_vertex, rois, out_pose, out_quaternion = self.posecnn_net(inputs, self.label_blob,
                                                                                     self.meta_data_blob,
                                                                                     self.extents_blob,
                                                                                     self.gt_boxes_blob,
                                                                                     self.poses_blob,
                                                                                     self.points_blob,
                                                                                     self.symmetry_blob)
            # print('out_pose = ', out_pose.shape, '\n', out_pose)
            # print('out_quat = ', out_quaternion.shape, '\n', out_quaternion)
        else:
            print('Pose Only')
            out_label, out_vertex, rois, out_pose = self.posecnn_net(inputs, self.label_blob, self.meta_data_blob,
                                                                     self.extents_blob, self.gt_boxes_blob,
                                                                     self.poses_blob, self.points_blob,
                                                                     self.symmetry_blob)

        label_tensor = out_label[0]
        labels = label_tensor.detach().cpu().numpy()
        # print('labels = ', labels.shape)

        # filter out detections
        rois = rois.detach().cpu().numpy()
        index = np.where(rois[:, -1] > self.posecnn_cfg.TEST.DET_THRESHOLD)[0]
        rois = rois[index, :]

        # non-maximum suppression within class
        index = nms(rois, 0.2)
        rois = rois[index, :]


        # out_poses
        # preprocessing needed
        print('out_pose')
        print('out_pose = ', out_pose.shape, '\n', out_pose)
        if self.posecnn_cfg.TRAIN.POSE_REG:
            out_pose = out_pose.cpu().detach().numpy()
            out_quaternion = out_quaternion.cpu().detach().numpy()
            poses = compute_poses(out_pose, out_quaternion, rois)
            print('poses = ', poses.shape)








        # render output image
        im_label = render_image_detection(self.posecnn_dataset, im_color, rois, labels)
        rgb_msg = self.cv_bridge.cv2_to_imgmsg(im_label, 'rgb8')
        rgb_msg.header.stamp = rgb_frame_stamp
        rgb_msg.header.frame_id = rgb_frame_id
        self.posecnn_pub.publish(rgb_msg)

        # publish segmentation mask
        label_msg = self.cv_bridge.cv2_to_imgmsg(labels.astype(np.uint8))
        label_msg.header.stamp = rgb_frame_stamp
        label_msg.header.frame_id = rgb_frame_id
        label_msg.encoding = 'mono8'
        self.posecnn_label_pub.publish(label_msg)


        test_string_out = 'This code is cool at: {}'.format(str(rospy.get_time()))
        self.test_pub.publish(test_string_out)

        ## End of part without TF-Messages

        # visualization
        if self.posecnn_cfg.TEST.VISUALIZE:
            fig = plt.figure()
            ax = fig.add_subplot(2, 3, 1)
            plt.imshow(im_color)
            ax.set_title('input image')

            ax = fig.add_subplot(2, 3, 2)
            plt.imshow(im_label)

            ax = fig.add_subplot(2, 3, 3)
            plt.imshow(labels)

            # show predicted vertex targets
            vertex_pred = out_vertex.detach().cpu().numpy()
            vertex_target = vertex_pred[0, :, :, :]
            center = np.zeros((3, height, width), dtype=np.float32)

            for j in range(1, self.posecnn_dataset._num_classes):
                index = np.where(labels == j)
                if len(index[0]) > 0:
                    center[0, index[0], index[1]] = vertex_target[3*j, index[0], index[1]]
                    center[1, index[0], index[1]] = vertex_target[3*j+1, index[0], index[1]]
                    center[2, index[0], index[1]] = np.exp(vertex_target[3*j+2, index[0], index[1]])

            ax = fig.add_subplot(2, 3, 4)
            plt.imshow(center[0,:,:])
            ax.set_title('predicted center x')
            ax = fig.add_subplot(2, 3, 5)
            plt.imshow(center[1,:,:])
            ax.set_title('predicted center y')
            ax = fig.add_subplot(2, 3, 6)
            plt.imshow(center[2,:,:])
            ax.set_title('predicted z')
            plt.show()

        if not rois.shape[0]:
            return

        self.n += 1

        # tf-stuff skipped for now
        return


        indexes = np.zeros((self.posecnn_dataset.num_classes, ), dtype=np.int32)
        index = np.argsort(rois[:, 2])
        rois = rois[index, :]
        now = rospy.Time.now()
        markers = []


        for i in range(rois.shape[0]):
            roi = rois[i]
            cls = int(roi[1])
            cls_name = self.posecnn_dataset._classes_test[cls]
            if cls > 0 and roi[-1] > self.posecnn_cfg.TEST.DET_THRESHOLD:

                # compute mask translation
                w = roi[4] - roi[2]
                h = roi[5] - roi[3]
                x1 = max(int(roi[2]), 0)
                y1 = max(int(roi[3]), 0)
                x2 = min(int(roi[4]), width - 1)
                y2 = min(int(roi[5]), height - 1)

                labels = torch.zeros_like(label_tensor)
                labels[y1:y2, x1:x2] = label_tensor[y1:y2, x1:x2]
                mask_label = labels == cls
                mask = mask_label * mask_depth_meas * mask_depth_valid
                pix_index = torch.nonzero(mask)
                n = pix_index.shape[0]
                print('[%s] points : %d' % (cls_name, n))
                if n == 0:
                    '''
                    fig = plt.figure()
                    ax = fig.add_subplot(1, 2, 1)
                    plt.imshow(depth.cpu().numpy())
                    ax.set_title('depth')

                    ax = fig.add_subplot(1, 2, 2)
                    plt.imshow(im_label.cpu().numpy())
                    plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='g', linewidth=3, clip_on=False))
                    ax.set_title('label')
                    plt.show()
                    '''
                    continue

                points = im_pcloud[pix_index[:, 0], pix_index[:, 1], :]
            
                # filter points
                m = torch.mean(points, dim=0, keepdim=True)
                mpoints = m.repeat(n, 1)
                distance = torch.norm(points - mpoints, dim=1)
                extent = np.mean(self.posecnn_dataset._extents_test[cls, :])
                points = points[distance < 1.5 * extent, :]
                if points.shape[0] == 0:
                    continue
                
                # transform points to base ()
                ones = torch.ones((points.shape[0], 1), dtype=torch.float32, device=0)
                points = torch.cat((points, ones), dim=1)
                points = torch.mm(Tbc, points.t())
                location = torch.mean(points[:3, :], dim=1).cpu().numpy()
                if location[2] > 2.5:
                    continue
                print('[%s] detection score: %f' % (cls_name, roi[-1]))
                print('[%s] location mean: %f, %f, %f' % (cls_name, location[0], location[1], location[2]))

                # extend the location away from camera a bit
                c = Tbc[:3, 3].cpu().numpy()
                d = location - c
                d = d / np.linalg.norm(d)
                location = location + (extent / 2) * d

                # publish tf raw
                self.br.sendTransform(location, [0, 0, 0, 1], now, cls_name + '_raw', self.target_frame)

                # project location to base plane
                location[2] = extent / 2
                print('[%s] location mean on table: %f, %f, %f' % (cls_name, location[0], location[1], location[2]))
                print('-------------------------------------------')

                # publish tf
                self.br.sendTransform(location, [0, 0, 0, 1], now, cls_name, self.target_frame)

                # publish tf detection
                indexes[cls] += 1
                name = cls_name + '_%02d' % (indexes[cls])
                tf_name = os.path.join("posecnn", name)

                # send another transformation as bounding box (mis-used)
                n = np.linalg.norm(roi[2:6])
                x1 = roi[2] / n
                y1 = roi[3] / n
                x2 = roi[4] / n
                y2 = roi[5] / n
                self.br.sendTransform([n, now.secs, roi[6]], [x1, y1, x2, y2], now, tf_name + '_roi', self.target_frame)

                # publish marker
                marker = Marker()
                marker.header.frame_id = self.target_frame
                marker.header.stamp = now
                marker.id = cls
                marker.type = Marker.SPHERE;
                marker.action = Marker.ADD;
                marker.pose.position.x = location[0]
                marker.pose.position.y = location[1]
                marker.pose.position.z = location[2]
                marker.pose.orientation.x = 0.
                marker.pose.orientation.y = 0.
                marker.pose.orientation.z = 0.
                marker.pose.orientation.w = 1.
                marker.scale.x = .05
                marker.scale.y = .05
                marker.scale.z = .05

                if self.posecnn_cfg.TEST.ROS_CAMERA == 'Azure':
                    marker.color.a = .3
                elif self.posecnn_cfg.TEST.ROS_CAMERA == 'D415':
                    marker.color.a = 1.
                marker.color.r = self.posecnn_dataset._class_colors_test[cls][0] / 255.0
                marker.color.g = self.posecnn_dataset._class_colors_test[cls][1] / 255.0
                marker.color.b = self.posecnn_dataset._class_colors_test[cls][2] / 255.0
                markers.append(marker)

        print('markers: ', len(markers))
        self.viz_pub.publish(MarkerArray(markers))


        # self.n += 1

