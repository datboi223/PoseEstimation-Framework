#!/usr/bin/env python3

import os
import sys
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

pose_est_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'pose_estimators')
print('pose_est_path = ', pose_est_path)
sys.path.insert(0, pose_est_path)

# ROS Imports
import roslib, rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
import message_filters
from cv_bridge import CvBridge, CvBridgeError
import tf


class PoseEstimator:

    def __init__(self, parameters: dict):
        self.parameters = parameters
        self.n = 0

    def evaluate(self):
        pass

def PoseEstimatorFactory(estimator, parameter, visualize):
    '''Factory to get the specified Estimator-Class'''

    if estimator == 'cosypose':
        import cosypose_estimator.cosypose_estimator as cp
        print('Creating Cosypose!')
        return cp.Cosypose(parameter, visualize)
    elif estimator == 'PoseRBPF':
        import poserbpf_estimator.poserbpf_estimator as pr
        print('Creating PoseRBPF!')
        return pr.Poserbpf(parameter)
    else:
        print('No valid pose estimator specified. Aborting!')
        exit(-1)

if __name__ == '__main__':

    try:
        args = {}
        estimator = rospy.get_param('estimator')
        args['estimator'] = estimator
        param = rospy.get_param('param')
        args['param'] = param
        object_class = rospy.get_param('object_class')
        args['object_class'] = object_class
        print('Using ROS-Parameters')

    except: # called from console
        parser = argparse.ArgumentParser(description='Arguments to initialize and start the pose estimation')
        parser.add_argument('--estimator', '-e', type=str, required=True)
        parser.add_argument('--param', '-p', type=str, required=True)
        parser.add_argument('--object_class', '-o', type=str, default='all')
        parser.add_argument('--debug', action='store_true')
        args = vars(parser.parse_args())
        print('Using Console-Parameters')

    print('Using the arguments:')
    print(args)

    rospy.init_node('pose_est')

    try:
        pose_estimator = PoseEstimatorFactory(estimator=args['estimator'], parameter=args['param'], visualize=args['debug'])
    except Exception as e:
        print(e)
        sys.exit()

    pose_estimator.choose_object(args['object_class'])
    while not rospy.is_shutdown():
        pose_estimator.evaluate()
