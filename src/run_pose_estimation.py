import os
import sys
import json
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

pose_est_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'pose_estimators')
sys.path.insert(0, pose_est_path)

# ROS Imports
import roslib, rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
import message_filters
from cv_bridge import CvBridge, CvBridgeError
import tf


def load_config(file):
    if not os.path.exists(file): # to be removed in final version (for testing)
        print('Config does not exist: ', file)
        return dict()
    else: # load data
        with open(file, 'r') as json_file:
            data = json.load(json_file)
            assert(type(data) == dict)
        return data

'''
class: 
    - gets called with estimation method name (maybe use factory)
        - every class uses a parameter dictionary as input to initialize
    - chooses subscriber accordingly
    - waiting for data
'''


class PoseEstimator:
    def __init__(self, parameters: dict):
        self.parameters = parameters
        self.n = 0
        # to call each specific class of preprocessing -> make it more general
        pass

    def initialize_subscriber(self):
        pass
    
    def initialize_publisher(self):
        pass

    def preprocess(self, data, parameters):
        pass

    # def callback(self, data):
    #     pass

    def evaluate(self):
        pass

def PoseEstimatorFactory(estimator, parameter):
    '''Factory to get the specified Estimator-Class'''

    for idx, p in enumerate(sys.path):
        print('Path {}: {}'.format(idx, p))

    # imports placed here to avoid circular import
    import cosypose_estimator.cosypose_estimator as cp  # the whole program
    import poserbpf_estimator.poserbpf_estimator as pr  # the class
    
    # import cosypose
    # print(cosypose.__file__)
    # exit()

    if estimator == 'cosypose':
        print('Creating Cosypose!')
        return cp.Cosypose(parameter)
    elif estimator == 'PoseRBPF':
        print('Creating PoseRBPF!')
        return pr.PoseRBPF(parameter)
    else:
        print('No valid pose estimator specified. Aborting!')
        exit(-1)

if __name__ == '__main__':
    print('__main__')

    parser = argparse.ArgumentParser(description='Arguments to initialize and start the pose estimation')
    parser.add_argument('--estimator', '-e', type=str, required=True)
    parser.add_argument('--param', '-p', type=str, required=True)

    args = parser.parse_args()
    print(args)

    # load the config/parameters for the approach you want to use
    cfg = load_config(args.param)

    rospy.init_node('pose_est')

    pose_estimator = PoseEstimatorFactory(estimator=args.estimator, parameter=cfg)

    while not rospy.is_shutdown():
        try:
            pose_estimator.evaluate()
        except Exception as e:
            print('Program stopped!')
            exit(-1)
