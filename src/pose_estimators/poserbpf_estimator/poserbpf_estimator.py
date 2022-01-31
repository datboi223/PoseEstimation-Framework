import os
import sys
# include the home-path to access the base-class
sys.path.insert(0, os.environ['EST_HOME'])
# from run_pose_estimation import PoseEstimator
import run_pose_estimation as pe

# TODO:
# import the importing the PoseRBPF-"backend"
# sys.path.insert(0, os.environ['POSERBPF_HOME'])

class PoseRBPF(pe.PoseEstimator):
    def __init__(self, parameters: dict):
        super().__init__(parameters)

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

    def callback(self, data: dict):
        pass

    def evaluate(self):
        self.n += 1
        print('PoseRBPF -> Evaluate: ' + str(self.n))
        return dict()

