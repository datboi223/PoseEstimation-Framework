import os
import sys
# include the home-path to access the base-class
sys.path.insert(0, os.environ['EST_HOME'])
# from run_pose_estimation import PoseEstimator
import run_pose_estimation as pe

# import the importing the PoseRBPF-"backend"
sys.path.insert(0, os.environ['POSERBPF_HOME'])

class PoseRBPF(pe.PoseEstimator):
    def __init__(self, parameters: dict):
        super().__init__(parameters)

    def initialize_subscriber(self) -> None:
        pass

    def initialize_publisher(self) -> None:
        pass

    def preprocess(self, data: dict, parameters: dict) -> dict:
        ''' Preprocessing of the data gotten (if needed) '''
        return data

    def preprocess(self, data: dict, parameters: dict) -> dict:
        ''' Preprocessing of the data gotten (if needed) '''
        return data

    def callback(self, data: dict) -> dict:
        pass

    def evaluate(self) -> dict:
        self.n += 1
        print('PoseRBPF -> Evaluate: ' + str(self.n))
        return dict()

