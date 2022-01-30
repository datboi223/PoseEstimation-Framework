#!/usr/bin/env python

# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md
## Partially based on the code of: start_posecnn_ros.py  (or not?)


class PoseEstimator:
    def __init__(self, parameters: dict):
        ''' read in all neccessary informations from given parameters '''
        self.parameters = parameters
        

    def preprocess(self, data : dict, parameters : dict) -> dict:
        ''' Preprocessing of the data gotten (if needed) '''
        pass

    def execute(self, data : dict) -> dict:
        pass 


def PoseEstimatorFactory(method : str, parameters : dict) -> PoseEstimator:
    ''' Function to generate the wanted type of pose estimator '''
    return PoseEstimator({})


def main():
    pass

if __name__ == '__main__':
    main()