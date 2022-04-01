# ROS-Node for 6D Pose-Estimation
This repository contains a ROS-Package which currently utilizes the __[cosypose](https://github.com/datboi223/cosypose)__ 6D Pose-Estimation approach.

To use the ROS-package and its node `pose_estimation`, please look in the subdirectories of the pose-estimation approach. There you find the install instructions for the mehtod of choice. Currently the usage of [cosypose](src/pose_estimators/cosypose_estimator/README.md) is possible.

Before executing the program you have to set the `EST_HOME`-Variable for the directory containing `run_pose_estimation.py` in the `src`-subdirectory
```bash
export EST_HOME=/path/to/src-dir/
```
To access cosypose you have to have followed the install instruction, as mentioned above.


The general query to run the pose-estimation node is
```bash
# name of the pose-estimation-approach to use
export ESTIMATOR=...

# json config-file which specifies relevant parameter to load the specified approach
export PARAM=...

rosrun pose_estimation run_pose_estimation.py --estimator $ESTIMATOR --param $PARAM
```
It is possible to save the predictions of __`cosypose`__ by adding `--debug` to the call of the program.

Currently available parameter for `ESTIMATOR` are:
- `cosypose`
- `PoseRBPF`

Currently available `Cosypose`-parameter for `PARAM` are:
- `ycb_config.json`
- `ycb_config_synth_real.json`
- `tless_config.json`
- `icbin_detect_config.json`

The `icbin_detect_config`-file currently only uses detection-parameter for `ic-bin`-objects, and uses 
pose-estimation-parameter trained on `ycb`-objects as placeholder

## Important:
There is still some kind of Bug present, since the program won't shut down completely using `Ctrl + C` and has to be stopped by the Task-Manager.