# ROS-Node for 6D Pose-Estimation
This repository contains a ROS-Package which currently wraps and utilizes te __[cosypose](https://github.com/datboi223/cosypose)__ 6D Pose-Estimation approach.

To use the ROS-package and its node `pose_estimation`, please look in the subdirectories of the pose-estimation approach. There you find the install instructions for the mehtod of choice. Currently the usage of [cosypose](src/pose_estimators/cosypose_estimator/README.md) is possible.

Before executing the program you have to set the `EST_HOME`-Variable for the directory containing `run_pose_estimation.py` in the `src`-subdirectory
```bash
export EST_HOME=/path/to/src-dir/
```
To access cosypose you have to have followed the install instruction, as mentioned above.
To access all 


The general query to run the pose-estimation node is
```bash
# name of the pose-estimation-approach to use
export ESTIMATOR=...

# json config-file which specifies relevant parameter to load the specified approach
export PARAM=...

rosrun pose_estimation run_pose_estimation.py --estimator $ESTIMATOR --param $PARAM
```

Currently available parameter for `ESTIMATOR` are:
- `cosypose`

Currently available parameter for `PARAM` are:
- `/path/to/estimator/cfg/ycb_config.json`
- `/path/to/estimator/cfg/tless_config.json`

## Important:
There is still some kind of Bug present, since the program wont shut down completely using `Ctrl + C` and has to be stopped by the Task-Manager.