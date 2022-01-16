# PoseEstimation-Framework
This Program tries to make multiple Pose-Estimation frameworks easily accessible over ros.
The first approaches to be integrated are [**`Cosypose`**](https://github.com/ylabbe/cosypose) and [**`PoseRBPF`**](https://github.com/NVlabs/PoseRBPF).

Before doing anything you have to set the `EST_HOME`-Variable as the main directory.
```bash
export EST_HOME=/path/to/root-dir/
```
Then set the `COSYPOSE-HOME` and the `PoseRBPF-HOME`
```bash
export COSYPOSE_HOME=/path/to/cosypose/
export POSERBPF_HOME=/path/to/poserbpf/
```

To execute the a specific approach for Pose-Estimation you have to specify name of the approach and a valid config for that approach. 
### Execution of cosypose-Approach
```bash
 python3 run_pose_estimation.py --estimator cosypose --param <path-to-config>.json
```
Currently available configs are for [YCB](pose_estimators/cosypose_estimator/cfg/ycb_config.json) and [T-Less](pose_estimators/cosypose_estimator/cfg/tless_config.json). These configs specify the parameters to use in the different parts of `cosypose`.

### Execution of PoseRBPF-Approach
```bash
 python3 run_pose_estimation.py --estimator PoseRBPF --param <path-to-config>.json
```
___TODO:___ To be integrated.
