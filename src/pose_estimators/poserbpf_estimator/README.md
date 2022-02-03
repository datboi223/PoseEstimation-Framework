# 6D Pose-Estimation with PoseCNN (and PoseRBPF)
In order to use the (PoseCNN + PoseRBPF)-approach for 6D-Pose Estimation you have to install the required programs first.
The authors of [PoseRBPF](https://github.com/datboi223/PoseRBPF) stated, that the ROS-version of the Pose-Estimation-method 
would only work with Python 2.7. However, it is possible to use the two components `PoseCNN` and `PoseRBPF` of this 
Pose-Estimation-method with Python 3 and ROS-Noetic.

To use PoseCNN and Pose follow the instructions below
Clone the Git-Repositry
```bash
git clone https://github.com/NVlabs/PoseRBPF.git --recursive
```

# Installation
Get the `PoseRBPF`-Repository from here 
```bash
git clone https://github.com/NVlabs/PoseRBPF.git --recursive
```
To install all dependencies and the program itself follow the instructions __[here](https://github.com/datboi223/PoseRBPF#online-real-world-pose-estimation-using-ros-noetic-and-python-3)__.

You now have to download the parameters for Pose-Estimation of `YCB-Objects`. You can find these parameters 
__[here](https://github.com/datboi223/PoseRBPF#download)__. Place the downloaded files in the directory specified in the 
instructions.

Now you have to set symbolic links to the `config`- and `checkpoints`-subdirectories of the `PoseRBPF`-Repository in 
order to load the downloaded model-parameters and relevant config-file.
```bash
ln -s /path/to/PoseRBPF/checkpoints/ checkpoints
ln -s /path/to/PoseRBPF/config config
```
