# 6D Pose-Estimation with PoseCNN (and PoseRBPF)
In order to use the (PoseCNN + PoseRBPF)-approach for 6D-Pose Estimation you have to install the required programs first.
The authors of [PoseRBPF](https://github.com/joweyel/PoseRBPF) stated, that the ROS-version of the Pose-Estimation-method 
would only work with Python 2.7. However, it is possible to use the two components `PoseCNN` and `PoseRBPF` of this 
Pose-Estimation-method with Python 3 and ROS-Noetic.

To use PoseCNN and Pose follow the instructions below

# Installation
Get the `PoseRBPF`-Repository from here 
```bash
git clone https://github.com/joweyel/PoseRBPF.git --recursive
```
To install all dependencies and the program itself follow the instructions __[here](https://github.com/joweyel/PoseRBPF#online-real-world-pose-estimation-using-ros-noetic-and-python-3)__.

__Important:__ When installing the YCB-Renderer inside the Conda-Environment, a non-Conda version of CUDA has to be present on the computer, in order to compile the program. Compilation was successfully tested with `CUDA 11.3` and `CUDA 11.4 Update 4`.

You now have to download the parameters for Pose-Estimation of `YCB-Objects`. You can find these parameters 
__[here](https://github.com/joweyel/PoseRBPF#download)__. Place the downloaded files in the directory specified in the 
instructions.

Now you have to set symbolic links to the `config`- and `checkpoints`-subdirectories of the `PoseRBPF`-Repository in 
order to load the downloaded model-parameters and relevant config-file.
```bash
ln -s /path/to/PoseRBPF/checkpoints/ checkpoints
ln -s /path/to/PoseRBPF/config config
```
Other than `cosypose`, the current state of `PoseRBPF` only works inside the `PyCharms`-IDE, because there is currently a problem with the anaconda environment.
