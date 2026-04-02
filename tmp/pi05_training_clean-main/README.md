## Installation

The training is based on the LeRobot codebase.
```
conda create -n lerobot python=3.10
conda activate lerobot
conda install ffmpeg -c conda-forge
pip install "lerobot[pi]@git+https://github.com/huggingface/lerobot.git"
pip install zmq scipy h5py matplotlib
```
You need to directly modify the installed LeRobot code to support new funcitonalities like validation loss evaluation.

Use `lerobot_replace` and `lerobot_replace_helper.py` provided to overwrite part of the LeRobot librabry.
```
python lerobot_replace_helper.py replace ../lerobot_replace ../lerobot_orig
```
This will also back up the replaced files in `lerobot_orig`. You can use the backup files to revert the changes.
```
python lerobot_replace_helper.py restore ../lerobot_orig
```

## Training
See `train.sh` for an example training command.
```
python train_pi05.py --config configs/pi05_training_config.yaml
```
You can set the dataset path, save path, WandB settings, distributed training (`distributed: true`) and other training hyperparameters in the config file. The dataset paths should already be accessible.
