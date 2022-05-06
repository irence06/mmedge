# Event-based Object Detection with Lightweight Spatial Attention Mechanism

## Introduction
This repository contains SPatial Attention Mechanism of our paper (publised on ICARM2021): Event-based Object Detection with Lightweight Spatial Attention Mechanism. Our model is developed based on MMDetection.
![Fig](https://github.com/irence06/mmedge/blob/master/readme/spattend.jpg)


## Environment
Our code is developed and evaluated on the following environment:
* Python 3.7
* Pytorch 1.7
* CUDA 11.4

After build the environment of MMDetection obeying its get_started.md, install the extra dependencies by running:
```
pip install -r requirements/mmedge_install.txt 
```

## Dataset Pre-processing
Event Stream represents edge of moving object with a four dimensional tuple (t,x,y,p), which is incompatible with the input of CNN-based model. Hence, we adopt event-encoding methods (i.e. SAE and HIS) and Canny Extractor to generate event frame and edge map respectively. Annotation follows COCO format and the dataset should be orgnaized as:

    .
    ├── DATA_DIR
    │   └── annotations
    │       ├──instances_train.json
    │       ├──instances_val.json
    │   └── train_dvs
    │       ├──000000.png
    │       ├──000001.png
    │       ├──...
    │   └── train_edge
    │       ├──000000.png
    │       ├──000001.png
    │       ├──...
    │   └── val_dvs
    │       ├──000000.png
    │       ├──000001.png
    │       ├──...
    │   └── val_edge
    │       ├──000000.png
    │       ├──000001.png
    │       ├──...
    
    
   
Our_dataset can be downloaded from Baidu Disk:

```
link: https://pan.baidu.com/s/1s5fIJsE5QY9QktMTAYYtWg  
password: cce0
```


## Training
Selecting ATSS as the baseline, event-based ATSS with lightweight spatial attention mechanism can be trained using:

```
python tools/train.py configs/atss/atss_r50_fpn_1x_coco.py --work_dir training_dir/kitti_dvs/atss_sp_add_adamw
```

Checkpoint file would be stored in the {work_dir}. Please remember to update *your dataset_path* and *your work_dir* before training.


## Testing
Evaluate the checkpoint by runing:
```
python tools/test.py configs/atss/atss_r50_fpn_1x_coco.py ./training_dir/kitti_dvs/atss_sp_add_adamw/epoh_11_747.pth --work-dir ./training_dir/kitti_dvs/atss_sp_add_adamw/ --out result_test.pkl --eval bbox --show_dir ./training_dir/kitti_dvs/atss_sp_add_adamw/eval_results_show 
```
Please remember to update *your dataset_path*, *your checkpoint_path*, *your work_dir*, *your out_pickle* and *your show_dir* before evaluation.

## Citation
If you use the code in your research, please cite as:

Liang Z, Chen G, Li Z, et al. Event-based object detection with lightweight spatial attention mechanism[C]//2021 6th IEEE International Conference on Advanced Robotics and Mechatronics (ICARM). IEEE, 2021: 498-503.


```
@INPROCEEDINGS{9536146,
  author={Liang, Zichen and Chen, Guang and Li, Zhijun and Liu, Peigen and Knoll, Alois},
  booktitle={2021 6th IEEE International Conference on Advanced Robotics and Mechatronics (ICARM)}, 
  title={Event-based Object Detection with Lightweight Spatial Attention Mechanism}, 
  year={2021},
  volume={},
  number={},
  pages={498-503},
  doi={10.1109/ICARM52023.2021.9536146}}
```
