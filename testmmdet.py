import glob
import os
import cv2
from PIL import  Image
import numpy as np

from mmdet.apis import init_detector, inference_detector, show_result_pyplot, save_result_pyplot

config_file = 'configs/atss/atss_r50_fpn_1x_coco.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = './training_dir/kitti_img/atss_sp_adamw/epoch_40_853.pth'
device = 'cuda:0'
# init a detector
model = init_detector(config_file, None, device=device)
# inference the demo image

org_imgs = glob.glob('./paper_visualize/org_img/*.png')
save_file = './paper_visualize/edge_img/'
imgs = [i.split('/')[-1] for i in org_imgs]
results, processed_datas = inference_detector(model,imgs)
show_result_pyplot(model,org_imgs[0],results[0],score_thr=0.5)

# color_map = {'Pedestrian':(10,215,255),'Car':(0,255,0),'Cyclist':(255,255,0)}
# from ipdb import set_trace;set_trace()
# color_map = [(0,255,0),(255,255,0),(10,215,255)]
# pre_threshold = 0.4

# for i,image in enumerate(org_imgs):
#     img=Image.open(image)

#     img = np.array(img)
#     for c_i, class_result in enumerate(results[i]):
#         color = color_map[c_i]
#         if len(class_result) == 0:
#             continue
#         for anno in class_result:
#             if anno[-1]< pre_threshold:
#                 continue
#             bbox = np.round(np.array(anno[:-1])).astype(np.int)
#             img = cv2.rectangle(img, bbox[:2], bbox[2:], color, thickness=2)

#     imgname = save_file+imgs[i]
#     img = Image.fromarray(img)
    # img.save(imgname)


print('end')