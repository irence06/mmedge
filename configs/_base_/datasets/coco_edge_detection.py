# dataset settings
dataset_type = 'CocoDataset'
# data_root = '/home/liangzichen/data/kitti/'
data_root = '/DATA2/event_kitti/kitti/'
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_norm_cfg = dict(
    mean=[100.94126304534403, 72.22494395790815, 100.94126304534403], std=[107.60201402147942, 96.6555451990785, 107.60201402147942], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img','img_edge', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img','img_edge']),
            dict(type='Collect', keys=['img','img_edge']),
        ])
]
data = dict(
    samples_per_gpu=12, #16
    workers_per_gpu=12, #16
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train_kitti.json',
        img_prefix=data_root + 'train_dvs/',
        img_edge_prefix=data_root + 'train_edge/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val_kitti.json',
        img_prefix=data_root + 'val_dvs/',
        img_edge_prefix=data_root + 'val_edge/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val_kitti.json',
        img_prefix=data_root + 'val_dvs/',
        img_edge_prefix=data_root + 'val_edge/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
