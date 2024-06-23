import torch
from ultralytics import YOLO
import os

data_root = 'spectral_detector'

model = YOLO(os.path.join(data_root, 'yolov8m.pt'))  # load pretrained model

version = 2601

resume = False
if os.path.exists(os.path.join(data_root, f'runs/detect/train{version}')):
    # raise Exception(f'run{version} already exists. Please choose a different version number or delete run.')
    resume = True
    print(f'run{version} already exists, starting training from last best.pt')
    model = YOLO(os.path.join(data_root, f'runs/detect/train{version}/weights/best.pt'))  # load pretrained model
    # version += 1
    results = model.train(
        # data=os.path.join(data_root, 'detection_dataset.yaml'),
        # project=os.path.join(data_root, f'runs/detect'),
        # name='train'+str(version),
        resume=resume,
        # device='mps',
        # imgsz=640, #default
    )
    results.print()  # print results to screen

else: 
    results = model.train(
        data=os.path.join(data_root, 'detection_dataset.yaml'),
        project=os.path.join(data_root, f'runs/detect'),
        name='train'+str(version),
        resume=resume,
        device='mps',
        imgsz=640, #default
        optimizer='auto', #default. try 'Adam', 'AdamW'
        save=True, #default
        save_period=30, #epochs per save, -1 to disable.
        patience=100, #default epochs without val improvement before early stopping
        epochs=100,
        close_mosaic=100, #turn off mosaic augmentation, =epochs
        batch=32, #default 16 batch size. -1 for autobatch. maximise batch size.
        
        # single_cls=True, #presence/absence detection
        
        autoaugment=False,
        erasing=0,
        
        translate=0,
        scale=0,
        fliplr=0,
        mosaic=0,
        crop_fraction=0,

        hsv_h=0, #default 0.015
        hsv_s=0.1, #default 0.7
        hsv_v=0.1, #default 0.4

        # iou=0.20, #default 0.20, this represents the IoU threshold for evaluation
        
        # cls=0, #class loss weight, default 0.5
        # resume=True # to use later with freeze
        )
    results.print()  # print results to screen

# test run
# model = YOLO('runs/detect/train5/weights/best.pt')  
# results = model.predict('datasets/artificial_dataset/images/val/815.jpg', save=True)

# print('')
# task=detect, mode=train, model=yolov8m.pt, data=detection_dataset.yaml, epochs=300, time=None, patience=100, batch=32, imgsz=640, save=True, save_period=30, cache=False, device=mps, workers=8, project=None, name=train22, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=True, rect=False, cos_lr=False, close_mosaic=300, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, multi_scale=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, embed=None, show=False, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=False, opset=None, workspace=4, nms=False, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, label_smoothing=0.0, nbs=64, hsv_h=0.015, hsv_s=0.1, hsv_v=0.1, degrees=0.0, translate=0, scale=0, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0, bgr=0.0, mosaic=0, mixup=0.0, copy_paste=0.0, auto_augment=randaugment, erasing=0.4, crop_fraction=0, cfg=None, tracker=botsort.yaml, save_dir=runs/detect/train22

# from  n    params  module                                       arguments                     
#   0                  -1  1      1392  ultralytics.nn.modules.conv.Conv             [3, 48, 3, 2]                 
#   1                  -1  1     41664  ultralytics.nn.modules.conv.Conv             [48, 96, 3, 2]                
#   2                  -1  2    111360  ultralytics.nn.modules.block.C2f             [96, 96, 2, True]             
#   3                  -1  1    166272  ultralytics.nn.modules.conv.Conv             [96, 192, 3, 2]               
#   4                  -1  4    813312  ultralytics.nn.modules.block.C2f             [192, 192, 4, True]           
#   5                  -1  1    664320  ultralytics.nn.modules.conv.Conv             [192, 384, 3, 2]              
#   6                  -1  4   3248640  ultralytics.nn.modules.block.C2f             [384, 384, 4, True]           
#   7                  -1  1   1991808  ultralytics.nn.modules.conv.Conv             [384, 576, 3, 2]              
#   8                  -1  2   3985920  ultralytics.nn.modules.block.C2f             [576, 576, 2, True]           
#   9                  -1  1    831168  ultralytics.nn.modules.block.SPPF            [576, 576, 5]                 
#  10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
#  11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
#  12                  -1  2   1993728  ultralytics.nn.modules.block.C2f             [960, 384, 2]                 
#  13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
#  14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
#  15                  -1  2    517632  ultralytics.nn.modules.block.C2f             [576, 192, 2]                 
#  16                  -1  1    332160  ultralytics.nn.modules.conv.Conv             [192, 192, 3, 2]              
#  17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
#  18                  -1  2   1846272  ultralytics.nn.modules.block.C2f             [576, 384, 2]                 
#  19                  -1  1   1327872  ultralytics.nn.modules.conv.Conv             [384, 384, 3, 2]              
#  20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
#  21                  -1  2   4207104  ultralytics.nn.modules.block.C2f             [960, 576, 2]                 
#  22        [15, 18, 21]  1   3776275  ultralytics.nn.modules.head.Detect           [1, [192, 384, 576]]          
# Model summary: 295 layers, 25856899 parameters, 25856883 gradients, 79.1 GFLOPs

# from  n    params  module                                       arguments                     
#   0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]                 
#   1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]                
#   2                  -1  1      7360  ultralytics.nn.modules.block.C2f             [32, 32, 1, True]             
#   3                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]                
#   4                  -1  2     49664  ultralytics.nn.modules.block.C2f             [64, 64, 2, True]             
#   5                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
#   6                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]           
#   7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]              
#   8                  -1  1    460288  ultralytics.nn.modules.block.C2f             [256, 256, 1, True]           
#   9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]                 
#  10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
#  11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
#  12                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]                 
#  13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
#  14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
#  15                  -1  1     37248  ultralytics.nn.modules.block.C2f             [192, 64, 1]                  
#  16                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                
#  17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
#  18                  -1  1    123648  ultralytics.nn.modules.block.C2f             [192, 128, 1]                 
#  19                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
#  20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
#  21                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]                 
#  22        [15, 18, 21]  1    751702  ultralytics.nn.modules.head.Detect           [2, [64, 128, 256]]           
# Model summary: 225 layers, 3011238 parameters, 3011222 gradients, 8.2 GFLOP