import os
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances

# Register COCO dataset
register_coco_instances("coco_train", {}, 
                       "data/coco/annotations/instances_train2017.json", 
                       "data/coco/train2017")
register_coco_instances("coco_val", {}, 
                       "data/coco/annotations/instances_val2017.json", 
                       "data/coco/val2017")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))

# Dataset configuration
cfg.DATASETS.TRAIN = ("coco_train",)
cfg.DATASETS.TEST = ("coco_val",)

# Hyperparameters matching FPN paper
cfg.SOLVER.IMS_PER_BATCH = 4  # Adjust based on your GPU memory
cfg.SOLVER.BASE_LR = 0.02
cfg.SOLVER.MAX_ITER = 90000
cfg.SOLVER.STEPS = (60000, 80000)

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
cfg.INPUT.MIN_SIZE_TRAIN = (800,)

cfg.OUTPUT_DIR = "fpn_output"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Save checkpoints every 2000 iterations to avoid losing progress
cfg.SOLVER.CHECKPOINT_PERIOD = 2000

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=True)
trainer.train()
