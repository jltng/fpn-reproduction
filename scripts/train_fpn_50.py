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
# CHANGE: Use ResNet-50 FPN 1x config
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))

# CHANGE: Load COCO pre-trained weights
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")

# Dataset configuration
cfg.DATASETS.TRAIN = ("coco_train",)
cfg.DATASETS.TEST = ("coco_val",)

# Fine-tuning hyperparameters (same as ResNet-101)
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = 5000  # ~2-4 hours
cfg.SOLVER.STEPS = (3000, 4000)

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
cfg.INPUT.MIN_SIZE_TRAIN = (800,)

# CHANGE: Separate output directory
cfg.OUTPUT_DIR = "fpn_output_50_pretrained"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

cfg.SOLVER.CHECKPOINT_PERIOD = 1000

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
