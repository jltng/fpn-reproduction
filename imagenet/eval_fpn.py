from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances

register_coco_instances("coco_val", {}, 
                       "data/coco/annotations/instances_val2017.json", 
                       "data/coco/val2017")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
cfg.MODEL.WEIGHTS = "fpn_output/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05

predictor = DefaultPredictor(cfg)
evaluator = COCOEvaluator("coco_val", cfg, False, output_dir="./eval_output/")
val_loader = build_detection_test_loader(cfg, "coco_val")
inference_on_dataset(predictor.model, val_loader, evaluator)
