import detectron2
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
import os

# Register the dataset
register_coco_instances("my_dataset", {}, "annotations.json", "../small-images/")

# Verify the registration
metadata = MetadataCatalog.get("my_dataset")
dataset_dicts = DatasetCatalog.get("my_dataset")



cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))  # Load a base config file for your model
cfg.DATASETS.TRAIN = ("my_dataset",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Load pre-trained weights
cfg.MODEL.DEVICE = "cpu" #TODO Uses MAC CPU instead of CUDA or Metal
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # Pick a good LearningRate
cfg.SOLVER.MAX_ITER = 3000    # Adjust up if val mAP is still rising, adjust down if overfitting
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only one class (text)

# Create output directory if it doesn't exist
cfg.OUTPUT_DIR = './output'
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Save the config for future reference
with open(os.path.join(cfg.OUTPUT_DIR, 'config.yaml'), 'w') as f:
    f.write(cfg.dump())



trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()
