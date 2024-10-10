import argparse
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, launch
from detectron2 import model_zoo

def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.config_file))
    cfg.DATASETS.TRAIN = (args.dataset_name,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.config_file)
    cfg.MODEL.DEVICE = args.device
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size * args.num_gpus  # Adjusted for multi-GPU
    cfg.SOLVER.BASE_LR = args.learning_rate
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.CHECKPOINT_PERIOD = args.checkpoint_period
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.batch_size_per_image
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes
    cfg.OUTPUT_DIR = args.output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg

def main(args):
    # Register dataset
    register_coco_instances(args.dataset_name, {}, args.annotations_json, args.image_dir)
    MetadataCatalog.get(args.dataset_name)

    # Setup configuration
    cfg = setup_cfg(args)

    # Save the config for future reference
    with open(os.path.join(cfg.OUTPUT_DIR, 'config.yaml'), 'w') as f:
        f.write(cfg.dump())

    # Start training
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Detectron2 model")
    parser.add_argument("--dataset_name", required=True, help="Name of the dataset")
    parser.add_argument("--annotations_json", required=True, help="Path to the annotations JSON file")
    parser.add_argument("--image_dir", required=True, help="Path to the image directory")
    parser.add_argument("--config_file", default="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", help="Path to the model config file")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of data loader workers")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.00025, help="Learning rate")
    parser.add_argument("--max_iter", type=int, default=3000, help="Maximum number of iterations")
    parser.add_argument("--batch_size_per_image", type=int, default=128, help="Batch size per image for ROI heads")
    parser.add_argument("--num_classes", type=int, default=1, help="Number of classes")
    parser.add_argument("--output_dir", default="./output", help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Whether to resume training")
    parser.add_argument("--checkpoint_period", type=int, default=100, help="Saves model at every checkpoint period")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
    args = parser.parse_args()

    # Launch multi-GPU training
    launch(main, args.num_gpus, num_machines=1, machine_rank=0, dist_url='auto', args=(args,))

