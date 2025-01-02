import itertools
import copy
import submitit
import os, sys
import argparse
import json
from train_cbm import main
from loguru import logger

def parse():
  parser = argparse.ArgumentParser(description="Settings for creating CBM")
  parser.add_argument("--dataset", type=str, default="cifar10")
  parser.add_argument(
      "--mock",
      action="store_true",
      help="Mock training for debugging purposes",
  )
  parser.add_argument(
      "--concept_set", type=str, default=None, help="path to concept set name"
  )
  parser.add_argument(
      "--filter_set", type=str, default=None, help="path to concept set name"
  )
  parser.add_argument(
      "--val_split", type=float, default=0.1, help="Validation split fraction"
  )
  parser.add_argument(
      "--backbone",
      type=str,
      default="clip_RN50",
      help="Which pretrained model to use as backbone",
  )
  parser.add_argument(
      "--feature_layer",
      type=str,
      default="layer4",
      help="Which layer to collect activations from. Should be the name of second to last layer in the model",
  )
  parser.add_argument(
      "--use_clip_penultimate",
      action="store_true",
      help="Whether to use the penultimate layer of the CLIP backbone",
  )
  parser.add_argument(
      "--skip_concept_filter",
      action="store_true",
      help="Whether to skip filtering concepts",
  )
  parser.add_argument(
      "--annotation_dir",
      type=str,
      default="annotations",
      help="where annotations are saved",
  )
  parser.add_argument(
      "--save_dir",
      type=str,
      default="saved_models",
      help="where to save trained models",
  )
  parser.add_argument(
      "--load_dir", type=str, default=None, help="where to load trained models from"
  )
  parser.add_argument(
      "--device", type=str, default="cuda", help="Which device to use"
  )
  parser.add_argument(
      "--num_workers",
      type=int,
      default=4,
      help="Number of workers used for loading data",
  )
  parser.add_argument(
      "--allones_concept",
      action="store_true",
      help="Change concept dataset to ones corresponding to class",
  )
  # arguments for CBL
  parser.add_argument(
      "--crop_to_concept_prob",
      type=float,
      default=0.0,
      help="Probability of cropping to concept granuality",
  )
  parser.add_argument(
      "--cbl_confidence_threshold",
      type=float,
      default=0.15,
      help="Confidence threshold for bouding boxes to use",
  )
  parser.add_argument(
      "--cbl_hidden_layers",
      type=int,
      default=1,
      help="how many hidden layers to use in the projection layer",
  )
  parser.add_argument(
      "--cbl_batch_size",
      type=int,
      default=512,
      help="Batch size used when fitting projection layer",
  )
  parser.add_argument(
      "--cbl_epochs",
      type=int,
      default=20,
      help="how many steps to train the projection layer for",
  )
  parser.add_argument(
      "--cbl_weight_decay",
      type=float,
      default=1e-5,
      help="weight decay for training the projection layer",
  )
  parser.add_argument(
      "--cbl_lr",
      type=float,
      default=5e-4,
      help="learning rate for training the projection layer",
  )
  parser.add_argument(
      "--cbl_loss_type",
      choices=["bce", "twoway"],
      default="bce",
      help="Loss type for training CBL",
  )
  parser.add_argument(
      "--cbl_twoway_tp",
      type=float,
      default=4.0,
      help="TPE hyperparameter for TwoWay CBL loss",
  )
  parser.add_argument(
      "--cbl_pos_weight",
      type=float,
      default=1.0,
      help="loss weight for positive examples",
  )
  parser.add_argument(
      "--cbl_auto_weight",
      action="store_true",
      help="whether to automatically weight positive examples",
  )
  parser.add_argument(
      "--cbl_finetune",
      action="store_true",
      help="Enable finetuning backbone in CBL training",
  )
  parser.add_argument(
      "--cbl_bb_lr_rate",
      type=float,
      default=1,
      help="Rescale the learning rate of backbone during finetuning",
  )
  parser.add_argument(
      "--cbl_optimizer",
      choices=["adam", "sgd"],
      default="sgd",
      help="Optimizer used in CBL training.",
  )
  parser.add_argument(
      "--cbl_scheduler",
      choices=[None, "cosine"],
      default=None,
      help="Scheduler used in CBL training.",
  )
  # arguments for SAGA
  parser.add_argument(
      "--saga_batch_size",
      type=int,
      default=512,
      help="Batch size used when fitting final layer",
  )
  parser.add_argument(
      "--saga_step_size", type=float, default=0.1, help="Step size for SAGA"
  )
  parser.add_argument(
      "--saga_lam",
      type=float,
      default=0.0007,
      help="Sparsity regularization parameter, higher->more sparse",
  )
  parser.add_argument(
      "--saga_n_iters",
      type=int,
      default=2000,
      help="How many iterations to run the final layer solver for",
  )
  parser.add_argument(
      "--seed", type=int, default=42, help="Random seed for reproducibility"
  )
  parser.add_argument(
      "--dense", action="store_true", help="train with dense or sparse method"
  )
  parser.add_argument(
      "--dense_lr",
      type=float,
      default=0.001,
      help="Learning rate for the dense final layer training",
  )
  parser.add_argument("--data_parallel", action="store_true")
  parser.add_argument(
      "--visualize_concepts", action="store_true", help="Visualize concepts"
  )

  config_parser = argparse.ArgumentParser()
  config_parser.add_argument("--config", type=str, default="configs/celeba.json")
  config_arg, remaining_args = config_parser.parse_known_args()
  if config_arg.config is not None:
      with open(config_arg.config, "r") as f:
          config_arg = json.load(f)
      parser.set_defaults(**config_arg)
  
  # run the training
  args = parser.parse_args(remaining_args)
  logger.info(args)
  
  return args

def launch_grid_search(args):
    # define setting
    #args.project="XOR"
    #args.wandb = True
    
    #args.seed = 0
    
    hyperparameters=[
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],  # --crop_to_concept_prob
            [0.15, 0.3, 0.45, 0.6, 0.75, 0.9], # --cbl_confidence_threshold
            [1, 2], # --cbl_hidden_layers
            #[256, 512, 1024], # --cbl_batch_size
                              #--cbl_epochs
            [1e-7, 1e-5, 1e-2], #--cbl_weight_decay
            [1e-5, 5e-4, 1e-3, 1e-2] # --cbl_lr
        ]

    args_list=[]
    for combination in itertools.product(*hyperparameters):
        args1= copy.copy(args)
        (args1.crop_to_concept_prob, 
        args1.cbl_confidence_threshold, 
        args1.cbl_hidden_layers, 
        args.cbl_weight_decay, 
        args.cbl_lr) = combination
        print(args1, '\n')
        args_list.append(args1)
        break
    return args_list

if __name__ == '__main__':
  #start_main()
  args = parse() # 
  #     args = prepare_args() #  parse_args() # 
  executor = submitit.AutoExecutor(folder="./logs", slurm_max_num_timeout=30)
  executor.update_parameters(
          mem_gb=20,
          gpus_per_task=1,
          tasks_per_node=1,  # one task per GPU
          cpus_per_gpu=4,
          nodes=1,
          timeout_min=20,
          # Below are cluster dependent parameters
          slurm_partition="edu-20h",
          slurm_signal_delay_s=120,
          slurm_array_parallelism=4)

  experiments=launch_grid_search(args) 
  executor.update_parameters(name="vlg")
  jobs = executor.map_array(main,experiments)