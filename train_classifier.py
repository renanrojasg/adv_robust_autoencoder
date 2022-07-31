import os
import sys
import dill
import torch as ch
from argparse import ArgumentParser
from datetime import datetime
from utils.core_utils import stdout_logger, fix_seeds
from robustness.datasets import DATASETS
from robustness.model_utils import make_and_restore_model
from robustness.train import train_model
from robustness.defaults import check_and_fill_args
from robustness.tools import constants, helpers
from robustness import defaults
from torchvision import transforms


# Parse arguments
parser = ArgumentParser()
parser = defaults.add_args_to_parser(defaults.MODEL_LOADER_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.TRAINING_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.PGD_ARGS, parser)
parser.add_argument( '--seed', type= int, default= None, help= "Fixed random seed.")
parser.add_argument( '--transform_train', type= str, default= 'default', help= "Train set preprocessing.")
parser.add_argument( '--transform_test', type= str, default= 'default', help= "Test set preprocessing.")
parser.add_argument( '--stdout_dir', type= str, help= "Standard outpur directory.")
parser.add_argument( '--stdout_logger', action= "store_true", default= False, help= "Log stdout.")
args = parser.parse_args()

# Custom arguments
args.exp_name= datetime.now().strftime( "%Y_%m_%d_%H_%M_%S")
args.out_dir= os.path.join( args.out_dir, args.exp_name)

if args.stdout_logger:
  # Set stdout
  args.stdout_str= os.path.join( args.stdout_dir, args.exp_name + '.txt')
  sys.stdout= stdout_logger( stdout_str= args.stdout_str)

# Fix random seed
if args.seed: fix_seeds( args.seed)

# Create output folder
if not os.path.exists( args.out_dir):
    os.makedirs( args.out_dir)

# Set dataset parameters 
assert args.dataset is not None, "Must provide a dataset"
ds_class = DATASETS[args.dataset]
args = check_and_fill_args(args, defaults.TRAINING_ARGS, ds_class)
args = check_and_fill_args(args, defaults.MODEL_LOADER_ARGS, ds_class)
if args.adv_train or args.adv_eval:
      args = check_and_fill_args(args, defaults.PGD_ARGS, ds_class)
data_path = os.path.expandvars(args.data)
dataset = DATASETS[args.dataset](data_path)

# Train data transformations
if args.transform_train== 'default':
    pass
elif args.transform_train== 'skip':
    dataset.transform_train= transforms.ToTensor() # No transformation
elif args.transform_train== 'resize':
    dataset.transform_train= transforms.Compose([ transforms.Resize(( 224, 224)),
                             transforms.ToTensor(),]) # Rescaling.
else:
    raise ValueError( "Wrong transform_train argument.")

# Test data transformations
if args.transform_test== 'default':
    pass
elif args.transform_test== 'skip':
    dataset.transform_test= transforms.ToTensor()
elif args.transform_test== 'resize':
    dataset.transform_test= transforms.Compose([ transforms.Resize(( 224, 224)),
                             transforms.ToTensor(),])
else:
    raise ValueError( "Wrong transform_test argument.")

# Build model
model, _ = make_and_restore_model( arch= args.arch,
                                   dataset= dataset,
                                   resume_path= args.resume)

# Create dataloaders
train_loader, val_loader = dataset.make_loaders( args.workers,
                                                 args.batch_size,
                                                 data_aug= bool( args.data_aug))

# Prefetch data
train_loader = helpers.DataPrefetcher( train_loader)
val_loader = helpers.DataPrefetcher( val_loader)

# Load checkpoint
if args.resume: checkpoint = ch.load( args.resume,
                                      pickle_module= dill)
else: checkpoint= None

# Train classifier
model= train_model( args= args,
                    model= model,
                    loaders= ( train_loader, val_loader),
                    checkpoint= checkpoint) # train model
print( "Output folder: ", args.out_dir)
