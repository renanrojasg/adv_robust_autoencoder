import os
import sys
import torch as ch
import numpy as np
from PIL import Image
import torch.nn as nn
from datetime import datetime
import torch.nn.functional as F
from parsing import anomaly_parser
import torchvision.models as models
from utils.ad_utils import one_vs_all
from utils.data_utils import gen_transform
from utils.core_utils import( fix_seeds, stdout_logger, make_single_class_loader,
                              InputScaling, InputDescaling, InputDenormalize)
from models.alexnet import( AlexNet, AlexNetGen, AlexNetEncDec,
                            AlexNetEncDec_config, AlexNetComp, AlexNetComp_config)


# Parse arguments
parser= anomaly_parser()
args = parser.parse_args()


# Extra arguments
args.exp_name= datetime.now().strftime( "%Y_%m_%d_%H_%M_%S")
args.out_folder= os.path.join( args.out_dir, args.exp_name, "output")
if not os.path.exists( args.out_folder): os.makedirs( args.out_folder) # create outliers folder
if args.seed: fix_seeds( args.seed) # fix pytorch random seed


# set device
device= ch.device( 'cuda' if ch.cuda.is_available() else 'cpu') 


if args.stdout_logger:
  # Set standard output folder
  args.stdout_str= os.path.join( args.stdout_dir, args.exp_name + '.txt') # Standard output filename
  sys.stdout= stdout_logger( stdout_str= args.stdout_str)


# Classifier settings
if args.data== "ImageNet":
    args.num_classes= 1000
    args.mean=  ch.tensor( [ 0.485, 0.456, 0.406])
    args.std=ch.tensor( [ 0.229, 0.224, 0.225])
else: raise ValueError( "Undefined dataset. Check 'data' input argument.")


# Set data transformations
transform_test, norm_flag= gen_transform( mode= args.transform_test,
                                          init_dim= args.transform_init_dim)


# Create dataloader
if args.one_vs_all:
    _, in_test_loader= make_single_class_loader( dataset= args.in_class_dataset,
                                                 data= args.in_data_path,
                                                 samples= args.in_samples,
                                                 workers= args.num_workers,
                                                 batch_size= args.batch_size,
                                                 transform_train= transform_test,
                                                 transform_test= transform_test,
                                                 random_samples= args.random_samples)
else:
    raise ValueError( "Undefined anomaly detection strategy. Check 'one_vs_all' argument.")


# Load autoencoder
if args.generator_arch== "alexnet":
    model= AlexNetEncDec( num_classes= args.num_classes,
                          mean= args.mean,
                          std= args.std,
                          output_layer= args.output_layer,
                          upsample_mode= args.upsample_mode,
                          spectral_init= args.spectral_init)


    # Set layers and load checkpoint
    AlexNetEncDec_config( classifier= model.classifier,
                          generator= model.generator,
                          load_classifier= args.load_classifier,
                          load_generator= args.load_generator,
                          output_layer= args.output_layer)
    if norm_flag:
        # Replace normalization by scaling
        model.normalize= InputScaling() 
        model.denormalize= InputDescaling()
    else:
        model.denormalize= InputDenormalize( new_mean= args.mean,
                                               new_std= args.std) 
else: raise ValueError( "Undefined autoencoer model, check 'generator_arch' input argument.")


# Load comparator
if args.comparator_arch== "alexnet":
    comparator= AlexNetComp( num_classes= args.num_classes,
                             output_layer= args.output_layer,
                             mean= args.mean,
                             std= args.std)
    AlexNetComp_config( comparator= comparator.classifier,
                        load_comparator= args.load_comparator,
                        output_layer= args.comparator_layer,
                        strict= False)
else: raise ValueError( "Undefined comparator model, check 'comparator_arch' input argument.")


# Pass model to device
model.to( device) 
model.classifier.eval()
model.generator.eval()
comparator.to( device) 
comparator.eval()


# Model optimization criterion
pixel_crit= nn.L1Loss( reduction= args.reduction)
feat_crit= nn.MSELoss( reduction= args.reduction)


# One vs all classification
one_vs_all( in_test_loader= in_test_loader,
            act_reference= args.act_reference,
            comparator_layer= args.comparator_layer,
            comparator= comparator,
            feat_crit= feat_crit,
            pixel_crit= pixel_crit,
            variable= args.variable,
            model= model,
            optimizer= args.optimizer,
            step_size= args.step_size,
            sched_step= args.sched_step,
            sched_gamma= args.sched_gamma,
            iterations= args.iterations,
            out_folder= args.out_folder,
            feature_loss_weight= args.feature_loss_weight,
            pixel_loss_weight= args.pixel_loss_weight,
            randn_init= args.randn_init,
            export_output= args.export_output,
            batch_limit= args.batch_limit)
print( "Output directory: ", args.out_folder)

