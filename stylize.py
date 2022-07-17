import os
import sys
import torch as ch
from datetime import datetime
from parsing import stylize_parser
from torch.utils.data import DataLoader
from utils.core_utils import( stdout_logger, InputDenormalize, InputScaling,
                              InputDescaling)
from utils.st_utils import st_universal, st_transforms, st_custom_ds
from models.vgg import vgg19Comp, vgg19Comp_config
from models.alexnet import( AlexNetEncDec, AlexNetEncDec_config, AlexNetComp,
                            AlexNetComp_config)


# Parse arguments
parser= stylize_parser()
args = parser.parse_args()

# Custom arguments
args.stylize_layers= tuple( args.stylize_layers)
args.exp_name= datetime.now().strftime( "%Y_%m_%d_%H_%M_%S")
args.out_folder= os.path.join( args.out_dir, args.exp_name, "output") # Output folder
if not os.path.exists( args.out_folder): os.makedirs( args.out_folder) # create output folder

# Dataset parameters
if args.dataset== "imagenet":
    args.num_classes= 1000
    args.mean= ch.tensor([0.485, 0.456, 0.406])
    args.std= ch.tensor([0.229, 0.224, 0.225])
else: raise ValueError( "Undefined dataset. Check 'dataset' input argument.")

if args.stdout_logger:
  # Set stdout
  args.stdout_str= os.path.join( args.stdout_dir, args.exp_name + '.txt') # Standard output filename
  sys.stdout= stdout_logger( stdout_str= args.stdout_str)

# Content and style data
content_transform, _= st_transforms( mode= args.transform_test,
                                     init_dim= args.content_transform_init_dim,
                                     final_dim= args.content_transform_final_dim)
style_transform, norm_flag= st_transforms( mode= args.transform_test,
                                           init_dim= args.style_transform_init_dim,
                                           final_dim= args.style_transform_final_dim)
content_data= st_custom_ds( im_path= args.content_images,
                            seg_path= args.content_labels,
                            preprocess= content_transform)
style_data= st_custom_ds( im_path= args.style_images,
                          seg_path= args.style_labels,
                          preprocess= style_transform)
content_loader= DataLoader( content_data,
                            batch_size= args.batch_size,
                            shuffle= False,
                            num_workers= args.workers)
style_loader= DataLoader( style_data,
                          batch_size= args.batch_size,
                          shuffle= False,
                          num_workers= args.workers)

# Multi-stage model
if args.arch== "alexnet":
    # Conv5 autoencoder
    model_conv5= AlexNetEncDec( num_classes= args.num_classes,
                                mean= args.mean,
                                std= args.std,
                                output_layer= "conv5",
                                upsample_mode= args.conv5_upsample_mode,
                                spectral_init= args.spectral_init)

    # Load checkpoints and set layers
    AlexNetEncDec_config( classifier= model_conv5.classifier,
                          generator= model_conv5.generator,
                          load_classifier= args.load_classifier,
                          load_generator= args.load_conv5_generator,
                          output_layer= "conv5")

    # Conv2 autoencoder
    model_conv2= AlexNetEncDec( num_classes= args.num_classes,
                                mean= args.mean,
                                std= args.std,
                                output_layer= "conv2",
                                upsample_mode= args.conv2_upsample_mode,
                                spectral_init= args.spectral_init)

    # Load checkpoints and set layers
    AlexNetEncDec_config( classifier= model_conv2.classifier,
                          generator= model_conv2.generator,
                          load_classifier= args.load_classifier,
                          load_generator= args.load_conv2_generator,
                          output_layer= "conv2")

    # Conv1 autoencoder
    model_conv1= AlexNetEncDec( num_classes= args.num_classes,
                                mean= args.mean,
                                std= args.std,
                                output_layer= "conv1",
                                upsample_mode= args.conv1_upsample_mode,
                                spectral_init= args.spectral_init)

    # Load checkpoints and set layers
    AlexNetEncDec_config( classifier= model_conv1.classifier,
                          generator= model_conv1.generator, 
                          load_classifier= args.load_classifier,
                          load_generator= args.load_conv1_generator,
                          output_layer= "conv1")

    # Replace normalizer by scaling.
    if norm_flag:
        model_conv5.normalize= InputScaling()
        model_conv2.normalize= InputScaling()
        model_conv1.normalize= InputScaling()
        if args.arch=="VGG16":
            model_conv4.normalize= InputScaling()
            model_conv3.normalize= InputScaling()
        denormalize= InputDescaling()
    else: denormalize= InputDenormalize( new_mean= args.mean,
                                         new_std= args.std)
else: raise ValueError( "Wrong model, check arch input argument.")

# Set comparator
if args.load_comparator:
    if args.comparator_arch== "alexnet":
        comparator= AlexNetComp( num_classes= args.num_classes,
                                 mean= args.mean,
                                 std= args.std,
                                 output_layer= "conv5")
        AlexNetComp_config( comparator= comparator.classifier,
                            load_comparator= args.load_comparator,
                            output_layer= "conv5")
    elif args.comparator_arch== "vgg19":
        comparator= vgg19Comp( num_classes= args.num_classes,
                               mean= args.mean,
                               std= args.std)
        vgg19Comp_config( comparator= comparator.classifier,
                          load_comparator= args.load_comparator)
    else: raise ValueError( "Undefined comparator architecture. Check 'comparator_arch' argument.")
else: comparator= None

# Pass model to device
device= ch.device( 'cuda' if ch.cuda.is_available() else 'cpu')
if args.load_conv1_generator:
    model_conv1= model_conv1.to( device)
    model_conv1.eval()
else: model_conv1= None
if args.load_conv2_generator:
    model_conv2= model_conv2.to( device)
    model_conv2.eval()
else: model_conv2= None
if args.load_conv3_generator:
    model_conv3= model_conv3.to( device)
    model_conv3.eval()
else: model_conv3= None
if args.load_conv4_generator:
    model_conv4= model_conv4.to( device)
    model_conv4.eval()
else: model_conv4= None
if args.load_conv5_generator:
    model_conv5= model_conv5.to( device)
    model_conv5.eval()
else: model_conv5= None
if denormalize: denormalize= denormalize.to( device)
if comparator:
    comparator= comparator.to( device)
    comparator.eval()

# Stylize
st_universal( args= args,
              model_conv5= model_conv5,
              model_conv4= model_conv4,
              model_conv3= model_conv3,
              model_conv2= model_conv2,
              model_conv1= model_conv1,
              denormalize= denormalize,
              comparator= comparator,
              comparator_arch= args.comparator_arch,
              compare_layers= args.compare_layers,
              content_loader= content_loader,
              style_loader= style_loader,
              out_folder= args.out_folder,
              infer_export= args.infer_export,
              device= device,
              compute_gram= args.compute_gram,
              compute_ssim= args.compute_ssim,
              input_pad= args.pad_input,
              reduction= args.reduction)
print( "Output folder: ", args.out_folder)
