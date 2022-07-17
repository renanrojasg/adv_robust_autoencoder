import os
import sys
import torch as ch
from datetime import datetime
from parsing import predict_parser
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.infer_utils import infer_dataset
from utils.core_utils import( stdout_logger, InputDenormalize, InputScaling,
                              InputDescaling, fix_seeds)
from utils.data_utils import gen_transform, custom_ds
from models.alexnet_skip import AlexNetWavEncDec, AlexNetWavEncDec_config
from models.alexnet import( AlexNetEncDec, AlexNetEncDec_config)
import torchvision.datasets as ds


# Parse arguments
parser= predict_parser()
args = parser.parse_args()

# Custom arguments
args.exp_name= datetime.now().strftime( "%Y_%m_%d_%H_%M_%S")
args._out_dir= os.path.join( args.out_dir, args.exp_name) # Full output dir
if args.reduction== "None": args.reduction= None
args.out_folder= os.path.join( args._out_dir, "output") # Predictions folder
if not os.path.exists( args.out_folder): os.makedirs( args.out_folder) 

if args.stdout_logger:
  # Set stdout logger
  args.stdout_str= os.path.join( args.stdout_dir, args.exp_name + '.txt') # Stdout filename
  sys.stdout= stdout_logger( stdout_str= args.stdout_str)

# Fix random seed
if args.seed: fix_seeds( args.seed)

# Dataset parameters
if args.dataset== "imagenet":
    args.num_classes= 1000
    args.mean= ch.tensor([0.485, 0.456, 0.406])
    args.std= ch.tensor([0.229, 0.224, 0.225])
elif args.dataset== "cifar10":
    # Normalization not applied
    args.num_classes= 10
else: raise ValueError( "Undefined dataset. Check 'dataset' input argument.")

# Set preprocessing
transform_test, norm_flag= gen_transform( mode= args.transform_test,
                                          init_dim= args.transform_init_dim)

# Set model
if args.arch== "alexnet":
    if args.wavelet_pooling:
        # Skip connections (Wavelet pooling)
        model= AlexNetWavEncDec( num_classes= args.num_classes,
                                 mean= args.mean,
                                 std= args.std,
                                 upsample_mode= args.upsample_mode,
                                 output_layer= args.output_layer,
                                 spectral_init= args.spectral_init)

        # Load checkpoint, set layers
        AlexNetWavEncDec_config( classifier= model.classifier, 
                                 num_classes= args.num_classes,
                                 generator= model.generator,
                                 load_classifier= args.load_classifier,
                                 load_generator= args.load_generator,
                                 output_layer= args.output_layer)
    else:
        # Original AlexNet
        model= AlexNetEncDec( num_classes= args.num_classes,
                              mean= args.mean,
                              std= args.std,
                              output_layer= args.output_layer,
                              upsample_mode= args.upsample_mode,
                              spectral_init= args.spectral_init)

        # Load checkpoint, set layers
        AlexNetEncDec_config( classifier= model.classifier,
                              generator= model.generator,
                              load_classifier= args.load_classifier,
                              load_generator= args.load_generator,
                              output_layer= args.output_layer)

    if norm_flag:
        # Replace standardization by rescaling
        model.normalize= InputScaling() 
        denormalize= InputDescaling()
    else:
        denormalize= InputDenormalize( new_mean= args.mean,
                                         new_std= args.std)
else:
    raise ValueError( "Undefined model, check 'arch' input argument.")

# Set model into parallel and eval mode
model= ch.nn.DataParallel( model).cuda()
model= model.eval()
if denormalize: denormalize= ch.nn.DataParallel( denormalize).cuda()

# Set dataloader
if args.dataset== "cifar10":
    # Default CIFAR10 dataloader
    test_dataset= ds.CIFAR10( args.data,
                              train= False,
                              transform= transform_test)
else:
    test_dataset= custom_ds( ImPath= args.data,
                             preprocess= transform_test)
val_loader= DataLoader( test_dataset,
                        shuffle= False,
                        batch_size= args.batch_size,
                        num_workers= args.num_workers)

# Infer
infer_dataset( lpips_net= args.lpips_net,
               transform_output= args.transform_output,
               transform_output_dim= args.transform_init_dim,
               model= model,
               denormalize= denormalize,
               infer_export= args.infer_export,
               val_loader= val_loader,
               out_folder= args.out_folder,
               compute_lpips= args.compute_lpips,
               compute_psnr= args.compute_psnr,
               compute_ssim= args.compute_ssim,
               norm_flag= norm_flag,
               noise_level= args.noise_level,
               reduction= args.reduction,
               pad_edges= args.pad_edges,
               pad_input= args.pad_input)
print( "Output directory: ", args._out_dir)
