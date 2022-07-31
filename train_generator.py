import os
import sys
import torch as ch
from datetime import datetime
from parsing import train_gen_parser
from torchvision import transforms
from argparse import ArgumentParser
from robustness.tools import helpers
from torch.utils.data import DataLoader
from torch.optim import SGD, lr_scheduler
from utils.train_utils import train_model
from utils.data_utils import gen_transform, custom_ds
from utils.core_utils import( fix_seeds, stdout_logger, make_single_class_loader,
                               weights_init, InputScaling, InputDescaling,
                               InputDenormalize)
from models.alexnet_skip import AlexNetWavEncDec, AlexNetWavEncDec_config
from models.alexnet import( AlexNetEncDec, AlexNetEncDec_config, AlexNetDisc,
                            AlexNetDisc_config, AlexNetComp, AlexNetComp_config)
import torchvision.datasets as ds


# Parse arguments
parser= train_gen_parser()
args = parser.parse_args()

# Custom arguments
args.disc_adam_betas= tuple( args.disc_adam_betas)
args.disc_labels= tuple( args.disc_labels)
args.exp_name= datetime.now().strftime( "%Y_%m_%d_%H_%M_%S")
args._out_dir= os.path.join( args.out_dir, args.exp_name) # Full output dir

# Check weight initialization
if ( args.weights_init and ( args.load_generator is not None)):
    raise ValueError( "Weight initialization and resume training cannot be set simultaneously.\
                       Check 'weights_init' and 'load_generator' input arguments.")

# Create output folders
if not os.path.exists( args._out_dir): os.makedirs( args._out_dir)

# Configure preview
args.preview= ( args.preview_path is not None)
if args.preview:
    args.infer_export= True
    args.prev_dir= os.path.join( args._out_dir, "preview")
    if not os.path.exists( args.prev_dir): os.makedirs( args.prev_dir)
else:
    args.infer_export= False
    args.prev_dir= None

if args.stdout_logger:
    # Set stdout
    args.stdout_str= os.path.join( args.stdout_dir, args.exp_name + '.txt')
    sys.stdout= stdout_logger( stdout_str= args.stdout_str)

# Fix random seed
if args.seed: fix_seeds( args.seed)

# Dataset parameters
if args.dataset== "imagenet":
    args.num_classes= 1000
    args.mean= ch.tensor([0.485, 0.456, 0.406])
    args.std= ch.tensor([0.229, 0.224, 0.225])
    args.data_train= os.path.join( args.data, "train")
    args.data_val= os.path.join( args.data, "val")
elif args.dataset== "imagenette":
    args.num_classes= 1000
    args.mean= ch.tensor([0.485, 0.456, 0.406])
    args.std= ch.tensor([0.229, 0.224, 0.225])
    args.data_train= os.path.join( args.data, "train")
    args.data_val= os.path.join( args.data, "val")
elif args.dataset== "cifar10":
    # Normalization not applied
    args.num_classes= 10
else: raise ValueError( "Undefined dataset. Check 'dataset' input argument.")

# Transformations
transform_train, norm_flag= gen_transform( mode= args.transform_train,
                                           init_dim= args.transform_init_dim)
transform_val, _= gen_transform( mode= args.transform_test,
                                 init_dim= args.transform_init_dim)

# Set model
if args.arch== "alexnet":
    if args.wavelet_pooling:
        # AlexNet + Wavelet Pooling
        model= AlexNetWavEncDec( num_classes= args.num_classes,
                                 mean= args.mean,
                                 std= args.std,
                                 upsample_mode= args.upsample_mode,
                                 output_layer= args.output_layer,
                                 spectral_init= args.spectral_init)
        # Load checkpoints, set layers
        AlexNetWavEncDec_config( classifier= model.classifier,
                                 generator= model.generator,
                                 load_classifier= args.load_classifier,
                                 load_generator= args.load_generator,
                                 num_classes= args.num_classes,
                                 output_layer= args.output_layer)
        if norm_flag:
            # Replace standardization by normalization
            model.normalize= InputScaling()
            denormalize= InputDescaling()

        if args.adversarial_loss:
            # Set discriminator
            discriminator= AlexNetDisc( disc_model= args.disc_model,
                                        disc_bn= args.disc_bn,
                                        leakyrelu_factor= args.leakyrelu_factor,
                                        spectral_init= args.spectral_init)
            if args.load_generator is not None:
                # Load discriminator
                # Discriminator included in generator cp
                AlexNetDisc_config( discriminator= discriminator,
                                    load_discriminator= args.load_generator) 
        else:
            discriminator= None

        # Set comparator
        if args.load_comparator:
            comparator= AlexNetComp( num_classes= args.num_classes,
                                     mean= args.mean,
                                     std= args.std,
                                     output_layer= args.comparator_layer)
            AlexNetComp_config( comparator= comparator.classifier,
                                load_comparator= args.load_comparator,
                                output_layer= args.comparator_layer)
        else: comparator= None
    else: 
        # Original AlexNet
        model= AlexNetEncDec( num_classes= args.num_classes,
                              mean= args.mean,
                              std= args.std,
                              output_layer= args.output_layer,
                              upsample_mode= args.upsample_mode,
                              spectral_init= args.spectral_init)

        # Set checkpoints and layers
        AlexNetEncDec_config( classifier= model.classifier, 
                              generator= model.generator,
                              load_classifier= args.load_classifier,
                              load_generator= args.load_generator,
                              output_layer= args.output_layer)
        if norm_flag:
            model.normalize= InputScaling()
            denormalize= InputDescaling()
        else:
            # Invert standardization
            denormalize= InputDenormalize( new_mean= args.mean,
                                             new_std= args.std)

        if args.adversarial_loss:
            # Set discriminator if required
            discriminator= AlexNetDisc( disc_model= args.disc_model,
                                        disc_bn= args.disc_bn,
                                        leakyrelu_factor= args.leakyrelu_factor,
                                        spectral_init= args.spectral_init)
            if args.load_generator is not None:
                AlexNetDisc_config( discriminator= discriminator,
                                    load_discriminator= args.load_generator)
        else:
            discriminator= None

        # Set comparator
        if args.load_comparator:
            comparator= AlexNetComp( num_classes= args.num_classes,
                                     mean= args.mean,
                                     std= args.std,
                                     output_layer= args.comparator_layer)
            AlexNetComp_config( comparator= comparator.classifier,
                                load_comparator= args.load_comparator,
                                output_layer= args.comparator_layer)
        else:
            comparator= None
else:
    raise ValueError( "Wrong model. Check 'arch' input argument.")

# Weight initialization
if args.weights_init:
    model.generator.apply( weights_init)
    if args.adversarial_loss:
        discriminator.apply( weights_init)

# Set generator optimizer and scheduler
if args.g_opt== 'adam':
    g_opt= ch.optim.Adam( model.parameters(),
                          lr= args.gen_lr,
                          betas= args.gen_adam_betas,
                          weight_decay= args.gen_adam_wd)
    g_sched = lr_scheduler.StepLR( g_opt,
                                   step_size= args.step_lr,
                                   gamma= args.step_lr_gamma)
else:
    raise ValueError( "Undefined generator optimizer. Check 'g_opt' input argument.")

# Model settings, classifier in eval mode
model= ch.nn.DataParallel( model).cuda()
model.module.classifier= model.module.classifier.eval()
if denormalize: denormalize= denormalize.cuda()

# Set discriminator optimizer and scheduler
if discriminator:
    d_opt= ch.optim.Adam( discriminator.parameters(),
                          lr= args.disc_lr,
                          betas= args.disc_adam_betas,
                          weight_decay= args.disc_adam_wd)
    d_sched= lr_scheduler.StepLR( d_opt,
                                  step_size= args.step_lr,
                                  gamma= args.step_lr_gamma)
    discriminator= ch.nn.DataParallel( discriminator).cuda()
else:
    discriminator= None
    d_opt= None
    d_sched= None

# Comparator settings
if comparator:
    comparator= ch.nn.DataParallel( comparator).cuda()
    comparator= comparator.eval()

# Set dataloaders
if args.train_single_class is not False: # 'is not False': allows for single_class to be 0
    train_loader, val_loader= make_single_class_loader( dataset= args.single_class_dataset,
                                                        data= args.data,
                                                        train_single_class= args.train_single_class,
                                                        workers= args.num_workers,
                                                        batch_size= args.batch_size,
                                                        transform_train= transform_train,
                                                        transform_test= transform_val)
else:
    if args.dataset== "cifar10":
        # Default CIFAR10 dataloader
        train_dataset= ds.CIFAR10( args.data,
                                   train= True,
                                   transform= transform_train)
        val_dataset= ds.CIFAR10( args.data,
                                 train= False,
                                 transform= transform_val)
    else:    
        train_dataset= custom_ds( ImPath= args.data_train,
                                  preprocess= transform_train,
                                  samples= args.samples)
        val_dataset= custom_ds( ImPath= args.data_val,
                                preprocess= transform_val,
                                samples= args.samples)
    train_loader= DataLoader( train_dataset,
                              shuffle= args.shuffle,
                              batch_size= args.batch_size,
                              num_workers= args.num_workers)
    val_loader= DataLoader( val_dataset,
                            shuffle= False,
                            batch_size= args.batch_size,
                            num_workers= args.num_workers)

# Set preview dataloader
if args.preview:
    if args.train_single_class is not False:
        _, preview_loader= make_single_class_loader( dataset= args.single_class_dataset,
                                                     data= args.data,
                                                     train_single_class= args.train_single_class,
                                                     workers= args.num_workers,
                                                     batch_size= args.batch_size,
                                                     transform_train= transform_train,
                                                     transform_test= transform_val,
                                                     samples= args.preview_samples)
    elif args.dataset== "cifar10":
        _, preview_loader= make_single_class_loader( dataset= args.dataset,
                                                     data= args.data,
                                                     workers= args.num_workers,
                                                     batch_size= args.batch_size,
                                                     transform_train= transform_train,
                                                     transform_test= transform_val,
                                                     samples= args.preview_samples,
                                                     random_samples= True)
    else:
        preview_dataset= custom_ds( ImPath= args.preview_path,
                                    preprocess= transform_val)
        preview_loader= DataLoader( preview_dataset,
                                    shuffle= False,
                                    batch_size= args.batch_size,
                                    num_workers= args.num_workers)
else: preview_loader= None

# Train model
train_model( transform_train= args.transform_train,
             noise_level= args.noise_level,
             pix_loss= args.pixel_loss,
             pix_loss_weight= args.pixel_loss_weight,
             feat_loss= args.feature_loss,
             feat_loss_weight= args.feature_loss_weight,
             adv_loss= args.adversarial_loss,
             adv_loss_weight= args.adversarial_loss_weight,
             disc_loss_weight= args.disc_loss_weight,
             reduction= args.reduction,
             model= model,
             discriminator= discriminator,
             comparator= comparator,
             g_opt= g_opt,
             g_sched= g_sched,
             d_opt= d_opt,
             d_sched= d_sched,
             denormalize= denormalize,
             train_loader= train_loader,
             val_loader= val_loader,
             preview_loader= preview_loader,
             load_generator= args.load_generator,
             norm_flag= norm_flag,
             out_dir= args._out_dir,
             prev_dir= args.prev_dir,
             preview= args.preview,
             epochs= args.epochs,
             disc_labels= args.disc_labels,
             disc_normalize= args.disc_normalize,
             transform_test= args.transform_test,
             transform_output= args.transform_output,
             transform_output_dim= args.transform_init_dim,
             transform_final_dim= args.transform_final_dim,
             infer_export= args.infer_export,
             cp_step= args.cp_step,
             val_step= args.val_step)
print( "Output folder: ", args._out_dir)

