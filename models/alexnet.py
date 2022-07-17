import dill
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from utils.core_utils import( InputNormalize, InputDenormalize, InputScaling,
                              InputDescaling, AlexNet_pad)


# AlexNet
class AlexNet(nn.Module):
    def __init__(self, output_layer, num_classes= 1000,
                 fully_convolutional= None, adapt_avg_pool= False):
        super(AlexNet, self).__init__()
        self.fully_convolutional= fully_convolutional
        self.output_layer= output_layer
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        if not( adapt_avg_pool):
            self.avgpool= nn.Identity()
        else: self.avgpool= nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),)
        self.last_relu= nn.ReLU( inplace= True)
        self.last_layer= nn.Linear( 4096, num_classes)


    # Extract single conv layers
    def single_level( self, x, layer):
        if layer== "conv1":
            for i in range( 0, 3):
                x= self.features[ i]( x)
            return x
        if layer== "conv2":
            for i in range( 3, 6):
                x= self.features[ i]( x)
            return x
        if layer== "conv3":
            for i in range( 6, 8):
                x= self.features[ i]( x)
            return x
        if layer== "conv4":
            for i in range( 8, 10):
                x= self.features[ i]( x)
            return x
        if layer== "conv5":
            for i in range( 10, 13):
                x= self.features[ i]( x)
            return x
        else:
            raise ValueError( "Wrong layer. Check layer input argument.")


    # Extract multiple convolutional layers
    def extract_layers( self, x, feat_dict, feat_level):
        for i in range( 0, 3):
            x= self.features[ i]( x)
        if "conv1" in feat_level: feat_dict[ "conv1"]= x
        for i in range( 3, 6):
            x= self.features[ i]( x)
        if "conv2" in feat_level: feat_dict[ "conv2"]= x
        for i in range( 6, 8):
            x= self.features[ i]( x)
        if "conv3" in feat_level: feat_dict[ "conv3"]= x       
        for i in range( 8, 10):
            x= self.features[ i]( x)
        if "conv4" in feat_level: feat_dict[ "conv4"]= x
        for i in range( 10, 13):
            x= self.features[ i]( x)
        if "conv5" in feat_level: feat_dict[ "conv5"]= x
        return x

    def forward( self, x, feature_loss= False):
        x_out= self.features( x)
        if feature_loss:
            # Keep conv5 for feature loss.
            feat= x_out.clone() 
        if self.output_layer== "fc8":
            x= self.avgpool( x_out)
            x= x.view( x.size( 0), 9216)
            x_latent= self.classifier( x)
            x_relu= self.last_relu( x_latent)
            x_out= self.last_layer( x_relu)
        elif self.output_layer== "fc6":
            # Fc6 output, inc. ReLU.
            x= self.avgpool( x_out)
            x= x.view( x.size( 0), 9216)
            x_latent= self.classifier( x)
            x_out= x_latent 
        elif self.output_layer== "conv5":
            # Conv5 output, inc. ReLU and Pooling.
            pass 
        elif self.output_layer== "conv2":
            # Conv2 output, inc. ReLU and Pooling.
            pass 
        elif self.output_layer== "conv1":
            # Conv1 output, inc. ReLU and Pooling.
            pass 
        else:
            raise ValueError( "Undefined output layer. Check 'output_layer' input argument.")
        if feature_loss:
            return feat, x_out
        else:
            return x_out


# AlexNet generator
class AlexNetGen( nn.Module):
    def __init__( self, output_layer, upsample_mode,
                  num_classes= 1000, fully_convolutional= None, leakyrelu_factor= 0.2,
                  spectral_init= False):
        super().__init__()
 
        self.output_layer= output_layer
        self.upsample_mode= upsample_mode
        sa_layer= nn.Identity

        # Spectral norm
        if spectral_init: init_layer= spectral_norm
        else: init_layer= nn.Identity

        # FC layers
        self.linear3= nn.Linear( num_classes, 4096)
        self.linear2= nn.Linear( 4096, 4096)
        self.linear1= nn.Linear( 4096, 256* 4* 4)
        self.pad1= nn.ZeroPad2d( 1) # change spatial support to [6, 6].

        # Convolutional module
        if self.upsample_mode== "conv5_tconv":
            self.gen= nn.Sequential( init_layer( nn.Conv2d( 256, 256, 3, padding= 1, bias= False)),
                                     nn.BatchNorm2d( 256),
                                     nn.ReLU(),
                                     # [ b, 256, 7, 7]
                                     init_layer( nn.ConvTranspose2d( 256, 256, 4, padding= 1, bias= False)),
                                     nn.BatchNorm2d( 256),
                                     nn.ReLU(),
                                     init_layer( nn.Conv2d( 256, 256, 3, padding= 1, bias= False)),
                                     nn.BatchNorm2d( 256),
                                     nn.ReLU(),
                                     # [ b, 256, 14, 14]
                                     init_layer( nn.ConvTranspose2d( 256, 256, 4, stride= 2, padding= 1, bias= False)),
                                     nn.BatchNorm2d( 256),
                                     nn.ReLU(),
                                     init_layer( nn.Conv2d( 256, 256, 3, padding= 1, bias= False)),
                                     nn.BatchNorm2d( 256),
                                     nn.ReLU(),
                                     # [ b, 128, 28, 28]
                                     init_layer( nn.ConvTranspose2d( 256, 128, 4, stride= 2, padding= 1, bias= False)),
                                     nn.BatchNorm2d( 128),
                                     nn.ReLU(),
                                     init_layer( nn.Conv2d( 128, 128, 3, padding= 1, bias= False)),
                                     nn.BatchNorm2d( 128),
                                     nn.ReLU(),
                                     # [ b, 64, 56, 56]
                                     init_layer( nn.ConvTranspose2d( 128, 64, 4, stride= 2, padding= 1, bias= False)),
                                     nn.BatchNorm2d( 64),
                                     nn.ReLU(),
                                     init_layer( nn.Conv2d( 64, 64, 3, padding= 1, bias= False)),
                                     nn.BatchNorm2d( 64),
                                     nn.ReLU(),
                                     # [ b, 32, 112, 112]
                                     init_layer( nn.ConvTranspose2d( 64, 32, 4, stride= 2, padding= 1, bias= False)), 
                                     nn.BatchNorm2d( 32),
                                     nn.ReLU(),
                                     init_layer( nn.Conv2d( 32, 32, 3, padding= 1, bias= False)),
                                     nn.BatchNorm2d( 32),
                                     nn.ReLU(),
                                     # [ b, 3, 224, 224]
                                     init_layer( nn.ConvTranspose2d( 32, 3, 4, stride= 2, padding= 1, bias= False)),
                                     nn.BatchNorm2d( 3),
                                     nn.ReLU(),
                                     init_layer( nn.Conv2d( 3, 3, 3, padding= 1)),
                                     nn.Tanh(),)
        elif self.upsample_mode== "conv5_interp":
            self.gen= nn.Sequential( init_layer( nn.Conv2d( 256, 256, 3, padding= 1, bias= False)),
                                     nn.BatchNorm2d( 256),
                                     nn.ReLU(),
                                     # [ b, 256, 7, 7]
                                     init_layer( nn.ConvTranspose2d( 256, 256, 4, padding= 1, bias= False)), 
                                     nn.BatchNorm2d( 256),
                                     nn.ReLU(),
                                     init_layer( nn.Conv2d( 256, 256, 3, padding= 1, bias= False)),
                                     nn.BatchNorm2d( 256),
                                     nn.ReLU(),
                                     # [ b, 256, 14, 14]
                                     nn.Upsample( scale_factor= 2, mode= 'nearest'),
                                     init_layer( nn.Conv2d( 256, 256, 3, padding= 1, bias= False)),
                                     nn.BatchNorm2d( 256),
                                     nn.ReLU(),
                                     # [ b, 256, 28, 28]
                                     nn.Upsample( scale_factor= 2, mode= 'nearest'),
                                     init_layer( nn.Conv2d( 256, 128, 3, padding= 1, bias= False)),
                                     nn.BatchNorm2d( 128),
                                     nn.ReLU(),
                                     # [ b, 128, 56, 56]
                                     nn.Upsample( scale_factor= 2, mode= 'nearest'),
                                     init_layer( nn.Conv2d( 128, 64, 3, padding= 1, bias= False)),
                                     nn.BatchNorm2d( 64),
                                     nn.ReLU(),
                                     # [ b, 64, 112, 112]
                                     nn.Upsample( scale_factor= 2, mode= 'nearest'),
                                     init_layer( nn.Conv2d( 64, 32, 3, padding= 1, bias= False)),
                                     nn.BatchNorm2d( 32),
                                     nn.ReLU(),
                                     # [ b, 32, 224, 224]
                                     nn.Upsample( scale_factor= 2, mode= 'nearest'),
                                     init_layer( nn.Conv2d( 32, 3, 3, padding= 1, bias= False)),
                                     nn.BatchNorm2d( 3),
                                     nn.ReLU(),
                                     init_layer( nn.Conv2d( 3, 3, 3, padding= 1)),
                                     nn.Tanh(),)
        elif self.upsample_mode== "conv1_interp":
            self.gen= nn.Sequential( init_layer( nn.Conv2d( 64, 64, 3, padding= 1, bias= False)),
                                     nn.BatchNorm2d( 64),
                                     nn.ReLU(),
                                     # [ b, 64, 28, 28]
                                     init_layer( nn.ConvTranspose2d( 64, 64, 4, padding= 1, bias= False)),
                                     nn.BatchNorm2d( 64),
                                     nn.ReLU(),
                                     init_layer( nn.Conv2d( 64, 64, 3, padding= 1, bias= False)),
                                     nn.BatchNorm2d( 64),
                                     nn.ReLU(),
                                     # [ b, 64, 56, 56]
                                     nn.Upsample( scale_factor= 2, mode= 'nearest'),
                                     init_layer( nn.Conv2d( 64, 64, 3, padding= 1, bias= False)),
                                     nn.BatchNorm2d( 64),
                                     nn.ReLU(),
                                     init_layer( nn.Conv2d( 64, 32, 3, padding= 1, bias= False)),
                                     nn.BatchNorm2d( 32),
                                     nn.ReLU(),
                                     # [ b, 32, 112, 112]
                                     nn.Upsample( scale_factor= 2, mode= 'nearest'),
                                     init_layer( nn.Conv2d( 32, 32, 3, padding= 1, bias= False)),
                                     nn.BatchNorm2d( 32),
                                     nn.ReLU(),
                                     # [ b, 32, 224, 224]
                                     nn.Upsample( scale_factor= 2, mode= 'nearest'),
                                     init_layer( nn.Conv2d( 32, 16, 3, padding= 1, bias= False)),
                                     nn.BatchNorm2d( 16),
                                     nn.ReLU(),
                                     # [ b, 3, 224, 224]
                                     init_layer( nn.Conv2d( 16, 3, 3, padding= 1, bias= False)),
                                     nn.BatchNorm2d( 3),
                                     nn.ReLU(),
                                     init_layer( nn.Conv2d( 3, 3, 3, padding= 1)),
                                     nn.Tanh(),)
        elif self.upsample_mode== "conv2_interp":
            self.gen= nn.Sequential( init_layer( nn.Conv2d( 192, 192, 3, padding= 1, bias= False)),
                                     nn.BatchNorm2d( 192),
                                     nn.ReLU(),
                                     # [ b, 192, 14, 14]
                                     init_layer( nn.ConvTranspose2d( 192, 192, 4, padding= 1, bias= False)),
                                     nn.BatchNorm2d( 192),
                                     nn.ReLU(),
                                     init_layer( nn.Conv2d( 192, 96, 3, padding= 1, bias= False)),
                                     nn.BatchNorm2d( 96),
                                     nn.ReLU(),
                                     # [ b, 96, 28, 28]
                                     nn.Upsample( scale_factor= 2, mode= 'nearest'),
                                     init_layer( nn.Conv2d( 96, 96, 3, padding= 1, bias= False)),
                                     nn.BatchNorm2d( 96),
                                     nn.ReLU(),
                                     init_layer( nn.Conv2d( 96, 64, 3, padding= 1, bias= False)),
                                     nn.BatchNorm2d( 64),
                                     nn.ReLU(),
                                     # [ b, 64, 56, 56]
                                     nn.Upsample( scale_factor= 2, mode= 'nearest'),
                                     init_layer( nn.Conv2d( 64, 64, 3, padding= 1, bias= False)),
                                     nn.BatchNorm2d( 64),
                                     nn.ReLU(),
                                     # [ b, 64, 112, 112]
                                     nn.Upsample( scale_factor= 2, mode= 'nearest'),
                                     init_layer( nn.Conv2d( 64, 64, 3, padding= 1, bias= False)),
                                     nn.BatchNorm2d( 64),
                                     nn.ReLU(),
                                     # [ b, 64, 224, 224]
                                     nn.Upsample( scale_factor= 2, mode= 'nearest'),
                                     init_layer( nn.Conv2d( 64, 32, 3, padding= 1, bias= False)),
                                     nn.BatchNorm2d( 32),
                                     nn.ReLU(),
                                     # [ b, 32, 224, 224]
                                     init_layer( nn.Conv2d( 32, 3, 3, padding= 1, bias= False)),
                                     nn.BatchNorm2d( 3),
                                     nn.ReLU(),
                                     # [ b, 32, 224, 224]
                                     init_layer( nn.Conv2d( 3, 3, 3, padding= 1)),
                                     nn.Tanh(),)
        else:
            raise ValueError( "Undefined upsample mode. Check 'upsample_mode' input argument.")

    def forward(self, x):
        # Invert features
        if self.output_layer== "fc8":
            out= F.relu( self.linear3( x))
            out= F.relu( self.linear2( out))
            out= F.relu( self.linear1( out))

            # Reshape to [ b, 256, 4, 4], pad to [ b, 256, 6, 6]
            out= self.pad1( out.view( out.shape[ 0], 256, 4, 4))
        elif self.output_layer== "conv5":
            # [ b, 256, 6, 6]
            out= x.clone()
        elif self.output_layer== "conv2":
            # [ b, 384, 13, 13]
            out= x.clone()
        elif self.output_layer== "conv1":
            # [ b, 64, 55, 55]
            out= x.clone()
        else:
            raise ValueError( "Undefined output layer. Check 'output_layer' input argument.")
        return self.gen( out)


# AlexNet autoencoder
class AlexNetEncDec( nn.Module):
    def __init__( self, output_layer, upsample_mode,
                  mean, std, load_generator= None,
                  num_classes= 1000, spectral_init= False, transform_init_dim= None):
        super().__init__()

        self.normalize= InputNormalize( mean, std)
        self.classifier= AlexNet( num_classes= num_classes,
                                  output_layer= output_layer)
        self.generator= AlexNetGen( num_classes= num_classes,
                                    output_layer= output_layer,
                                    spectral_init= spectral_init,
                                    upsample_mode= upsample_mode)

    # Preprocessing, encoder
    def encoder_pass( self, x, feature_loss,
                      apply_normalization= True):
        if apply_normalization: x= self.normalize( x)
        if feature_loss: 
            # Output prediction and feature map
            feat, out= self.classifier( x, feature_loss= feature_loss)
            return feat, out
        else:
            # Output prediction
            out= self.classifier( x, feature_loss= feature_loss)
            return out

    # Decoder
    def decoder_pass( self, x):
        out= self.generator( x)
        return out

    # Forward pass
    def forward( self, x, feature_loss,
                 apply_normalization= True):
        if apply_normalization: x= self.normalize( x)
        if feature_loss:
            # output prediction and feature map
            feat, x= self.classifier( x, feature_loss= True)
            x= self.generator( x)
            if self.scaling_module:
                return feat, self.downscale( x)
            else:
                return feat, x
        else:
            # output prediction
            x= self.classifier( x, feature_loss= False)
            x= self.generator( x)
            return x


# AlexNet comparator
class AlexNetComp( nn.Module):
    def __init__( self, output_layer, mean,
                  std, num_classes= 1000):
        super().__init__()
        self.normalize= InputNormalize( mean, std)
        self.classifier= AlexNet( num_classes= num_classes,
                                  output_layer= output_layer)
    def forward( self, x, apply_normalization= True):
        if apply_normalization: x= self.normalize( x)
        x= self.classifier( x, feature_loss= False)
        return x


# AlexNet comparator configuration
def AlexNetComp_config( comparator, load_comparator, output_layer,
                        strict= True):
    if load_comparator== None:
        raise ValueError( "No comparator weights specified.")
    checkpoint= torch.load( load_comparator, pickle_module= dill)
    sd= checkpoint[ 'model']
    # Remove 'module.model.' prefix, keep items from 'model'.
    sd= { k[ len( 'module.model.'):]:v for k,v in sd.items()\
          if k.startswith( 'module.model.')}
    comparator.load_state_dict( sd, strict= strict)

    # Freeze classifier weights
    for p in comparator.parameters():
        p.requires_grad = False

    # Reshape model according to output layer
    if output_layer== "fc8":
        # Keep all layers.
        pass
    elif output_layer== "conv5":
        # Remove dense layers
        comparator.avgpool= nn.Sequential()
        comparator.classifier= nn.Sequential()
        comparator.last_layer= nn.Sequential()
    else:
        raise ValueError( "Undefined final layer specified. Check 'output_layer' argument.")
    return


# Configure AlexNetEncDec
def AlexNetEncDec_config( classifier, generator, output_layer,
                          load_classifier, load_generator= None, resume_training= False,
                          transform_init_dim= None, transform_upscaling= None):
    freeze_generator= not( resume_training)

    # Load classifier weights
    if load_classifier== None:
        raise ValueError( "No classifier weights specified.")
    checkpoint= torch.load( load_classifier, pickle_module= dill)
    sd= checkpoint[ 'model']
    # Remove 'module.model.' prefix, keep items from 'model'.
    sd= { k[ len( 'module.model.'):]:v for k,v in sd.items()\
          if k.startswith( 'module.model.')}
    classifier.load_state_dict( sd, strict= True)
    for p in classifier.parameters(): p.requires_grad = False

    # Reshape model according to output layer
    if output_layer== "fc8":
        # Keep all layers.
        pass
    elif output_layer== "conv5":
        # Remove dense layer
        classifier.avgpool= nn.Sequential()
        classifier.classifier= nn.Sequential()
        classifier.last_layer= nn.Sequential()
        generator.linear3= nn.Sequential()
        generator.linear2= nn.Sequential()
        generator.linear1= nn.Sequential()
    elif output_layer== "conv2":
        for i in range( 6, 13): classifier.features[ i]= nn.Sequential()
        classifier.avgpool= nn.Sequential()
        classifier.classifier= nn.Sequential()
        classifier.last_layer= nn.Sequential()
        generator.linear3= nn.Sequential()
        generator.linear2= nn.Sequential()
        generator.linear1= nn.Sequential()
    elif output_layer== "conv1":
        for i in range( 3, 13): classifier.features[ i]= nn.Sequential()
        classifier.avgpool= nn.Sequential()
        classifier.classifier= nn.Sequential()
        classifier.last_layer= nn.Sequential()
        generator.linear3= nn.Sequential()
        generator.linear2= nn.Sequential()
        generator.linear1= nn.Sequential()
    else:
        raise ValueError( "Undefined final layer specified. Check 'output_layer' argument.")

    # Load generator weights
    if load_generator is not None:
        checkpoint= torch.load( load_generator, pickle_module= dill)
        sd= checkpoint[ 'model']
        # Remove the 'module.model.' prefix, keep items from 'model'.
        sd= { k[ len( 'module.generator.'):]:v for k,v in sd.items()\
              if k.startswith( 'module.generator.')}
        generator.load_state_dict( sd, strict= True)
        if freeze_generator:
            for p in generator.parameters(): p.requires_grad = False
    return


# Configure AlexNetDisc
def AlexNetDisc_config( discriminator, load_discriminator):
    checkpoint= torch.load( load_discriminator, pickle_module= dill)
    sd= checkpoint[ 'discriminator']
    sd= { k[ len( 'module.'):]:v for k,v in sd.items()\
          if k.startswith( 'module.')}
    discriminator.load_state_dict( sd)
    return


# AlexNet discriminator
class AlexNetDisc( nn.Module):
    def __init__( self, disc_bn, disc_model= "new",
                  spectral_init= False, leakyrelu_factor= 0.2):
        super( AlexNetDisc, self).__init__()
        self.disc_model= disc_model
        sa_layer= nn.Identity

        # Spectral normalization
        if spectral_init: init_layer= spectral_norm
        else: init_layer= nn.Sequential([])

        if disc_bn:
            raise ValueError( "Discriminator model with BN not implemented.\
                              Check 'disc_model' and 'disc_bn' input arguments.")
        else:
            if disc_model== "original":
                # input: [ b, 3, 224, 224]
                self.features= nn.Sequential( 
                               # [ b, 32, 56, 56]
                               init_layer( nn.Conv2d( 3, 32, 7, stride= 4, padding= 1)),
                               nn.ReLU( inplace= True),
                               sa_layer( in_channels= 32, key_channels= 8, head_count= 4, value_channels= 8,
                                         spectral_init= spectral_init,
                                         leakyrelu_factor= leakyrelu_factor),
                               # [ b, 64, 52, 52]
                               init_layer( nn.Conv2d( 32, 64, 5)),
                               nn.ReLU( inplace= True),
                               sa_layer( in_channels= 64, key_channels= 16, head_count= 4, value_channels= 16,
                                         spectral_init= spectral_init,
                                         leakyrelu_factor= leakyrelu_factor),
                               # [ b, 128, 23, 23]
                               init_layer( nn.Conv2d( 64, 128, 3, stride= 2)),
                               nn.ReLU( inplace= True),
                               sa_layer( in_channels= 128, key_channels= 32, head_count= 4, value_channels= 32,
                                         spectral_init= spectral_init,
                                         leakyrelu_factor= leakyrelu_factor),
                               # [ b, 256, 21, 21]
                               init_layer( nn.Conv2d( 128, 256, 3, stride= 1)),
                               nn.ReLU( inplace= True),
                               sa_layer( in_channels= 256, key_channels= 64, head_count= 4, value_channels= 64,
                                         spectral_init= spectral_init,
                                         leakyrelu_factor= leakyrelu_factor),
                               # [ b, 256, 11, 11]
                               init_layer( nn.Conv2d( 256, 256, 3, stride= 2)),
                               nn.ReLU( inplace= True),
                               )
                # [ b, 256, 1, 1]
                self.avpool= nn.AdaptiveAvgPool2d( 1)
                self.classifier01= nn.Sequential( # input: [ b, 1, 256]
                                 # [ b, 1, 1024] # 9216 corresponds to flattened conv5 [ b, 256, 6, 6]
                                 init_layer( nn.Linear( 9216, 1024)),
                                 nn.ReLU( inplace= True),
                                 # [ b, 1, 1], Modification: output single channel instead of two channels.
                                 init_layer( nn.Linear( 1024, 512)),
                                 nn.ReLU( inplace= True),
                                 )
                self.classifier02= nn.Sequential( # input: [ b, 1, 256]
                                 # 50% (default) dropout
                                 nn.Dropout(),
                                 # [ b, 1, 512]
                                 init_layer( nn.Linear( 256+ 512, 512)),
                                 nn.ReLU( inplace= True),
                                 nn.Dropout(),
                                 # [ b, 1, 1], Modification: output single channel
                                 init_layer( nn.Linear( 512, 1)),
                                 nn.Sigmoid(), # Keep between 0 and 1
                                 )
            else:
                raise ValueError( "Selected Discriminator model without BN not implemented.\
                                  Check 'disc_model' and 'disc_bn' input arguments.")

    def forward( self, input, feat_target= None):
        if self.disc_model== "original":
            out= self.features( input)
            cat01= self.avpool( out).view( out.size( 0), 256) # input branch
            cat02= self.classifier01( feat_target) # feature branch
            out= torch.cat( ( cat01, cat02), 1) # concatenate along channels
            out= self.classifier02( out)
        else:
            raise ValueError( "Undefined discriminator model. Check 'disc_model' argument.")
        return out

