import dill
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.alexnet import AlexNet
from torch.nn.utils import spectral_norm
from utils.core_utils import InputNormalize
from robustness.tools.custom_modules import FakeReLUM

# Haar wavelet operators
# based on https://github.com/clovaai/WCT2
def get_wav( in_channels, pool= True):
    harr_wav_L = 1 / np.sqrt( 2) * np.ones(( 1, 2))
    harr_wav_H = 1 / np.sqrt( 2) * np.ones(( 1, 2))
    harr_wav_H[ 0, 0] = -1 * harr_wav_H[ 0, 0]

    harr_wav_LL= np.transpose( harr_wav_L)* harr_wav_L
    harr_wav_LH= np.transpose( harr_wav_L)* harr_wav_H
    harr_wav_HL= np.transpose( harr_wav_H)* harr_wav_L
    harr_wav_HH= np.transpose( harr_wav_H)* harr_wav_H

    filter_LL= torch.from_numpy( harr_wav_LL).unsqueeze( 0)
    filter_LH= torch.from_numpy( harr_wav_LH).unsqueeze( 0)
    filter_HL= torch.from_numpy( harr_wav_HL).unsqueeze( 0)
    filter_HH= torch.from_numpy( harr_wav_HH).unsqueeze( 0)

    if pool:
        net= nn.Conv2d
    else:
        net= nn.ConvTranspose2d

    LL= net( in_channels, in_channels, kernel_size= 2,
             stride= 2, padding= 0, bias= False,
             groups= in_channels)
    LH= net( in_channels, in_channels, kernel_size= 2,
             stride= 2, padding= 0, bias= False,
             groups=in_channels)
    HL= net( in_channels, in_channels, kernel_size=2,
             stride= 2, padding= 0, bias= False,
             groups= in_channels)
    HH= net( in_channels, in_channels, kernel_size= 2,
             stride= 2, padding= 0, bias= False,
             groups= in_channels)

    LL.weight.requires_grad= False
    LH.weight.requires_grad= False
    HL.weight.requires_grad= False
    HH.weight.requires_grad= False

    LL.weight.data= filter_LL.float().unsqueeze( 0).expand( in_channels, -1, -1, -1)
    LH.weight.data= filter_LH.float().unsqueeze( 0).expand( in_channels, -1, -1, -1)
    HL.weight.data= filter_HL.float().unsqueeze( 0).expand( in_channels, -1, -1, -1)
    HH.weight.data= filter_HH.float().unsqueeze( 0).expand( in_channels, -1, -1, -1)

    return LL, LH, HL, HH


# Wavelet pooling class
# Based on https://github.com/clovaai/WCT2
class WavePool( nn.Module):
    def __init__( self, in_channels):
        super( WavePool, self).__init__()
        self.LL, self.LH, self.HL, self.HH= get_wav( in_channels)

    def forward( self, x):
        return self.LL( x), self.LH( x), self.HL( x), self.HH( x)


# Wavelet unpooling class
# Based on https://github.com/clovaai/WCT2
class WaveUnpool( nn.Module):
    def __init__( self, in_channels, option_unpool= 'sum'):
        super( WaveUnpool, self).__init__()
        self.in_channels= in_channels
        self.option_unpool= option_unpool
        self.LL, self.LH, self.HL, self.HH= get_wav( self.in_channels, pool= False)

    def forward( self, LL, LH, HL, HH, original= None):
        if self.option_unpool== 'sum':
            return self.LL( LL) + self.LH( LH) + self.HL( HL) + self.HH( HH)
        elif self.option_unpool== 'cat5' and original is not None:
            return torch.cat( [ self.LL( LL), self.LH( LH), self.HL( HL),
                                self.HH( HH), original],
                              dim= 1)
        else:
            raise NotImplementedError


# AlexNet Wavelet Pooling
class AlexNetWav( nn.Module):
    def __init__( self, output_layer, num_classes= 1000):
        super( AlexNetWav, self).__init__()
        self.output_layer= output_layer

        # Feature extraction
        self.conv1_1= nn.Conv2d( 3, 64, kernel_size= 11, stride= 4, padding= 2)
        self.pool1= WavePool( 64) # downsampling factor 2
        self.conv2_1= nn.Conv2d( 64, 192, kernel_size= 5, padding= 2)
        self.pool2= WavePool( 192) # downsampling factor 2
        self.conv3_1= nn.Conv2d( 192, 384, kernel_size= 3, padding= 1)
        self.conv3_2= nn.Conv2d( 384, 256, kernel_size= 3, padding= 1)
        self.conv3_3= nn.Conv2d( 256, 256, kernel_size= 3, padding= 1)
        self.pool3= WavePool( 256) # downsampling factor 2


        # Classification module
        self.avgpool= nn.AdaptiveAvgPool2d( ( 6, 6))
        self.classifier= nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),)
        self.last_relu= nn.ReLU( inplace= True)
        self.last_relu_fake= FakeReLUM()
        self.last_layer= nn.Linear( 4096, num_classes)


    # Extract single conv layers
    def single_level( self, x, layer, skips= None):
        if layer== "conv1":
            x= self.conv1_1( x)
            return x
        if layer== "conv2":
            x= F.relu( x)
            x, LH1, HL1, HH1= self.pool1( x)
            if skips is not None:
                # Save details as skip connections
                skips[ 'pool1']=[ LH1, HL1, HH1]
            x= self.conv2_1( x)           
            return x
        if layer== "conv3":
            x= F.relu( x)
            x, LH2, HL2, HH2= self.pool2( x)
            if skips is not None:
                skips[ 'pool2']=[ LH2, HL2, HH2]
            x= self.conv3_1( x)
            return x
        if layer== "conv4":
            x= F.relu( x)
            x= self.conv3_2( x)
            return x
        if layer== "conv5":
            x= F.relu( x) 
            x= self.conv3_3( x)
            return x
        if layer== "top":
            x= F.relu( x)
            x, LH3, HL3, HH3= self.pool3( x)
            if skips is not None:
                skips[ 'pool3']=[ LH3, HL3, HH3]
            return x
        else:
            raise ValueError( "Undefined layer. Check layer input argument.")


    def forward( self, x, skips,
                 with_latent=False, fake_relu=False, no_relu=False,
                 feature_loss= False):
        x_out= F.relu( self.conv1_1( x))
        x_out, LH1, HL1, HH1= self.pool1( x_out)
        # Save details as skip connections
        skips[ 'pool1']=[ LH1, HL1, HH1]
        x_out= F.relu( self.conv2_1( x_out))
        x_out, LH2, HL2, HH2= self.pool2( x_out)
        skips[ 'pool2']=[ LH2, HL2, HH2]
        x_out= F.relu( self.conv3_1( x_out))
        x_out= F.relu( self.conv3_2( x_out))
        x_out= F.relu( self.conv3_3( x_out))
        x_out, LH3, HL3, HH3= self.pool3( x_out)
        skips[ 'pool3']=[ LH3, HL3, HH3]

        if feature_loss:
            # Keep conv5 for feature loss.
            feat= x_out.clone()
        if self.output_layer== "fc":
            x= self.avgpool( x_out)
            x= x.view(x.size( 0), 256 * 6 * 6)
            x_latent= self.classifier( x)
            x_relu= self.last_relu_fake( x_latent) if fake_relu \
                    else self.last_relu(x_latent)
            x_out= self.last_layer( x_relu)

        if with_latent and no_relu:
            return x_out, x_latent
        if with_latent:
            return x_out, x_relu
        if feature_loss:
            return feat, x_out
        else:
            return x_out


# AlexNet generator (Wavelet Pooling)
class AlexNetWavGen( nn.Module):
    def __init__( self, output_layer, upsample_mode,
                  num_classes, spectral_init, option_unpool= "sum"):
        super().__init__()
 
        self.output_layer= output_layer
        self.upsample_mode= upsample_mode
        self.option_unpool= option_unpool

        # Spectral normalization layer
        if spectral_init: init_layer= spectral_norm
        else: init_layer= nn.Sequential()

        # FC layers
        self.linear3= nn.Linear( num_classes, 4096)
        self.linear2= nn.Linear( 4096, 4096)
        self.linear1= nn.Linear( 4096, 256* 4* 4)
        self.pad1= nn.ReflectionPad2d( 1)

        # Convolutional module
        if self.upsample_mode== "conv5_tconv":
            self.conv7_1= nn.Conv2d( 256, 256, 3, padding= 1, bias= False)
            self.bn7_1= nn.BatchNorm2d( 256)
            self.conv6_1= nn.Conv2d( 256, 256, 3, padding= 1, bias= False)
            self.bn6_1= nn.BatchNorm2d( 256)
            self.conv5_1= nn.Conv2d( 256, 256, 3, padding= 1, bias= False)
            self.bn5_1= nn.BatchNorm2d( 256)
            self.conv5_2= nn.Conv2d( 256, 192, 3, padding= 1, bias= False)
            self.bn5_2= nn.BatchNorm2d( 192)
            self.conv4_1= nn.Conv2d( 192, 128, 3, padding= 1, bias= False)
            self.bn4_1= nn.BatchNorm2d( 128)
            self.conv4_2= nn.Conv2d( 128, 64, 3, padding= 1, bias= False)
            self.bn4_2= nn.BatchNorm2d( 64)
            self.conv3_1= nn.Conv2d( 64, 64, 3, padding= 1, bias= False)
            self.bn3_1= nn.BatchNorm2d( 64)
            # [ b, 32, 112, 112]
            self.tconv2= nn.ConvTranspose2d( 64, 32, 4, stride= 2, padding= 1, bias= False)
            self.bn2= nn.BatchNorm2d( 32)
            self.conv2_1= nn.Conv2d( 32, 32, 3, padding= 1, bias= False)
            self.bn2_1= nn.BatchNorm2d( 32)
            # [ b, 3, 224, 224]
            self.tconv1= nn.ConvTranspose2d( 32, 3, 4, stride= 2, padding= 1, bias= False)
            self.bn1= nn.BatchNorm2d( 3)
            self.conv1_1= nn.Conv2d( 3, 3, 3, padding= 1)

            # Padding for skip connections
            self.pad= nn.ReflectionPad2d( ( 1, 0, 0, 1))
            self.pad5= nn.ReflectionPad2d( ( 1, 0, 0, 1))
            self.pad4= nn.ReflectionPad2d( ( 1, 0, 0, 1))
            self.pad3= nn.ReflectionPad2d( ( 2, 0, 0, 2))
            self.unpool3= WaveUnpool( 256, self.option_unpool)
            self.unpool2= WaveUnpool( 192, self.option_unpool)
            self.unpool1= WaveUnpool( 64, self.option_unpool)
        elif self.upsample_mode== "conv5_interp":
            self.conv7_1= init_layer( nn.Conv2d( 256, 256, 3, padding= 1, bias= False))
            self.bn7_1= nn.BatchNorm2d( 256)
            self.conv6_1= init_layer( nn.Conv2d( 256, 256, 3, padding= 1, bias= False))
            self.bn6_1= nn.BatchNorm2d( 256)
            self.conv5_1= init_layer( nn.Conv2d( 256, 256, 3, padding= 1, bias= False))
            self.bn5_1= nn.BatchNorm2d( 256)
            self.conv5_2= init_layer( nn.Conv2d( 256, 192, 3, padding= 1, bias= False))
            self.bn5_2= nn.BatchNorm2d( 192)
            self.conv4_1= init_layer( nn.Conv2d( 192, 128, 3, padding= 1, bias= False))
            self.bn4_1= nn.BatchNorm2d( 128)
            self.conv4_2= init_layer( nn.Conv2d( 128, 64, 3, padding= 1, bias= False))
            self.bn4_2= nn.BatchNorm2d( 64)
            self.conv3_1= init_layer( nn.Conv2d( 64, 64, 3, padding= 1, bias= False))
            self.bn3_1= nn.BatchNorm2d( 64)
            # [ b, 32, 112, 112]
            self.nn2= nn.Upsample( scale_factor= 2, mode= 'nearest')
            self.conv2= init_layer( nn.Conv2d( 64, 32, 3, padding= 1, bias= False))
            self.bn2= nn.BatchNorm2d( 32)
            self.conv2_1= init_layer( nn.Conv2d( 32, 32, 3, padding= 1, bias= False))
            self.bn2_1= nn.BatchNorm2d( 32)
            # [ b, 3, 224, 224]
            self.nn1= nn.Upsample( scale_factor= 2, mode= 'nearest')
            self.conv1= init_layer( nn.Conv2d( 32, 3, 3, padding= 1, bias= False))
            self.bn1= nn.BatchNorm2d( 3)
            self.conv1_1= init_layer( nn.Conv2d( 3, 3, 3, padding= 1))

            # Padding for skip connections
            self.pad= nn.ReflectionPad2d( ( 1, 0, 0, 1))
            self.pad5= nn.ReflectionPad2d( ( 1, 0, 0, 1))
            self.pad4= nn.ReflectionPad2d( ( 1, 0, 0, 1))
            self.pad3= nn.ReflectionPad2d( ( 2, 0, 0, 2))
            self.unpool3= WaveUnpool( 256, self.option_unpool)
            self.unpool2= WaveUnpool( 192, self.option_unpool)
            self.unpool1= WaveUnpool( 64, self.option_unpool)       
        else:
            raise ValueError( "Undefined upsample mode. Check upsample_mode argument.")

    def forward(self, x, skips):
        if self.output_layer== "fc8":
            # Original linear layer as final layer
            out= F.relu( self.linear3( x))
            out= F.relu( self.linear2( out))
            out= F.relu( self.linear1( out))
            # Reshape to [ b, 256, 4, 4], padded to [ b, 256, 6, 6]
            out= self.pad1( out.view( out.shape[ 0], 256, 4, 4)) 
        elif self.output_layer== "conv5":
            out= x.clone() # [ b, c, 6, 6]
        else:
            raise ValueError( "Undefined output layer. Check 'output_layer' argument.")
        if self.upsample_mode== "conv5_tconv":
            # [ b, 256, 6, 6]
            out= F.relu( self.bn7_1( self.conv7_1( out)))
            LH1, HL1, HH1= skips[ 'pool3']
            out= self.unpool3( out, LH1, HL1, HH1)
            out= F.relu( self.bn6_1( self.conv6_1( out)))
            # [ b, 256, 14, 14]
            out= self.pad5( out)
            out= F.relu( self.bn5_1( self.conv5_1( out)))
            out= F.relu( self.bn5_2( self.conv5_2( out)))
            LH2, HL2, HH2= skips[ 'pool2']
            out= self.unpool2( out, LH2, HL2, HH2)
            out= self.pad4( out)
            out= F.relu( self.bn4_1( self.conv4_1( out)))
            out= F.relu( self.bn4_2( self.conv4_2( out)))
            LH3, HL3, HH3= skips[ 'pool1']
            out= self.unpool1( out, LH3, HL3, HH3)
            out= self.pad3( out)
            out= F.relu( self.bn3_1( self.conv3_1( out)))
            # [ b, 32, 112, 112]
            out= F.relu( self.bn2( self.tconv2( out)))
            out= F.relu( self.bn2_1( self.conv2_1( out)))
            # [ b, 3, 224, 224]
            out= F.relu( self.bn1( self.tconv1( out)))
            # No regularization on final layer
            out= self.conv1_1( out)
            # Keep output between -1 and 1.
            return F.tanh( out)
        elif self.upsample_mode== "conv5_interp":
            # [ b, 256, 6, 6]
            out= F.relu( self.bn7_1( self.conv7_1( out)))
            LH1, HL1, HH1= skips[ 'pool3']
            out= self.unpool3( out, LH1, HL1, HH1)
            out= F.relu( self.bn6_1( self.conv6_1( out)))
            # [ b, 256, 14, 14]
            out= self.pad5( out)
            out= F.relu( self.bn5_1( self.conv5_1( out)))
            out= F.relu( self.bn5_2( self.conv5_2( out)))
            LH2, HL2, HH2= skips[ 'pool2']
            out= self.unpool2( out, LH2, HL2, HH2)
            out= self.pad4( out)
            out= F.relu( self.bn4_1( self.conv4_1( out)))
            out= F.relu( self.bn4_2( self.conv4_2( out)))
            LH3, HL3, HH3= skips[ 'pool1']
            out= self.unpool1( out, LH3, HL3, HH3)
            out= self.pad3( out)
            out= F.relu( self.bn3_1( self.conv3_1( out)))
            out= self.nn2( out)
            # [ b, 32, 112, 112]
            out= F.relu( self.bn2( self.conv2( out)))
            out= F.relu( self.bn2_1( self.conv2_1( out)))
            out= self.nn1( out)
            # [ b, 3, 224, 224]
            out= F.relu( self.bn1( self.conv1( out)))
            out= self.conv1_1( out)
            return F.tanh( out)
        else:
            raise ValueError( "Undefined upsample mode. Check upsample_mode argument.")


# AlexNet (Wavelet pooling) autoencoder
class AlexNetWavEncDec( nn.Module):
    def __init__( self, mean, std,
                  output_layer, num_classes, spectral_init,
                  upsample_mode, leakyrelu_factor= None, option_unpool= "sum"):
        super().__init__()

        self.normalize= InputNormalize( mean, std)
        self.classifier= AlexNetWav( num_classes= num_classes,
                                     output_layer= output_layer)
        self.generator= AlexNetWavGen( num_classes= num_classes,
                                       output_layer= output_layer,
                                       upsample_mode= upsample_mode,
                                       spectral_init=spectral_init,
                                       option_unpool= option_unpool)

    def forward( self, x, feature_loss):
        # Skip connections dictionary
        skips={}

        x= self.normalize( x)
        if feature_loss:
            # Output prediction and feature map
            feat, x= self.classifier( x, skips, feature_loss= feature_loss)
            x= self.generator( x, skips)
            return feat, x
        else:
            # Output prediction only
            x= self.classifier( x, skips, feature_loss= feature_loss)
            x= self.generator( x, skips)
            return x


# AlexNet (wavelet pooling) autoencoder config
def AlexNetWavEncDec_config( classifier, generator, load_classifier, load_generator,
                             num_classes, output_layer, fully_convolutional= False):
    if load_classifier== None:
        raise ValueError( "No classifier weights specified. Check 'load_classifier' input argument")
    _classifier= AlexNet( num_classes= num_classes,
                          output_layer= output_layer,
                          fully_convolutional= fully_convolutional)
    checkpoint= torch.load( load_classifier, pickle_module= dill)
    sd= checkpoint[ 'model']
    # Remove the 'module.model.' prefix. Keep items from 'model'
    sd= { k[ len( 'module.model.'):]:v for k,v in sd.items()\
          if k.startswith( 'module.model.')}
    _classifier.load_state_dict( sd)

    # Load weights from sequential
    classifier.conv1_1.weight= _classifier.features[ 0].weight
    classifier.conv1_1.bias= _classifier.features[ 0].bias
    classifier.conv2_1.weight= _classifier.features[ 3].weight
    classifier.conv2_1.bias= _classifier.features[ 3].bias
    classifier.conv3_1.weight= _classifier.features[ 6].weight
    classifier.conv3_1.bias= _classifier.features[ 6].bias
    classifier.conv3_2.weight= _classifier.features[ 8].weight
    classifier.conv3_2.bias= _classifier.features[ 8].bias
    classifier.conv3_3.weight= _classifier.features[ 10].weight
    classifier.conv3_3.bias= _classifier.features[ 10].bias

    # Freeze classifier weights
    for p in classifier.parameters():
        p.requires_grad= False      

    # Set classifier output layer
    if output_layer== "fc8":
        pass
    elif output_layer== "conv5":
        # Remove FC layer from encoder
        classifier.avgpool= nn.Sequential()
        classifier.classifier= nn.Sequential()
        classifier.last_layer= nn.Sequential()
        generator.linear3= nn.Sequential()
        generator.linear2= nn.Sequential()
        generator.linear1= nn.Sequential()
    else:
        raise ValueError( "Undefined classifier final layer specified.\
                          Check 'output_layer' argument.")

    # Load generator weights
    if load_generator is not None:
        checkpoint= torch.load( load_generator, pickle_module= dill)
        sd= checkpoint[ 'model']

        # Remove the 'module.model.' prefix. Keep items from 'model'
        sd= { k[ len( 'module.generator.'):]:v for k,v in sd.items()\
                if k.startswith( 'module.generator.')}

        # Remove 'unpool' items.
        sd= { k: v for k, v in sd.items()\
              if not k.startswith( 'unpool')}

        # Load, dismiss unpooling weights.
        log= generator.load_state_dict( sd, strict= False)
        print( "log: ", log)
        assert log.missing_keys==[ "unpool3.LL.weight",
                                   "unpool3.LH.weight",
                                   "unpool3.HL.weight",
                                   "unpool3.HH.weight",
                                   "unpool2.LL.weight",
                                   "unpool2.LH.weight",
                                   "unpool2.HL.weight",
                                   "unpool2.HH.weight",
                                   "unpool1.LL.weight",
                                   "unpool1.LH.weight",
                                   "unpool1.HL.weight",
                                   "unpool1.HH.weight"]

        # Freeze generator weights
        for p in generator.parameters():
            p.requires_grad = False
    return
