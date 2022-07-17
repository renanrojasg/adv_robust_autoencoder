import os
import sys
import random
import numpy as np
import torch as ch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# Fix random seeds for pytorch, numpy, and python
def fix_seeds(seed= 0):
    ch.manual_seed(seed)
    ch.cuda.manual_seed(seed)
    ch.cuda.manual_seed_all(seed)
    ch.backends.cudnn.deterministic= True 
    return


# Configure stdout logger
class stdout_logger( object):
    def __init__( self, stdout_str):
        self.terminal= sys.stdout
        self.log= open( stdout_str, "a")
    def write( self, message):
        self.terminal.write( message)
        self.log.write( message)
    def flush( self):
        pass


# Squeeze feature map
def get_squeeze_feat(feat):
    _feat = feat.squeeze(0)
    size = _feat.size(0)
    return _feat.view(size, -1).clone()


# Get rank
def get_rank(singular_values, dim, eps= 0.00001):
    r = dim
    for i in range(dim - 1, -1, -1):
        if singular_values[i] >= eps:
            r = i + 1
            break
    return r


# Set common semantic labels
# Implementation based on https://github.com/clovaai/WCT2
def compute_label_info( content_segment, style_segment):
    if not content_segment.size or not style_segment.size:
        return None, None
    max_label= np.max( content_segment) + 1
    label_set= np.unique( content_segment)
    label_indicator= np.zeros( max_label)
    for l in label_set:
        content_mask= np.where( content_segment.reshape( content_segment.shape[ 0] * content_segment.shape[ 1])== l)
        style_mask= np.where( style_segment.reshape( style_segment.shape[ 0] * style_segment.shape[ 1]) == l)

        c_size= content_mask[ 0].size
        s_size= style_mask[ 0].size
        if c_size> 10 and s_size> 10 and c_size /s_size< 100 and s_size/ c_size< 100:
            label_indicator[ l] = True
        else:
            label_indicator[ l] = False
    return label_set, label_indicator


# Check file extension
# Implementation based on https://github.com/clovaai/WCT2
def is_image_file( filename):
    return any( filename.endswith( extension) for extension in [ ".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG", ".tif"])

# Whitening and coloring transformation
def feature_wct( content_feat, style_feat, content_segment=None, style_segment=None,
                 label_set=None, label_indicator=None, weight=1,
                 registers=None, alpha=1, device= 'cpu'):
    if label_set is not None:
        target_feature = wct_core_segment( content_feat, style_feat, content_segment, style_segment,
                                           label_set, label_indicator, weight, registers, device=device)
    else:
        target_feature = wct_core( content_feat, style_feat, device= device)
    target_feature = target_feature.view_as( content_feat)
    target_feature = alpha* target_feature+ (1 - alpha)* content_feat
    return target_feature


# (Core) Whitening and coloring transformation
# Implementation based on https://github.com/clovaai/WCT2
def wct_core( cont_feat, styl_feat, weight= 1, registers= None, device= 'cpu'):
    cont_feat= get_squeeze_feat(cont_feat)
    cont_min= cont_feat.min()
    cont_max= cont_feat.max()
    cont_mean= ch.mean(cont_feat, 1).unsqueeze(1).expand_as(cont_feat)
    cont_feat-= cont_mean

    if not registers:
        _, c_e, c_v = svd(cont_feat, iden=True, device=device)

        styl_feat = get_squeeze_feat(styl_feat)
        s_mean = ch.mean(styl_feat, 1)
        _, s_e, s_v = svd(styl_feat, iden=True, device=device)
        k_s = get_rank(s_e, styl_feat.size()[0])
        s_d = (s_e[0:k_s]).pow(0.5)
        EDE = ch.mm(ch.mm(s_v[:, 0:k_s], ch.diag(s_d) * weight), (s_v[:, 0:k_s].t()))

        if registers is not None:
            registers['EDE'] = EDE
            registers['s_mean'] = s_mean
            registers['c_v'] = c_v
            registers['c_e'] = c_e
    else:
        EDE = registers['EDE']
        s_mean = registers['s_mean']
        _, c_e, c_v = svd(cont_feat, iden=True, device=device)

    k_c = get_rank(c_e, cont_feat.size()[0])
    c_d = (c_e[0:k_c]).pow(-0.5)

    step1 = ch.mm(c_v[:, 0:k_c], ch.diag(c_d))
    step2 = ch.mm(step1, (c_v[:, 0:k_c].t()))
    whiten_cF = ch.mm(step2, cont_feat)

    targetFeature = ch.mm(EDE, whiten_cF)
    targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)
    targetFeature.clamp_(cont_min.detach().cpu(), cont_max.detach().cpu())

    return targetFeature


# SVD
# Implementation based on https://github.com/clovaai/WCT2
def svd(feat, iden=False, device='cpu'):
    size = feat.size()
    mean = ch.mean(feat, 1)
    mean = mean.unsqueeze(1).expand_as(feat)
    _feat = feat.clone()
    _feat -= mean
    if size[1] > 1:
        conv = ch.mm(_feat, _feat.t()).div(size[1] - 1)
    else:
        conv = ch.mm(_feat, _feat.t())
    if iden:
        conv += ch.eye(size[0]).to(device)
    
    # Precondition if svd diverges
    try:
        u, e, v = ch.svd(conv, some=False)
    except:
         print( "SVD convergence error. Preconditioning.")
         u, e, v = ch.svd(conv+ 1e-4* ch.eye( size[ 0]).to( device), some=False)
    return u, e, v



# Export output
def export_output( image, out_name, out_path,
                   epoch, index, transform_output= "skip",
                   transform_arg= None):

    for j in range( image.shape[ 0]):
        image_pil= transforms.ToPILImage()( image.detach().cpu()[ j, :])

        # Post-process
        if transform_output== "resize":
            image_pil= image_pil.resize( ( transform_arg, transform_arg)) # Bilinear interpolation by default.
        elif transform_output== "skip": pass
        else:
            raise ValueError( "Wrong output transformation. Check 'transform_output' argument.")
        fname= os.path.join( out_path, "img_" + str( epoch) + "_" + str( index) + "_" + str( j) + "_" + out_name + ".png")
        image_pil.save( fname, "PNG")
    return


# Single class dataloader
def make_single_class_loader( dataset, data, workers,
                              batch_size, transform_train, transform_test,
                              samples= False, train_single_class= False,
                              random_samples= False):
    if dataset== "cifar10":
        # Set train dataset
        train_dataset= datasets.CIFAR10( root= data,
                                         train= True,
                                         transform= transform_train)

        # is not False to include train_single_class= 0
        if train_single_class is not False:
            idx= [ i for i in range( len( train_dataset.targets)) if train_dataset.targets[ i] == train_single_class]
            train_dataset.targets= [ train_dataset.targets[ i] for i in idx]
            train_dataset.data= [ train_dataset.data[ i] for i in idx]

        # Set test dataset
        val_dataset= datasets.CIFAR10( root= data,
                                       train= False,
                                       transform= transform_test)

        if train_single_class is not False:
            idx= [ i for i in range( len( val_dataset.targets)) if val_dataset.targets[ i] == train_single_class]
            val_dataset.targets= [ val_dataset.targets[ i] for i in idx]
            val_dataset.data= [ val_dataset.data[ i] for i in idx]

        if samples:
            if random_samples:
                # Keep random indices instead of the first ones.
                L= len( val_dataset.targets)
                indices= random.sample( range( L), samples)
                val_dataset.targets= [ val_dataset.targets[ i] for i in indices]
                val_dataset.data= [ val_dataset.data[ i] for i in indices]
            else:
                # Keep first 'samples' samples (for preview purposes).
                val_dataset.targets= val_dataset.targets[ : samples]
                val_dataset.data= val_dataset.data[ : samples]

        # create dataloaders
        train_loader= DataLoader( train_dataset,
                                  batch_size= batch_size,
                                  shuffle= True,
                                  num_workers= workers,
                                  pin_memory= True)
        val_loader= DataLoader( val_dataset,
                                batch_size= batch_size,
                                shuffle= False,
                                num_workers= workers,
                                pin_memory= True)
    elif dataset== "cats_vs_dogs":
        # Set train dataset
        train_root= os.path.join( data, "train")
        train_dataset= datasets.ImageFolder( root= train_root,
                                             transform= transform_train)

        if train_single_class is not False:
            # imgs is a list of tuples that includes both fnames and labels
            labels = [ img[ 1] for img in train_dataset.imgs]
            idx= [ i for i in range( len( labels)) if labels[ i] == train_single_class]
            train_dataset.imgs= [ train_dataset.imgs[ i] for i in idx]
            train_dataset.samples= [ train_dataset.samples[ i] for i in idx]

        # Set val dataset
        val_root= os.path.join( data, "test")
        val_dataset= datasets.ImageFolder( root= val_root,
                                           transform= transform_train)

        if train_single_class is not False:
            labels = [ img[ 1] for img in val_dataset.imgs]
            idx= [ i for i in range( len( labels)) if labels[ i] == train_single_class]
            val_dataset.imgs= [ val_dataset.imgs[ i] for i in idx]
            val_dataset.samples= [ val_dataset.samples[ i] for i in idx]
 
        # Keep first 'samples' samples (for preview purposes).
        if samples:
            if random_samples:
                # Keep random indices instead of the first ones.
                L= len( val_dataset.imgs)
                indices= random.sample( range( L), samples)
                val_dataset.imgs= [ val_dataset.imgs[ i] for i in indices]
                val_dataset.samples= [ val_dataset.samples[ i] for i in indices]
            else:
                # Keep first 'samples' samples (for preview purposes).
                val_dataset.imgs= val_dataset.imgs[ : samples]
                val_dataset.samples= val_dataset.samples[ : samples]

        # create dataloaders
        train_loader= DataLoader( train_dataset,
                                  batch_size= batch_size,
                                  shuffle= True,
                                  num_workers= workers,
                                  pin_memory= True)
        val_loader= DataLoader( val_dataset,
                                batch_size= batch_size,
                                shuffle= False,
                                num_workers= workers,
                                pin_memory= True)
    else:
        raise ValueError( "Undefined dataset. Check 'dataset' input parameter.")

    return train_loader, val_loader


# Weights initialization
def weights_init( m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Input normalize
# Source: https://github.com/MadryLab/robustness/blob/master/robustness/tools/helpers.py
class InputNormalize( nn.Module):
    '''
    A module (custom layer) for normalizing the input to have a fixed 
    mean and standard deviation (user-specified).
    '''
    def __init__(self, new_mean, new_std):
        super(InputNormalize, self).__init__()
        new_std = new_std[..., None, None]
        new_mean = new_mean[..., None, None]

        self.register_buffer("new_mean", new_mean)
        self.register_buffer("new_std", new_std)

    def forward(self, x):
        x = ch.clamp(x, 0, 1)
        x_normalized = (x - self.new_mean)/self.new_std
        return x_normalized


# Denormalization
class InputDenormalize(nn.Module):
    def __init__( self, new_mean, new_std):
        super( InputDenormalize, self).__init__()
        new_std= new_std[ ..., None, None]
        new_mean= new_mean[ ..., None, None]
        self.register_buffer( "new_mean", new_mean)
        self.register_buffer( "new_std", new_std)
    def forward( self, x):
        x_denormalized= x* self.new_std+ self.new_mean
        x_denormalized= ch.clamp( x_denormalized, 0, 1)
        return x_denormalized


# Scaling
class InputScaling( nn.Module):
    def __init__( self):
        super( InputScaling, self).__init__()
    def forward( self, x):
        x_rescaled= ch.clamp( x, 0, 1)
        x_rescaled= 2* x- 1
        return x_rescaled


# Descaling
class InputDescaling( nn.Module):
    def __init__( self):
        super( InputDescaling, self).__init__()
    def forward( self, x):
        x_descaled= 0.5*( x+ 1)
        x_descaled= ch.clamp( x_descaled, 0, 1)
        return x_descaled


# AlexNet input pad
def AlexNet_pad( x, m= 224):
    [ _, _, r, c]= x.shape
    r_mod= np.mod( r, m)
    c_mod= np.mod( c, m)
    if r_mod!= 0:
        x= F.pad( x,
                  ( 0, 0, 0, m- r_mod),
                  mode= 'replicate') # Pad rows
    if c_mod!= 0:
        x= F.pad( x,
                  ( 0, m- c_mod, 0, 0),
                  mode= 'replicate') # Pad cols
    return x, r, c

