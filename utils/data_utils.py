import os
from torchvision import transforms
import torch.utils.data as data
from utils.core_utils import is_image_file
from PIL import Image


# Custom dataset
class custom_ds( data.Dataset):
    def __init__( self, ImPath, preprocess,
                  samples= None):
        super( custom_ds, self).__init__()
        self.ImPath= ImPath
        self.image_list= [ os.path.join( root, name)
                           for root, dirs, files in os.walk( self.ImPath)
                           for name in files
                           if is_image_file( name)]

        # Keep first samples
        if samples is not None:
            self.image_list= self.image_list[ :samples]
        self.ImTransform= preprocess # set preprocessing

    def __getitem__( self, index):
        # Get image 
        ImgPath= self.image_list[ index]
        Img= self.ImTransform( Image.open( ImgPath).convert( 'RGB')) # resize
        fname= self.image_list[ index].rsplit( "/", 2)[ -1].rsplit( ".", 2)[ 0] # get filename
        return Img, fname

    def __len__( self):
        return len( self.image_list)


# Generic transformations
def gen_transform(mode,init_dim,crop_dim=224,
                  model_config="train",final_dim=None,rf_pad=None,
                  rf_crop01=None,rf_crop02=None,rf_full_dim=None):
    # Check model_config
    if model_config not in ( "train", "predict"):
        raise ValueError( "Undefined model configuration.\
                          Check the 'model_config' argument.")
    if mode== "skip":
        # No transformation applied.
        norm_flag= True
        transf= transforms.ToTensor()
    elif mode== "resize":
        # Resize
        norm_flag= False
        if model_config== "train":
            # Final scaling applied by training routine
            transf= transforms.Compose([ transforms.Resize( ( init_dim, init_dim)),
                                         transforms.ToTensor(),])
        else:
            transf= transforms.Compose([ transforms.Resize( ( init_dim, init_dim)),
                                         transforms.Resize( ( final_dim, final_dim)),
                                         transforms.ToTensor(),])
    elif mode== "resize_norm":
        # Resize and normalize/scale
        norm_flag= True
        if final_dim is not None:
            transf= transforms.Compose([ transforms.Resize(( init_dim, init_dim)),
                                         transforms.Resize(( final_dim, final_dim)),
                                         transforms.ToTensor(),])
        else:
            transf= transforms.Compose([ transforms.Resize(( init_dim, init_dim)),
                                         transforms.ToTensor(),])
    elif mode== "resize_crop_norm":
        # Resize, crop and normalize/scale
        norm_flag= True
        if final_dim is not None:
            transf= transforms.Compose([ transforms.Resize(( init_dim, init_dim)),
                                         transforms.Resize(( final_dim, final_dim)),
                                         transform.CenterCrop((crop_dim,crop_dim)),
                                         transforms.ToTensor(),])
        else:
            transf= transforms.Compose([ transforms.Resize(( init_dim, init_dim)),
                                         transforms.CenterCrop((crop_dim,crop_dim)),
                                         transforms.ToTensor(),])
    else:
        raise ValueError( "Wrong input transformations. Check 'mode' input argument.")
    return transf, norm_flag

