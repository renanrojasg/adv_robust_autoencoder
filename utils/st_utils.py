import os
import numpy as np
import torch as ch
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
import torch.utils.data as data
from piqa import LPIPS, PSNR, SSIM
from torchvision import transforms
from models.alexnet import AlexNet_pad
from utils.losses_utils import extract_features
from losses.losses import stylization_criterion
from utils.core_utils import( export_output, is_image_file, compute_label_info,
                              feature_wct)


# Set transformations 
def st_transforms( mode, init_dim, final_dim):
    norm_flag= True
    if mode== 'skip': transf= transforms.Compose([])
    elif mode== 'resize':
        transf= transforms.Compose([ transforms.Resize( ( init_dim, init_dim)),
                                     transforms.Resize( ( final_dim, final_dim)),])
    else: raise ValueError( "Undefined input transformations. Check 'mode' input argument.")
    return transf, norm_flag


# Stylization custom dataset
class st_custom_ds( data.Dataset):
    def __init__( self, im_path, seg_path, preprocess):
        super( st_custom_ds, self).__init__()
        self.im_path= im_path
        self.image_list= [ x for x in os.listdir( im_path) if is_image_file( x)]
        self.seg_path= seg_path
        self.preprocess= preprocess

        # Set preprocessing.
        self.im_transform= preprocess
        self.seg_transform= preprocess

    def __getitem__( self, index):
        # Get image, resize and cast to torch tensor.
        im_path= os.path.join( self.im_path, self.image_list[ index])
        img= self.im_transform( Image.open( im_path).convert( 'RGB'))
        img= transforms.ToTensor()( img)
        
        # Get label.
        try:
            seg_im_path= os.path.join( self.seg_path,self.image_list[ index])
            seg_img= self.seg_transform( Image.open( seg_im_path))
            if len( np.asarray( seg_img).shape) == 3:
                seg_img= change_seg( seg_img)
            seg_img= np.asarray( seg_img)
        except:
            seg_img= np.asarray( [])
        seg_img= ch.from_numpy( seg_img.copy())

        # Get filename.
        fname= self.image_list[ index].rsplit( "/", 2)[ -1].rsplit( ".", 2)[ 0]
        return img, seg_img, fname

    def __len__( self):
        return len( self.image_list)


# Universal style transfer
def st_universal( args, model_conv5, model_conv4,
                  model_conv3, model_conv2, model_conv1,
                  denormalize, comparator, comparator_arch,
                  content_loader, compare_layers, style_loader,
                  out_folder, device, infer_export,
                  compute_gram, compute_ssim, input_pad,
                  reduction):

    # Configure optimization criterion.
    criterion= stylization_criterion( loss_style_layers= compare_layers,
                                      loss_style_weight= 1,
                                      reduction= reduction)
    ssim_func= SSIM().cuda()
    style_loss_list=[]
    ssim_score_list=[]

    with ch.no_grad():
        # Iterate on dataloader
        content_iterator= tqdm( enumerate( content_loader), total= len( content_loader))
        for i, ( content, content_segment, content_fname) in content_iterator:
            content_segment= np.asarray( content_segment.squeeze( 0))
            if infer_export:
                # Export content reference
                export_output( image= content,
                               out_name= "content_" + content_fname[ 0],
                               out_path= os.path.join( out_folder),
                               epoch= 0,
                               index= i,
                               transform_output= args.transform_output,
                               transform_arg= args.transform_output_dim)
            content= content.cuda()
            style_iterator= tqdm( enumerate( style_loader), total= len( style_loader))
            for ii, ( style, style_segment, style_fname) in style_iterator:
                style_segment= np.asarray( style_segment.squeeze( 0))
                if infer_export:
                    # Export style reference
                    export_output( image= style,
                                   out_name= "style_" + style_fname[ 0],
                                   out_path= os.path.join( out_folder),
                                   epoch= 0,
                                   index= ii,
                                   transform_output= args.transform_output,
                                   transform_arg= args.transform_output_dim)
                output_name= content_fname[ 0] + "_" + style_fname[ 0]
                label_set, label_indicator = compute_label_info(content_segment, style_segment) # Set semantic labels
                style= style.cuda()
                stylized= _st_universal( model_conv5= model_conv5,
                                         model_conv4= model_conv4,
                                         model_conv3= model_conv3,
                                         model_conv2= model_conv2,
                                         model_conv1= model_conv1,
                                         denormalize= denormalize,
                                         content= content,
                                         style= style,
                                         label_set= label_set,
                                         label_indicator= label_indicator,
                                         input_pad= input_pad,
                                         device= device)

                # Compute metrics
                style_features, stylized_style_features= {}, {}
                if compute_gram:
                    extract_features( style= style,
                                      stylized= stylized,
                                      loss_style_layers= compare_layers,
                                      model= comparator,
                                      model_arch= comparator_arch,
                                      style_features= style_features,
                                      stylized_style_features= stylized_style_features)
                    style_loss= criterion( style_features= style_features,
                                           stylized_style_features= stylized_style_features)
                    style_loss_list.append( style_loss.flatten().tolist())
                if compute_ssim:
                    ssim_score= ssim_func( stylized, content)
                    ssim_score_list.append( ssim_score.flatten().tolist())
                if infer_export:
                    # Export style image
                    export_output( image= stylized,
                                   out_name= output_name,
                                   out_path= os.path.join( out_folder),
                                   epoch= 0,
                                   index= 0,
                                   transform_output= args.transform_output,
                                   transform_arg= args.transform_output_dim)

        # Export metrics
        if compute_gram:
            style_loss_list = [item for sublist in style_loss_list for item in sublist]
            np.savetxt( os.path.join( out_folder, "gram.csv"), style_loss_list, delimiter=",")
            print( "Average Gram loss: ", sum( style_loss_list)/ len( style_loss_list))
        if compute_ssim:
            ssim_score_list = [item for sublist in ssim_score_list for item in sublist]
            np.savetxt( os.path.join( out_folder, "ssim.csv"), ssim_score_list, delimiter=",")
            print( "Average SSIM: ", sum( ssim_score_list)/ len( ssim_score_list))
    return

# (Core) universal style transfer
def _st_universal( model_conv5, model_conv4, model_conv3,
                   model_conv2, model_conv1, denormalize,
                   content, style, device,
                   content_segment= None, style_segment= None, label_set= None,
                   input_pad= False, label_indicator= None):
    stylized= content.clone()

    # Pad image (False by default).
    if input_pad:
        stylized, r, c= AlexNet_pad( x= stylized, m= 224)

        # Further padding to avoid edge distorsions.
        stylized= F.pad( input= stylized,
                         pad= ( 112, 112, 112, 112),
                         mode= "reflect")

    # Multi-level stylization.
    # Align conv5 features.
    if model_conv5:
        content_feat= model_conv5.encoder_pass( x= stylized, feature_loss= False)
        style_feat= model_conv5.encoder_pass( x= style, feature_loss= False)
        stylized_feat= feature_wct( content_feat, style_feat,
                                    content_segment, style_segment, label_set,
                                    label_indicator, alpha=1, device= device)
        stylized= denormalize( model_conv5.decoder_pass( x= stylized_feat))

    # Align conv4 features.
    if model_conv4:
        content_feat= model_conv4.encoder_pass( x= stylized, feature_loss= False)
        style_feat= model_conv4.encoder_pass( x= style, feature_loss= False)
        stylized_feat= feature_wct( content_feat, style_feat,
                                    content_segment, style_segment, label_set,
                                    label_indicator, alpha=1, device= device)
        stylized= denormalize( model_conv4.decoder_pass( x= stylized_feat))

    # Align conv3 features.
    if model_conv3:
        content_feat= model_conv3.encoder_pass( x= stylized, feature_loss= False)
        style_feat= model_conv3.encoder_pass( x= style, feature_loss= False)
        stylized_feat= feature_wct( content_feat, style_feat,
                                    content_segment, style_segment, label_set,
                                    label_indicator, alpha=1, device= device)
        stylized= denormalize( model_conv3.decoder_pass( x= stylized_feat))

    # Align conv2 features.
    if model_conv2:
        content_feat= model_conv2.encoder_pass( x= stylized, feature_loss= False)
        style_feat= model_conv2.encoder_pass( x= style, feature_loss= False)
        stylized_feat= feature_wct( content_feat, style_feat,
                                    content_segment, style_segment, label_set,
                                    label_indicator, alpha=1, device= device)
        stylized= denormalize( model_conv2.decoder_pass( x= stylized_feat))

    # Align conv1 features.
    if model_conv1:
        content_feat= model_conv1.encoder_pass( x= stylized, feature_loss= False)
        style_feat= model_conv1.encoder_pass( x= style, feature_loss= False)
        stylized_feat= feature_wct( content_feat, style_feat,
                                    content_segment, style_segment, label_set,
                                    label_indicator, alpha=1, device= device)
        stylized= denormalize( model_conv1.decoder_pass( x= stylized_feat))

    # Remove edge and image padding.
    if input_pad:
        stylized= stylized[ :, :, 112: -112, 112:- 112]
        stylized= stylized[ :, :, :r, :c]
    return stylized

