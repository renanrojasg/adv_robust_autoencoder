import os
import numpy as np
import torch as ch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from piqa import LPIPS, PSNR, SSIM
from models.alexnet import AlexNet_pad


# Infer function
def infer_dataset( transform_output, transform_output_dim, infer_export,
                   lpips_net, model, denormalize,
                   val_loader, out_folder, compute_lpips,
                   compute_psnr, compute_ssim, norm_flag,
                   noise_level, reduction, pad_edges,
                   pad_input):
    with ch.no_grad():
        _infer_loop( transform_output= transform_output,
                     transform_output_dim= transform_output_dim,
                     infer_export= infer_export,
                     lpips_net= lpips_net,
                     model= model,
                     denormalize= denormalize,
                     val_loader= val_loader,
                     out_folder= out_folder,
                     compute_lpips= compute_lpips,
                     compute_psnr= compute_psnr,
                     compute_ssim= compute_ssim,
                     norm_flag= norm_flag,
                     pad_edges= pad_edges,
                     pad_input= pad_input,
                     noise_level= noise_level,
                     reduction= reduction)
    return


# Inference loop
def _infer_loop( transform_output, transform_output_dim, model,
                 denormalize, infer_export, val_loader,
                 out_folder, norm_flag, lpips_net= None,
                 reduction= "sum", compute_lpips= False, compute_ssim= False,
                 compute_psnr= False, pad_edges= False, noise_level= False,
                 pad_input= False, epoch= None):

    model= model.eval()
    iterator= tqdm( enumerate( val_loader), total= len( val_loader))

    # Create score list
    batch_list, sample_list, fname_list= None, None, None
    lpips_list, psnr_list, ssim_list= None, None, None
    lpips_noisy_list, psnr_noisy_list, ssim_noisy_list= None, None, None
    lpips_func, psnr_func, ssim_func= None, None, None
    if compute_lpips or compute_psnr or compute_ssim:
        batch_list= []
        sample_list= []
        fname_list= []
    if compute_lpips:
        lpips_func= LPIPS( network= lpips_net, reduction= reduction).cuda()
        lpips_list= []
        # Log observation LPIPS
        if noise_level: lpips_noisy_list=[] 
    if compute_psnr:
        psnr_func= PSNR( reduction= reduction).cuda()
        psnr_list= []
        # Log observation PSNR
        if noise_level: psnr_noisy_list=[]
    if compute_ssim:
        ssim_func= SSIM( reduction= reduction).cuda()
        ssim_list= []
        # Log observation SSIM
        if noise_level: ssim_noisy_list=[]

    # Iterate
    for i, ( inp, fname) in iterator:
        inp= inp.cuda()

        # Add AWGN and clamp
        if noise_level:
            inp02= inp.clone()+ noise_level* ch.randn_like( inp).cuda()
            inp02= ch.clamp( inp02, 0, 1)
        else: inp02= inp.clone().cuda()

        if pad_input:
            # Pad input (x224)
            inp02, r, c= AlexNet_pad( inp02)

        if pad_edges:
            # Reduce border artifacts
            inp02= F.pad( input= inp02,
                          pad= ( 224, 224, 224, 224),
                          mode= "reflect")

        # Predict
        output= model( x= inp02, feature_loss= 0)
        if norm_flag: output= denormalize( output)

        if pad_edges:
            # Remove padding
            output= output[ :, :, 224: -224, 224: -224]
            inp02= inp02[ :, :, 224: -224, 224: -224]

        if pad_input:
            # Restore original size
            output= output[ :, :, :r, :c]
            inp02= inp02[ :, :, :r, :c]

        # Output transformation
        if transform_output== "resize":
            output= F.interpolate( input= output,
                                   size= transform_output_dim,
                                   mode= "bilinear")
            inp02= F.interpolate( input= inp02,
                                  size= transform_output_dim,
                                  mode= "bilinear")
        elif transform_output== "skip": pass # leave unaltered
        else: raise ValueError( "Undefined output transform. Check\
                                'transform_output' input argument.")

        # Compute metrics
        compute_metrics( i= i,
                         inp= inp,
                         inp02= inp02,
                         output= output,
                         fname= fname,
                         noise_level= noise_level,
                         compute_lpips= compute_lpips,
                         compute_psnr= compute_psnr,
                         compute_ssim= compute_ssim,
                         lpips_list= lpips_list,
                         psnr_list= psnr_list,
                         ssim_list= ssim_list,
                         lpips_noisy_list= lpips_noisy_list,
                         psnr_noisy_list= psnr_noisy_list,
                         ssim_noisy_list= ssim_noisy_list,
                         sample_list= sample_list,
                         batch_list= batch_list,
                         fname_list= fname_list,
                         lpips_func= lpips_func,
                         psnr_func= psnr_func,
                         ssim_func= ssim_func)
        
        # Export predictions
        if infer_export:
            export_predict( i= i,
                            inp= inp,
                            inp02= inp02,
                            output= output,
                            noise_level= noise_level,
                            epoch= epoch,
                            out_folder= out_folder)

    # Export metrics
    export_metrics( compute_lpips= compute_lpips,
                    compute_psnr= compute_psnr,
                    compute_ssim= compute_ssim,
                    noise_level= noise_level,
                    batch_list= batch_list,
                    sample_list= sample_list,
                    fname_list= fname_list,
                    lpips_list= lpips_list,
                    lpips_noisy_list= lpips_noisy_list,
                    ssim_list= ssim_list,
                    ssim_noisy_list= ssim_noisy_list,
                    psnr_list= psnr_list,
                    psnr_noisy_list= psnr_noisy_list,
                    out_folder= out_folder)
    return


# Compute metrics
def compute_metrics( inp, inp02, output,
                     i, fname, compute_lpips,
                     noise_level, compute_psnr, compute_ssim,
                     lpips_list, psnr_list, ssim_list,
                     lpips_noisy_list, psnr_noisy_list, ssim_noisy_list,
                     sample_list, batch_list, fname_list,
                     lpips_func, psnr_func, ssim_func):
    if compute_lpips:
        lpips_val= lpips_func( inp, output)
        lpips_list+= lpips_val.flatten().tolist()
        if noise_level:
            lpips_val= lpips_func( inp, inp02)
            lpips_noisy_list+= lpips_val.flatten().tolist()
    if compute_psnr:
        psnr_val= psnr_func( inp, output)
        psnr_list+= psnr_val.flatten().tolist()
        if noise_level:
            psnr_val= psnr_func( inp, inp02)
            psnr_noisy_list+= psnr_val.flatten().tolist()
    if compute_ssim:
        ssim_val= ssim_func( inp, output)
        ssim_list+= ssim_val.flatten().tolist()
        if noise_level:
            ssim_val= ssim_func( inp, inp02)
            ssim_noisy_list+= ssim_val.flatten().tolist()
    if compute_lpips or compute_psnr or compute_lpips:
        sample_list+= list( range( inp.shape[ 0]))
        batch_list+= [ i]* inp.shape[ 0]
        fname_list+= fname
    return


# Export predictions
def export_predict( i, inp, inp02,
                    output, noise_level,
                    epoch, out_folder):
    for j in range( inp.shape[ 0]):
        input_pil= transforms.ToPILImage()( inp.detach().cpu()[ j, :, :, :])
        output_pil= transforms.ToPILImage()( output.detach().cpu()[ j, :, :, :])
        if noise_level:
            noisy_pil= transforms.ToPILImage()( inp02.detach().cpu()[ j, :, :, :])
        elif noise_level:
            noisy_pil= transforms.ToPILImage()( inp02.detach().cpu()[ j, :, :, :])
  
        if epoch is not None:
            input_str= os.path.join( out_folder,
                                     "img_" + str( i) + "_" + str( j) + "_gtruth.png")
            output_str= os.path.join( out_folder,
                                      "epoch_" + str( epoch) + "_img_" + str( i) + "_" + str( j) + ".png")
            input_pil.save( input_str, "PNG") # do not generate gtruth at each epoch
            output_pil.save( output_str, "PNG")
            if noise_level:
                noisy_str= os.path.join( out_folder, "epoch_" + str( epoch) + "_img_" + str( i) + "_" + str( j) + "_noisy.png")
                noisy_pil.save( noisy_str, "PNG")
        else:
            input_str= os.path.join( out_folder, "img_" + str( i) + "_" + str( j) + "_gtruth.png")
            output_str= os.path.join( out_folder, "img_" + str( i) + "_" + str( j) + ".png")
            input_pil.save( input_str, "PNG") # do not generate gtruth at each epoch
            output_pil.save( output_str, "PNG")
            if noise_level:
                noisy_str= os.path.join( out_folder, "img_" + str( i) + "_" + str( j) + "_noisy.png")
                noisy_pil.save( noisy_str, "PNG")
    return


# Export metrics
def export_metrics( compute_lpips, compute_psnr, compute_ssim,
                    noise_level, batch_list,
                    sample_list, fname_list, lpips_list,
                    lpips_noisy_list, ssim_list, ssim_noisy_list,
                    psnr_list, psnr_noisy_list, out_folder):
    if compute_lpips:
        if noise_level:
            # Attach observation LPIPS
            lpips_data= np.column_stack( ( batch_list, sample_list, fname_list, lpips_list, lpips_noisy_list))
            _lpips_noisy_list= [ k for k in lpips_noisy_list if not( np.isnan( k))]
            print( "Average Observation LPIPS (discarding NaNs): ", sum( _lpips_noisy_list)/ len( _lpips_noisy_list))
        else:
            lpips_data= np.column_stack( ( batch_list, sample_list, fname_list, lpips_list))
        np.savetxt( os.path.join( out_folder, "lpips.csv"), lpips_data, delimiter=",", fmt= '% s')
        _lpips_list= [ k for k in lpips_list if not( np.isnan( k))]
        print( "Average Reconstruction LPIPS (discarding NaNs): ", sum( _lpips_list)/ len( _lpips_list))
    if compute_ssim:
        if noise_level:
            ssim_data= np.column_stack( ( batch_list, sample_list, fname_list, ssim_list, ssim_noisy_list))
            print( "Average Observation SSIM: ", sum( ssim_noisy_list)/ len( ssim_noisy_list))
        else:
            ssim_data= np.column_stack( ( batch_list, sample_list, fname_list, ssim_list))
        np.savetxt( os.path.join( out_folder, "ssim.csv"), ssim_data, delimiter=",", fmt= '% s')
        print( "Average Reconstruction SSIM: ", sum( ssim_list)/ len( ssim_list))
    if compute_psnr:
        if noise_level:
            psnr_data= np.column_stack( ( batch_list, sample_list, fname_list, psnr_list, psnr_noisy_list))
            print( "Average Observation PSNR: ", sum( psnr_noisy_list)/ len( psnr_noisy_list))
        else:
            psnr_data= np.column_stack( ( batch_list, sample_list, fname_list, psnr_list))
        np.savetxt( os.path.join( out_folder, "psnr.csv"), psnr_data, delimiter=",", fmt= '% s')
        print( "Average Reconstruction PSNR: ", sum( psnr_list)/ len( psnr_list))
    return
