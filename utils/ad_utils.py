import os
import torch as ch
import numpy as np
from tqdm import tqdm
from tqdm import trange
import torch.optim as optim
from torchvision.utils import save_image
from torch.autograd import Variable


# One vs. All anomaly detection
def one_vs_all( in_test_loader, act_reference, comparator_layer,
                comparator, feat_crit,
                pixel_crit, variable, model, optimizer,
                step_size, sched_step, sched_gamma,
                iterations, out_folder, feature_loss_weight,
                pixel_loss_weight, randn_init, export_output,
                batch_limit):
    # Create output lists
    in_sample_list= []
    in_label_list= []
    in_pixel_loss_list= []
    in_feat_loss_list= []
    
    iterator= tqdm( enumerate( in_test_loader),
                    total= len( in_test_loader))
    for i, ( i01, l01) in iterator:
        # Run only first 'batch_limit' batches
        if batch_limit and ( i> batch_limit): break
        else:
            z01, pixel_loss, feat_loss= optimize_loop( target= i01,
                                                       act_reference= act_reference,
                                                       comparator_layer= comparator_layer,
                                                       comparator= comparator,
                                                       feat_crit= feat_crit,
                                                       pixel_crit= pixel_crit,
                                                       variable= variable,
                                                       model= model,
                                                       optimizer= optimizer,
                                                       step_size= step_size,
                                                       sched_step= sched_step,
                                                       sched_gamma= sched_gamma,
                                                       iterations= iterations,
                                                       out_folder= out_folder,
                                                       i= i,
                                                       feature_loss_weight= feature_loss_weight,
                                                       pixel_loss_weight= pixel_loss_weight,
                                                       randn_init= randn_init,
                                                       export_output= export_output)
            in_sample_list+= [ i]
            in_label_list+=[ l01.item()]
            in_pixel_loss_list+= [ pixel_loss.detach().cpu().numpy()]
            in_feat_loss_list+= [ feat_loss.detach().cpu().numpy()]
    
    # Export metrics
    in_data= np.column_stack( ( in_sample_list, in_label_list, in_pixel_loss_list, in_feat_loss_list))
    np.savetxt( os.path.join( out_folder, "inlier_data.csv"), in_data, delimiter=",", fmt= '% s')
    
    print( "Average inlier pixel loss: ", sum( in_pixel_loss_list)/ len( in_pixel_loss_list))
    print( "Average inlier feature loss: ", sum( in_feat_loss_list)/ len( in_feat_loss_list))
    return


# Optimize loop
def _optimize_loop( iterations, variable, model,
                    feat, comparator,
                    feat_crit, target_act, feature_loss_weight,
                    pixel_loss_weight, lowres, 
                    opt, optimizer, scheduler,
                    i, pixel_crit, out_folder,
                    export_output= False):
    iterator= trange( iterations)
    for j in iterator:
        inp= model.denormalize( model.decoder_pass( feat))
        inp_clamp= ch.clamp( inp, 0, 1)
        act= comparator( inp_clamp)

        # Compute loss
        feat_loss= feature_loss_weight* feat_crit( act, target_act)
        pixel_loss= pixel_loss_weight* pixel_crit( inp_clamp, lowres)
        loss= feat_loss+ pixel_loss 

        # Update latent
        opt.zero_grad()
        loss.backward()
        opt.step()
        if optimizer== "sgd_step" or optimizer== "sgd_lambda":
            scheduler.step()
 
        # Update description
        desc=( 'Total loss: {loss:.8f} | Px. loss: {pixel_loss:.8f} |'
               ' Feat. loss: {feat_loss:.8f} | lr: {lr:.8f} ||'.format( loss= loss,
               pixel_loss= pixel_loss, feat_loss= feat_loss,
               lr= opt.param_groups[0]['lr']))
        iterator.set_description( desc)
 
    # Export maximizer
    inp_clamp= ch.clamp( model.denormalize( model.decoder_pass( feat)), 0, 1)
    if export_output:
        save_image( inp_clamp.detach().cpu(),
                    os.path.join( out_folder, str( i) + "_est_image.png"))
    return feat, pixel_loss, feat_loss


# Optimize latent
def optimize_loop( target, act_reference, comparator_layer,
                   comparator, feat_crit,
                   variable, model, optimizer,
                   step_size, sched_step, sched_gamma,
                   iterations, out_folder, i,
                   feature_loss_weight, pixel_loss_weight,
                   pixel_crit, randn_init= False, export_output= False,
                   tv_crit= None, tv_loss= False, tv_lambdar= None):
    target= target.cuda()
    lowres= target.clone()
    target_act= comparator( target)

    lowres_feat= model.encoder_pass( lowres, feature_loss= False)
    if randn_init:
        lowres_feat= ch.randn_like( lowres_feat).cuda()
    feat= Variable( lowres_feat.cuda(), requires_grad= True)
    inp= model.denormalize( model.decoder_pass( feat)) # Initial guess

    # Set optimizer and scheduler
    if optimizer== "adam":
        opt= optim.Adam( [ feat], lr= step_size, betas= ( 0.5, 0.999)) # set optimizer
    elif optimizer== "sgd_step":
        opt= optim.SGD( [ feat], lr= step_size, momentum= 0.9)
        scheduler= optim.lr_scheduler.StepLR( opt,
                                              step_size= sched_step,
                                              gamma= sched_gamma)
    elif optimizer== "sgd_lambda":
        opt= optim.SGD( [ feat], lr= step_size, momentum= 0.9)
        lambda1= lambda epoch: 1- 0.95* epoch/ iterations # lr linearly changes from 1 to 0.01
        scheduler= optim.lr_scheduler.LambdaLR( opt, lr_lambda= lambda1)
    else: raise ValueError( "Undefined optimizer. Check 'optimizer' input argument.")
    inp_clamp= ch.clamp( inp, 0, 1) # Force image to be between [ 0, 1].

    if export_output:
        save_image( target.detach().cpu(), os.path.join( out_folder, str( i) + "_target_image.png")) # Export target resolution image.
        save_image( lowres.detach().cpu(), os.path.join( out_folder, str( i) + "_init_image.png")) # Export initial guess.

    # Optimize loop
    out_feat, pixel_loss, feat_loss= _optimize_loop( iterations= iterations,
                                                     variable= variable,
                                                     model= model,
                                                     feat= feat,
                                                     comparator= comparator,
                                                     feat_crit= feat_crit,
                                                     target_act= target_act,
                                                     feature_loss_weight= feature_loss_weight,
                                                     pixel_loss_weight= pixel_loss_weight,
                                                     lowres= lowres,
                                                     pixel_crit= pixel_crit,
                                                     opt= opt,
                                                     optimizer= optimizer,
                                                     scheduler= scheduler,
                                                     i= i,
                                                     out_folder= out_folder,
                                                     export_output= export_output)
    return out_feat, pixel_loss, feat_loss

