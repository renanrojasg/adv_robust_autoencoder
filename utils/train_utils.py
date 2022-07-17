import os
import sys
import dill 
import time
import warnings
import torch as ch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image, ImageOps
from torch.optim import SGD, lr_scheduler
from torchvision.utils import make_grid
from cox.utils import Parameters
from tqdm import tqdm, trange
from models.alexnet import AlexNet_pad
from piqa import LPIPS, PSNR, SSIM
from utils.core_utils import export_output, is_image_file
from utils.infer_utils import _infer_loop
from utils.tb_utils import AverageMeter


# Save checkpoint
def save_checkpoint( sd_info, fname, out_dir):
    cp_save_path = os.path.join( out_dir, fname)
    ch.save( sd_info, cp_save_path, pickle_module= dill)
    return


def load_opt( load_generator, g_opt, g_sched,
              discriminator, d_opt, d_sched):
    cp= ch.load( load_generator,
                 pickle_module= dill)
    # Load optimizer
    g_opt.load_state_dict( cp[ 'optimizer'])
    g_opt.param_groups[ 0][ 'params']= model.parameters()

    # Load and update scheduler
    if schedule:
        g_sched.load_state_dict( cp[ 'schedule'])
        g_sched.step()

    if discriminator:
        d_opt.load_state_dict( checkpoint[ 'opt_discriminator'])
        if d_sched:
            d_sched.load_state_dict( checkpoint[ 'schedule_discriminator'])
            d_sched.step()
    return


# Train model
def train_model( cp_step, transform_test, transform_output,
                 transform_output_dim, transform_final_dim, epochs,
                 disc_labels, disc_normalize, disc_loss_weight,
                 pix_loss_weight, feat_loss_weight, adv_loss_weight,
                 infer_export, transform_train, noise_level,
                 pix_loss, feat_loss, adv_loss,
                 reduction, model, discriminator,
                 denormalize, train_loader, val_loader,
                 norm_flag, out_dir, prev_dir,
                 g_opt, g_sched, d_opt,
                 d_sched, load_generator, preview,
                 val_step, comparator= None, checkpoint= None,
                 preview_loader= None):


    # Load optimizer if resuming
    if load_generator is not None:
        load_opt( load_generator= load_generator,
                  g_opt= g_opt,
                  g_sched= g_sched,
                  discriminator= discriminator,
                  d_opt= d_opt,
                  d_sched= d_sched)


    # Set training criteria
    pix_c, feat_c, adv_c= set_criteria( pix_loss= pix_loss,
                                        feat_loss= feat_loss,
                                        adv_loss= adv_loss,
                                        reduction= reduction)


    best_tot_l, start_epoch= (0, 0)
    if checkpoint: start_epoch= checkpoint[ 'epoch']


    # Train loop
    for epoch in range( start_epoch, epochs):
        last_epoch= ( epoch== ( epochs - 1))
        save_cp= ( epoch % cp_step== 0)
        val_cp= ( epoch!= 0 and epoch % val_step== 0)

        # Training
        tot_l, disc_l, pix_l, feat_l, gadv_l= _model_loop( noise_level= noise_level,
                                                           loop_type= 'train',
                                                           loader= train_loader, 
                                                           model= model,
                                                           discriminator= discriminator,
                                                           comparator= comparator,
                                                           denormalize= denormalize,
                                                           norm_flag= norm_flag,
                                                           g_opt= g_opt,
                                                           d_opt= d_opt,
                                                           epoch= epoch,
                                                           disc_labels= disc_labels,
                                                           disc_normalize= disc_normalize,
                                                           pix_c= pix_c,
                                                           feat_c= feat_c,
                                                           adv_c= adv_c,
                                                           pix_loss= pix_loss,
                                                           feat_loss= feat_loss,
                                                           adv_loss= adv_loss,
                                                           pix_loss_weight= pix_loss_weight,
                                                           feat_loss_weight= feat_loss_weight,
                                                           adv_loss_weight= adv_loss_weight,
                                                           disc_loss_weight= disc_loss_weight)


        # Validation
        if val_cp:
            with ch.no_grad():
                tot_l, disc_l, pix_l, feat_l, gadv_l= _model_loop( noise_level= noise_level,
                                                                   adv_loss= adv_loss,
                                                                   feat_loss= feat_loss,
                                                                   pix_loss= pix_loss,
                                                                   loop_type= 'eval',
                                                                   loader= val_loader,
                                                                   model= model,
                                                                   discriminator= discriminator,
                                                                   comparator= comparator,
                                                                   denormalize= denormalize,
                                                                   g_opt= None,
                                                                   d_opt= None,
                                                                   epoch= epoch,
                                                                   norm_flag= norm_flag,
                                                                   disc_labels= disc_labels,
                                                                   disc_normalize= disc_normalize,
                                                                   disc_loss_weight= disc_loss_weight,
                                                                   pix_loss_weight= pix_loss_weight,
                                                                   feat_loss_weight= feat_loss_weight,
                                                                   adv_loss_weight= adv_loss_weight,
                                                                   pix_c= pix_c,
                                                                   feat_c= feat_c,
                                                                   adv_c= adv_c)

            # Preview
            if preview:
                _infer_loop( transform_output= transform_output,
                             transform_output_dim= transform_output_dim,
                             reduction= reduction,
                             noise_level= noise_level,
                             model= model,
                             denormalize= denormalize,
                             val_loader= preview_loader,
                             out_folder= prev_dir,
                             epoch= epoch,
                             norm_flag= norm_flag,
                             infer_export= infer_export)

        # Log and get best
        if last_epoch or save_cp:
            # Build checkpoint data
            sd_info = { 'model': model.state_dict(),
                        'optimizer': g_opt.state_dict(),
                        'discriminator': discriminator.state_dict() if discriminator else None,
                        'opt_discriminator': d_opt.state_dict() if discriminator else None,
                        'schedule':( g_sched and g_sched.state_dict()),
                        'schedule_discriminator':( d_sched and d_sched.state_dict())\
                                                   if discriminator else None,
                        'epoch': epoch+ 1}

            # Save checkpoints
            save_checkpoint( sd_info= sd_info,
                             #fname= "checkpoint.pt",
                             fname= "checkpoint_" + str(epoch) + ".pt",
                             out_dir= out_dir)
            is_best= tot_l> best_tot_l
            if is_best: save_checkpoint( sd_info= sd_info,
                                         fname= "checkpoint_best.pt",
                                         out_dir= out_dir)

        # Update schedulers
        if g_sched: g_sched.step()
        if d_sched: d_sched.step()
    return


# Generator training loop
def _model_loop( disc_labels, disc_normalize, disc_loss_weight,
                 noise_level, pix_loss,
                 feat_loss, adv_loss, pix_loss_weight,
                 feat_loss_weight, adv_loss_weight,
                 loop_type, loader,
                 model, discriminator, denormalize, 
                 g_opt, d_opt, epoch,
                 pix_c, feat_c, adv_c,
                 norm_flag, g_schedule= None,
                 comparator= None, d_schedule= None):
    # Store accuracy
    tot_l= AverageMeter()
    disc_l= AverageMeter()
    pix_l= AverageMeter()
    feat_l= AverageMeter()
    gadv_l= AverageMeter()

    # Set train/eval
    if loop_type== "train":
        model= model.train()
        if adv_loss: disciminator= discriminator.train()
    elif loop_type== "eval":
        model= model.eval()
        if adv_loss: disciminator= discriminator.eval()
    else: raise ValueError( "Undefined loop type. Check 'loop_type' argument.")

    # Iterator
    iterator= tqdm( enumerate( loader), total= len( loader))
    for i, ( inp, _) in iterator:
        target= inp.clone().cuda( non_blocking= True).detach() # target: real image
        
        # Add noise
        if noise_level:
            inp= ( inp+ noise_level* ch.randn( inp.shape)).cuda()
            inp= ch.clamp( inp, 0, 1)

        # Prediction
        if feat_loss is not None:
            output= model( inp, feature_loss= False)

            # Target feature
            if comparator:
                # External comparator
                feat_target= comparator( target) 
            else:
                # Use encoder as comparator
                feat_target, _= model( target, feature_loss= True)
            feat_target= feat_target.detach()
        else:
            output= model( inp, feature_loss= False)
            feat_target= None

        # Denormalize and clamp
        if norm_flag:
            output= denormalize( output)
            ch.clamp( output, 0, 1) 


        if adv_loss:
            if disc_normalize: _normalizer= model.module.normalizer
            else: _normalizer= None

            # Discriminator loss
            disc_adv_loss= compute_disc_loss( output= output,
                                              target= target,
                                              feat_target= feat_target,
                                              comparator= comparator,
                                              model= model,
                                              discriminator= discriminator,
                                              feat_c= feat_c,
                                              adv_c= adv_c,
                                              normalizer= _normalizer,
                                              disc_labels= disc_labels,
                                              disc_loss_weight= disc_loss_weight)

            # Update discriminator
            if loop_type== "train":
                d_opt.zero_grad()
                disc_adv_loss.backward()
                d_opt.step()
        else:
            _normalizer= None
            disc_adv_loss= 0


        # Generator loss
        pix_loss, feat_loss, gen_adv_loss= func_compute_gen_loss( output= output,
                                                                  target= target,
                                                                  feat_target= feat_target,
                                                                  comparator= comparator,
                                                                  model= model,
                                                                  discriminator= discriminator,
                                                                  pix_c= pix_c,
                                                                  feat_c= feat_c,
                                                                  adv_c= adv_c,
                                                                  disc_labels= disc_labels,
                                                                  normalizer= _normalizer,
                                                                  pix_loss_weight= pix_loss_weight,
                                                                  feat_loss_weight= feat_loss_weight,
                                                                  adv_loss_weight= adv_loss_weight)

        # Full loss
        loss= pix_loss+ feat_loss+ gen_adv_loss

        # Update generator
        if loop_type== "train":
            g_opt.zero_grad()
            loss.backward()
            g_opt.step()

        # Tensorboard logger
        b= inp.size( 0)
        tot_l.update( loss.data, b)
        pix_l.update( pix_loss.item(), b)
        if feat_c:
            feat_l.update( feat_loss.item(), b)
        if adv_c:
            disc_l.update( disc_adv_loss.item(), b)
            gadv_l.update( gen_adv_loss.item(), b)

        # Update description
        func_update_description( loop_type= loop_type,
                                 g_opt= g_opt,
                                 d_opt= d_opt,
                                 adv_c= adv_c,
                                 feat_c= feat_c,
                                 epoch= epoch,
                                 losses= tot_l.avg,
                                 pix_loss= pix_l.avg,
				 feat_loss= feat_l.avg,
                                 gen_adv_loss= gadv_l.avg,
                                 disc_adv_loss= disc_l.avg,
                                 iterator= iterator)
    return tot_l.avg, disc_l.avg, pix_l.avg, feat_l.avg, gadv_l.avg


# Set training losses
def set_criteria( pix_loss, feat_loss, adv_loss,
                  reduction):
    if pix_loss is None and feat_loss is None and adv_loss is None:
        raise ValueError( "No loss defined. At least one must be set.")

    # Pixel loss
    if pix_loss== "l2_loss": pix_c= ch.nn.MSELoss( reduction= reduction)
    elif pix_loss== "l1_loss": pix_c= ch.nn.L1Loss( reduction= reduction)
    elif pix_loss is None: pix_c= None
    else: raise ValueError( "Pixel loss: Undefined norm. Check 'pixel_loss' input argument.")

    # Feature loss
    if feat_loss== "l2_loss": feat_c= ch.nn.MSELoss( reduction= reduction)
    elif feat_loss== "l1_loss": feat_c= ch.nn.L1Loss( reduction= reduction)
    elif feature_loss== "None": feat_c= None
    else: raise ValueError( "Feature loss: Undefined norm. Check 'feature_loss' input argument.")

    # Adversarial loss
    if adv_loss: adv_c= ch.nn.BCELoss( reduction= reduction)
    else: adv_c= None

    return pix_c, feat_c, adv_c


# Compute discriminator loss
def compute_disc_loss( output, target, feat_target,
                       comparator, model, discriminator,
                       feat_c, adv_c, disc_labels,
                       normalizer, disc_loss_weight):
    if feat_target is not None:
        feat_target= feat_target.view( feat_target.size( 0),
                                       feat_target.size( 1)*\
                                       feat_target.size( 2)*\
                                       feat_target.size( 3))

    # Detach output, no backprop to generator
    data_fake= output.clone().detach()

    _disc_adv_loss= func_disc_adv_loss( model= discriminator,
                                        normalizer= normalizer,
                                        criterion= adv_c,
                                        data_real= target,
                                        data_fake= data_fake,
                                        gtruth_fake= disc_labels[ 0],
                                        gtruth_real= disc_labels[ 1],
                                        feat_target= feat_target)
    disc_adv_loss= disc_loss_weight* _disc_adv_loss
    return disc_adv_loss


# Compute generator loss
def func_compute_gen_loss( output, target, feat_target,
                           comparator, model, discriminator,
                           pix_c, feat_c, adv_c,
                           disc_labels, pix_loss_weight, feat_loss_weight,
                           adv_loss_weight, normalizer):
    pixel_loss, feat_loss, gen_adv_loss= 0, 0, 0

    # Pixel loss, detached target
    if pix_c:
        pix_loss= pix_loss_weight* pix_c( output, target)

    # Feature loss
    if feat_c:
        if comparator: feat= comparator( output)
        else: feat, _= model( output, feature_loss= True)
        feat_loss= feat_loss_weight* feat_c( feat, feat_target)

    # Adversarial loss
    if adv_c:
        if feat_target is not None:
            feat_target= feat_target.view( feat_target.size( 0),
                                           feat_target.size( 1)*\
                                           feat_target.size( 2)*\
                                           feat_target.size( 3))

        _gen_adv_loss= func_gen_adv_loss( model= discriminator,
                                          normalizer= normalizer,
                                          criterion= adv_c,
                                          data_fake= output,
                                          gtruth_real= disc_labels[ 1],
                                          feat_target= feat_target)
        gen_adv_loss= adv_loss_weight* _gen_adv_loss 
    return pix_loss, feat_loss, gen_adv_loss


# Update tqdm description
def func_update_description( loop_type, g_opt, d_opt,
                             adv_c, feat_c,
                             epoch, losses, pix_loss,
                             feat_loss, gen_adv_loss, disc_adv_loss,
                             iterator):
    if loop_type== "train":
        lr_current= g_opt.param_groups[0]['lr']
        if adv_c and feat_c:
            lr_disc_current= d_opt.param_groups[0]['lr']
            desc= ('{loop_msg} E: {epoch} | G. Loss: {loss:.2f} | Pix. Loss: {pixel_loss:.2f} | '
                    'Feat. Loss: {feat_loss:.2f} | Gadv. Loss: {gen_adv_loss:.2f} | '
                    'D. Loss {disc_adv_loss:.2f} ||'.format( loop_msg= loop_type,
                    epoch= epoch, loss= losses, pixel_loss= pix_loss, feat_loss= feat_loss,
                    gen_adv_loss= gen_adv_loss, disc_adv_loss= disc_adv_loss))
        elif adv_c:
            lr_disc_current= d_opt.param_groups[0]['lr']
            desc= ('{loop_msg} E: {epoch} | G. Loss: {loss:.2f} | Pix. Loss: {pixel_loss:.2f} | '
                    'Gadv. Loss: {gen_adv_loss:.2f} | D. Loss {disc_adv_loss:.2f} ||'.format( loop_msg= loop_type,
                    epoch= epoch, loss= losses, pix_loss= pixel_loss, gen_adv_loss= gen_adv_loss,
                    disc_adv_loss= disc_adv_loss))
        elif feat_c:
            desc= ('{loop_msg} E: {epoch} | G. Loss: {loss:.4f} | Pix. Loss: {pixel_loss:.4f} | '
                   'Feat. Loss: {feat_loss:.4f} | lr: {lr_current} ||'.format( loop_msg= loop_type,
                   epoch= epoch, loss= losses, pixel_loss= pix_loss, feat_loss= feat_loss,
                   lr_current= lr_current))
        else:
            desc= ('{loop_msg} E: {epoch} | G. Loss: {loss:.2f} | Pix. Loss: {pixel_loss:.2f} | '
                    'lr: {lr_current}||'.format( loop_msg= loop_type, epoch= epoch,
                    loss= losses, pixel_loss= pix_loss, lr_current= lr_current))
    else:
        desc= ( '{loop_msg} E: {epoch} | Loss: {loss:.8f} ||'
                 .format( loop_msg= loop_type, epoch= epoch, loss=losses))
    iterator.set_description(desc)
    iterator.refresh()
    return


# Discriminator loss
def func_disc_adv_loss( model, criterion, data_real,
                        data_fake, gtruth_fake, gtruth_real,
                        feat_target, normalizer):
    real_label= gtruth_real* ch.ones( data_real.size( 0)).cuda()
    fake_label= gtruth_fake* ch.ones( data_real.size( 0)).cuda()

    # Prob. of real images being real.
    if normalizer is not None:
        prob_real= model( input= normalizer( data_real), feat_target= feat_target).view(-1)
    else:
        prob_real= model( input= data_real, feat_target= feat_target).view(-1)

    # Guide disc. to classify real images as real.
    loss_real= criterion( prob_real, real_label)

    # Prob. of fake images being real.
    if normalizer:
        prob_fake= model( input= normalizer( data_fake), feat_target= feat_target).view(-1)
    else:
        prob_fake= model( input= data_fake, feat_target= feat_target).view(-1)

    # Guide disc. to classify fake images as fake
    loss_fake= criterion( prob_fake, fake_label) 

    loss= loss_real+ loss_fake
    return loss
 

# Generator loss
def func_gen_adv_loss( model, criterion, data_fake, gtruth_real, feat_target= None, normalizer= None):
    # 'Real' labels
    real_label= gtruth_real* ch.ones( data_fake.size( 0)).cuda()
    if normalizer:
        # Prob. of generated being fake. Normalize input first.
        prob= model( input= normalizer( data_fake), feat_target= feat_target).view( -1)
    else:
        prob= model( input= data_fake, feat_target= feat_target).view( -1) 

    # Guide the generator to create images that look real
    loss= criterion( prob, real_label) 
    return loss

