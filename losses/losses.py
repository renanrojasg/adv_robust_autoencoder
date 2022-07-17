import numpy as np
import torch as ch
import torch.nn as nn
import torch.nn.functional as F


# Gram loss
class GramLoss(nn.Module):
    def __init__( self, reduction= "sum"):
        super( GramLoss, self).__init__()
        self.criterion= nn.MSELoss( reduction= reduction)
    def forward( self, input, target):
        ib,ic,ih,iw = input.size()
        iF = input.view(ib,ic,-1)
        iMean = ch.mean(iF,dim=2)
        iCov = GramMatrix()(input)

        tb,tc,th,tw = target.size()
        tF = target.view(tb,tc,-1)
        tMean = ch.mean(tF,dim=2)
        tCov = GramMatrix()(target)

        loss= self.criterion( iMean, tMean) + self.criterion( iCov, tCov)
        return loss


# Gram matrix
class GramMatrix(nn.Module):
    def forward(self,input):
        b, c, h, w = input.size()
        f = input.view(b,c,h*w) # bxcx(hxw)
        G = ch.bmm(f,f.transpose(1,2)) # f: bxcx(hxw), f.transpose: bx(hxw)xc -> bxcxc
        return G.div_(c*h*w)


# Stylization loss criterion
class stylization_criterion( nn.Module):
    def __init__( self, loss_style_layers, loss_style_weight,
                  reduction= "mean"):
        super( stylization_criterion, self).__init__()
        self.loss_style_layers= loss_style_layers
        self.loss_style_weight= loss_style_weight
        self.style_loss= GramLoss( reduction= reduction) # Gram loss

    def forward( self, style_features, stylized_style_features):
        """
        Input.
            - Style_feat: Style dictionary containing feature maps of different dimensions.
            - Stylized_feat: Stylized dictionary containing feature maps of different dimensions.
        """
        style_loss= 0
        for key in self.loss_style_layers:
            style_loss+= self.style_loss( style_features[ key].detach(), stylized_style_features[ key])
        style_loss= self.loss_style_weight* style_loss
        return style_loss

