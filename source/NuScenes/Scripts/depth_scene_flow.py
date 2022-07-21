import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import skimage.io as sio
from PIL import Image
from PIL import ImageFile

from extractor import DepthEncoder, BasicEncoder, SmallEncoder
from update import BasicUpdateBlock, SmallUpdateBlock
from corr import CorrBlock, AlternateCorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8


try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

class Depth_Scene_Flow(nn.Module):
    def __init__(self, args):
        super(Depth_Scene_Flow, self).__init__()
        self.args = args

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
        
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)        
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            self.dnet = DepthEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
            self.dnet = DepthEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
    
    def forward(self, depthmap, image1, image2):
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2* (image2 / 255.0) - 1.0
        
        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])        
        
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            dnet = self.dnet(depthmap)
            combined_features = torch.stack(cnet, dnet, dim=1) #Depth + RGB Context features

        #GRU Based Update to scene flow and depth 
        #Not able to code this


    