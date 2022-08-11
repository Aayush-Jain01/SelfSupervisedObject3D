import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import skimage.io as sio
from PIL import Image
from PIL import ImageFile

import network_run
from torch.utils.data.dataset import Dataset

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
def parse_args():
    parser = argparse.ArgumentParser(description='MARS CNN Script')
    parser.add_argument('--checkpoint', action='append',
                        help='Location of the checkpoints to evaluate.')
    parser.add_argument('--train', type=int, default=1,
                        help='If set to nonzero train the network, otherwise will evaluate.')
    parser.add_argument('--save', type=str, default='',
                        help='The path to save the network checkpoints and logs.')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--root', type=str, default='/mars/mnt/dgx/FrameNet')
    parser.add_argument('--epoch', type=int, default=0,
                        help='The epoch to resume training from.')
    parser.add_argument('--iter', type=int, default=0,
                        help='The iteration to resume training from.')
    parser.add_argument('--dataset_pickle_file', type=str, default='./data/scannet_depth_completion_split.pkl')
    parser.add_argument('--dataloader_test_workers', type=int, default=16)
    parser.add_argument('--dataloader_train_workers', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1.e-4)
    parser.add_argument('--save_every_n_iteration', type=int, default=1000,
                        help='Save a checkpoint every n iterations (iterations reset on new epoch).')
    parser.add_argument('--save_every_n_epoch', type=int, default=1,
                        help='Save a checkpoint on the first iteration of every n epochs (independent of iteration).')
    parser.add_argument('--enable_multi_gpu', type=int, default=0,
                        help='If nonzero, use all available GPUs.')
    parser.add_argument('--skip_every_n_image_test', type=int, default=40,
                        help='Skip every n image in the test split.')
    parser.add_argument('--skip_every_n_image_train', type=int, default=1,
                        help='Skip every n image in the test split.')
    parser.add_argument('--eval_test_every_n_iterations', type=int, default=1000,
                        help='Evaluate the network on the test set every n iterations when in training.')
    parser.add_argument('--dataset_type', type=str, default='scannet',
                        help='The dataset loader fromat. Closely related to the pickle file (scannet, nyu, azure).')
    parser.add_argument('--max_epochs', type=int, default=10000,
                        help='Maximum number of epochs for training.')
    parser.add_argument('--depth_loss', type=str, default='L1',
                        help='Depth loss function: L1/L2')

    parser.add_argument('--window', type=int, required=True, nargs='+')
    parser.add_argument('--save_flow', type=int, default=0)
    parser.add_argument('--resize', type=int, default=2)
    parser.add_argument('--azure_ba', type=int, default=0)
    parser.add_argument('--flip_flow', type=int, default=1)
    # RAFT
    parser.add_argument('--raft_model', type=str, default='')
    parser.add_argument('--wdecay', type=float, default=0.0001)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

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
    
    def _on_eval_mode(self):
        self.test_mode = True

    def _on_train_mode(self):
        self.test_mode = False

    def initialize_flow(self, image1, image2): #Get the calculated flow
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1
    
    def projection_to_flow(self, ): #Write code for function where you project depth features + RGB features to flow i.e the P function

    def forward(self, depthmap, image1, image2, flow_init):
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
            net, inp = torch.split(combined_features, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)
            
        #Get flow
        coords0, coords1 = self.initialize_flow(image1, image2)
        if flow_init is not None: #Here we will input the pre-computed flow in forward and this will get added instead of initialization being 0
            coords1 = coords1 + flow_init

        #GRU Based Update to scene flow and depth 
        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t).
            #Instead of this we will use projected delta_flow as we have depth features and not flow features like before
            coords1 = coords1 + projection_to_flow(delta_flow)

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up
            
        return flow_predictions