import sys
sys.path.insert(1, '../../../datasets/Kitti/Tracking/utils/Scripts')

from utils import readLabelFiles

import argparse
import math
import numpy as np
import os
import random
import time
import torch

from PIL import Image
from PIL.ImageOps import mirror

from torch import nn
from torch.nn.functional import normalize
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine import train_one_epoch, evaluate
import utils
#Let MaskRCNN be the 2D Detector as per the paper

''' ======================================================================== '''
''' ------------------------- Command-line options ------------------------- '''
''' ======================================================================== '''

argparser = argparse.ArgumentParser(description=__doc__)
argparser.add_argument(
    '--batchSize',
    metavar='BATCH',
    type=int,
    default=32,
    help='Mini-batch size.')
argparser.add_argument(
    '--cameraNumber',
    metavar='N',
    type=int,
    default=2,
    help='Number corresponding to Kitti camera.')
argparser.add_argument(
    '--dataRoot',
    metavar='PATH',
    type=str,
    default='../../../datasets/NuScenes',
    help='Path to dataset root.')
argparser.add_argument(
    '--experimentNumber',
    metavar='N',
    type=int,
    default=1,
    help='Number corresponding to experiment.')
argparser.add_argument(
    '--experimentRoot',
    metavar='PATH',
    type=str,
    default='..',
    help='Path to experiment base directory.')
argparser.add_argument(
    '--huberQuadSize',
    metavar='SIZE',
    type=float,
    default=20.0,
    help='Size of quadratic region in Huber loss.')
argparser.add_argument(
    '--labelNumber',
    metavar='NUMBER',
    type=int,
    choices=[2, 3, 4, 5, 6, 7, 8],
    default=7,
    help='2: Gt. detection and angle differences. All cars.\n\
          3: Custom detection and gt. angle differences. All cars.\n\
          4: Gt. detection and angle differences. Stationary cars only.\n\
          5: Gt. detection and custom angle differences. Stationary cars only.\n\
          6: Custom detection and angle differences. Stationary cars only.\n\
          7: Custom detection and angle differences. All cars.\n\
          8: Gt. detection and custom angle differences. All cars.')
argparser.add_argument(
    '--learningRate',
    metavar='LR',
    type=float,
    default=2e-5,
    help='Initial learning rate.')
argparser.add_argument(
    '--loadModelNumber',
    metavar='N',
    type=int,
    default=0,
    help='Model number of model to load (relative load).')
argparser.add_argument(
    '--loadModelPath',
    metavar='PATH',
    type=str,
    default='',
    help='Path of model to load (absolute load).')
argparser.add_argument(
    '--loadModelRendering',
    metavar='MODE',
    type=str,
    choices=['', 'all', 'clone', 'fog', 'morning', 'overcast', 'rain', 'sunset'],
    default='all',
    help='If non-empty, load Virtual Kitti model trained on specified render mode.')
argparser.add_argument(
    '--minBbSize',
    metavar='SIZE',
    type=float,
    default=0.0,
    help='Minimum bounding box size used during training.')
argparser.add_argument(
   '--multiStepLrMilestones',
    metavar='MILESTONES',
    nargs='+',
    type=int,
    default=[7],
    help='Epochs at which learning rate is changed according to scheduler.')
argparser.add_argument(
    '--optimizer',
    metavar='OPTIM',
    type=str,
    choices=['SGD'],
    default='SGD',
    help='Optimizer used during training.')
argparser.add_argument(
    '--optimizerMomentum',
    metavar='MOMENTUM',
    type=float,
    default=0.9,
    help='Momentum factor used by optimizer.')
argparser.add_argument(
    '--pruneThreshold',
    metavar='PRUNE',
    type=float,
    default=1.0,
    help='Threshold determining which entries are pruned from scene.')
argparser.add_argument(
    '--removeThreshold',
    metavar='REMOVE',
    type=float,
    default=1.0,
    help='Threshold determining which entries should be removed.')
argparser.add_argument(
    '--representation',
    metavar='REPR',
    type=str,
    choices=['Single', 'Double'],
    default='Single',
    help='Type of angle representation.')
argparser.add_argument(
    '--resume',
    action='store_true',
    help='Resume training from checkpoint.')
argparser.add_argument(
    '--scheduler',
    metavar='SCHED',
    type=str,
    choices=['MultiStepLR'],
    default='MultiStepLR',
    help='Scheduler type.')
argparser.add_argument(
    '--split',
    metavar='SPLIT',
    type=str,
    choices=['All', 'Boston', 'Singapore'],
    default='All',
    help='Train-validation split.')
argparser.add_argument(
    '--splitPercentage',
    metavar='PERCENT',
    type=float,
    default=0.8,
    help='Percentage of scene belonging to training (All-split case only).')
argparser.add_argument(
    '--stepLrGamma',
    metavar='GAMMA',
    type=float,
    default=0.1,
    help='Step scheduler decay rate.')
argparser.add_argument(
    '--trainingCycles',
    metavar='CYCLES',
    type=int,
    default=1,
    help='Total number of training cycles.')
argparser.add_argument(
    '--trainingEpochs',
    metavar='EPOCHS',
    type=int,
    default=10,
    help='Total number of training epochs.')
argparser.add_argument(
    '--version',
    metavar='VERSION',
    type=str,
    choices=['v1.0-mini', 'v1.0-trainval'],
    default='v1.0-trainval',
    help='Dataset version.')
argparser.add_argument(
    '--virtualModelsRoot',
    metavar='PATH',
    type=str,
    default='../../VirtualKitti/Models',
    help='Path used to retrieve models trained on Virtual Kitti.')
argparser.add_argument(
    '--weightDecay',
    metavar='WEIGHT',
    type=float,
    default=1e-4,
    help='L2 weight decay factor.')
argparser.add_argument(
    '--workers',
    metavar='N',
    type=int,
    default=8,
    help='Number of workers.')

''' ================================================================================ '''
''' ------------------------- Collection of custom classes ------------------------- '''
''' ================================================================================ '''

class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, x):
        return nn.functional.normalize(x)

class SinCosToAngle(nn.Module):
    def __init__(self):
        super(SinCosToAngle, self).__init__()

    def forward(self, x):
        return torch.atan2(x[:,0], x[:,1])


class TrainDataset(Dataset):
    def __init__(self, dictionary, percentage, root, transform):
        super(TrainDataset, self).__init__()

        self.root = root
        self.transform = transform
      
        labelDir = '%s/training/label_0%d' % (root, args.cameraNumber)
        sceneIds = dictionary['indices']
        readMode = dictionary['readMode']
        sceneLabelsDict = readLabelFiles(labelDir, sceneIds, readMode=readMode)

        self.targets = []

        self.toFrameIndexList = []
        self.toNuScenesPathDict = {}
        self.toSequenceIndexList = []

        for sceneIndex, objectLabels in sceneLabelsDict.items():
            kittiPath = '%s/training/image_0%d/%04d.txt' % (self.root, args.cameraNumber, sceneIndex)
            nuScenesPaths = []
            num_objs = 0
            boxes = []
            areas = []
            with open(kittiPath, 'r') as imageFile:
                lines = imageFile.read().splitlines()
                for line in lines:
                    nuScenesPaths.append(line.split(' ')[-1])
                
            self.toNuScenesPathDict[sceneIndex] = nuScenesPaths
            endIndex = int(len(objectLabels)*percentage)
            for objectLabel in objectLabels[:endIndex]:
                if objectLabel['type'] == 'car':
                    left = objectLabel['left']
                    top = objectLabel['top']
                    right = objectLabel['right']
                    bottom = objectLabel['bottom']   

                    occlusionCondition = objectLabel['occluded'] <= args.maxOcclusion
                    sizeCondition = (right-left)*(bottom-top) >= args.minBbSize

                    if occlusionCondition and sizeCondition:
                        num_objs += 1
                        boxes.append((left, top, right, bottom))
                        areas.append(sizeCondition)
                        self.toFrameIndexList.append(objectLabel['frame'])
                        self.toSequenceIndexList.append(sceneIndex)

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # self.targets["boundingBoxes"] = boxes
            labels = torch.ones((num_objs,), dtype=torch.int64)
            # self.targets["labels"] = labels
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
            # self.targets["iscrowd"].append(iscrowd)
            areas = torch.as_tensor(areas)
            # self.targets["area"] = areas
            temp = {}
            temp["boundingBoxes"] = boxes
            temp["labels"] = labels
            temp["iscrowd"] = iscrowd
            temp["area"] = areas
            self.targets.append(temp)
    
    def __getitem__(self, index):
        frameIndex = self.toFrameIndexList[index]
        sceneIndex = self.toSequenceIndexList[index]
        nuScenesPath = self.toNuScenesPathDict[sceneIndex][frameIndex]

        fullImagePath = '%s/%s' % (self.root, nuScenesPath)
        fullImage = Image.open(fullImagePath).convert('RGB')

        boundingBox = self.targets[index]["boundingBoxes"][index]
        bbImage = fullImage.crop(boundingBox)
        target = self.targets[index]
        
        bbImage = self.transform(bbImage)      
        return bbImage, target

    def __len__(self):
        return len(self.toFrameIndexList)

class ValDataset(Dataset):
    def __init__(self, dictionary, percentage, root, transform):
        super(ValDataset, self).__init__()

        self.root = root
        self.transform = transform

        labelDir = '%s/training/label_0%d' % (root, args.cameraNumber)
        sceneIds = dictionary['indices']
        readMode = dictionary['readMode']
        sceneLabelsDict = readLabelFiles(labelDir, sceneIds, readMode=readMode)

        self.targets = []

        self.toFrameIndexList = []
        self.toNuScenesPathDict = {}
        self.toSceneIndexList = []

        for sceneIndex, objectLabels in sceneLabelsDict.items():
            kittiPath = '%s/training/image_0%d/%04d.txt' % (self.root, args.cameraNumber, sceneIndex)
            nuScenesPaths = []
            num_objs = 0
            boxes = []
            areas = []    
            with open(kittiPath, 'r') as imageFile:
                lines = imageFile.read().splitlines()
                for line in lines:
                    nuScenesPaths.append(line.split(' ')[-1])
                
            self.toNuScenesPathDict[sceneIndex] = nuScenesPaths
            beginIndex = int(len(objectLabels)*(1.0-percentage))

            for objectLabel in objectLabels[beginIndex:]:
                if objectLabel['type'] == 'car':
                    left = objectLabel['left']
                    top = objectLabel['top']
                    right = objectLabel['right']
                    bottom = objectLabel['bottom']   

                    num_objs += 1
                    boxes.append((left, top, right, bottom))
                    areas.append(sizeCondition)
                    self.toFrameIndexList.append(objectLabel['frame'])
                    self.toSceneIndexList.append(sceneIndex)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # self.targets["boundingBoxes"] = boxes
            labels = torch.ones((num_objs,), dtype=torch.int64)
            # self.targets["labels"] = labels
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
            # self.targets["iscrowd"].append(iscrowd)
            areas = torch.as_tensor(areas)
            # self.targets["area"] = areas
            temp = {}
            temp["boundingBoxes"] = boxes
            temp["labels"] = labels
            temp["iscrowd"] = iscrowd
            temp["area"] = areas
            self.targets.append(temp)
            
    def __getitem__(self, index):
        frameIndex = self.toFrameIndexList[index]
        sceneIndex = self.toSceneIndexList[index]
        nuScenesPath = self.toNuScenesPathDict[sceneIndex][frameIndex]

        fullImagePath = '%s/%s' % (self.root, nuScenesPath)
        fullImage = Image.open(fullImagePath).convert('RGB')

        boundingBox = self.targets[index]["boundingBoxes"][index]
        bbImage = fullImage.crop(boundingBox)

        bbImage = self.transform(bbImage)
        target = self.targets[index]
        
        return bbImage, target

    def __len__(self):
        return len(self.toFrameIndexList)

class TestDataset(Dataset):
    def __init__(self, dictionary, root, transform):
        super(TestDataset, self).__init__()

        self.root = root
        self.transform = transform

        labelDir = '%s/testing/label_0%d' % (root, args.cameraNumber)
        sceneIds = dictionary['indices']
        readMode = dictionary['readMode']
        sceneLabelsDict = readLabelFiles(labelDir, sceneIds, readMode=readMode)

        self.targets = []
        self.toFrameIndexList = []
        self.toNuScenesPathDict = {}
        self.toSceneIndexList = []

        for sceneIndex, objectLabels in sceneLabelsDict.items():
            kittiPath = '%s/testing/image_0%d/%04d.txt' % (self.root, args.cameraNumber, sceneIndex)
            nuScenesPaths = []
            num_objs = 0
            boxes = []
            areas = []
            with open(kittiPath, 'r') as imageFile:
                lines = imageFile.read().splitlines()
                for line in lines:
                    nuScenesPaths.append(line.split(' ')[-1])
                
            self.toNuScenesPathDict[sceneIndex] = nuScenesPaths

            for objectLabel in objectLabels:
                if objectLabel['type'] == 'car':
                    left = objectLabel['left']
                    top = objectLabel['top']
                    right = objectLabel['right']
                    bottom = objectLabel['bottom']   

                    num_objs += 1
                    boxes.append((left, top, right, bottom))
                    areas.append(sizeCondition)
                    self.toFrameIndexList.append(objectLabel['frame'])
                    self.toSceneIndexList.append(sceneIndex)
        
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # self.targets["boundingBoxes"] = boxes
            labels = torch.ones((num_objs,), dtype=torch.int64)
            # self.targets["labels"] = labels
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
            # self.targets["iscrowd"].append(iscrowd)
            areas = torch.as_tensor(areas)
            # self.targets["area"] = areas
            temp = {}
            temp["boundingBoxes"] = boxes
            temp["labels"] = labels
            temp["iscrowd"] = iscrowd
            temp["area"] = areas
            self.targets.append(temp)
    
    def __getitem__(self, index):
        frameIndex = self.toFrameIndexList[index]
        sceneIndex = self.toSceneIndexList[index]
        nuScenesPath = self.toNuScenesPathDict[sceneIndex][frameIndex]

        fullImagePath = '%s/%s' % (self.root, nuScenesPath)
        fullImage = Image.open(fullImagePath).convert('RGB')

        boundingBox = self.targets[index]["boundingBoxes"][index]
        bbImage = fullImage.crop(boundingBox)
        bbImage = self.transform(bbImage)

        target = self.targets[index]    
        return bbImage

    def __len__(self):
        return len(self.toFrameIndexList)

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = TrainDataset
    dataset_test = TestDataset

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")