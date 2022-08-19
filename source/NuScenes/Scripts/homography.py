# Helpful Link - https://cseweb.ucsd.edu/classes/wi07/cse252a/homography_estimation/homography_estimation.pdf

import torch
import torch.nn as nn
import numpy as np

#Get Q_pred and Q_gt for each of the N detected objects from the 3D object detector.
a_x_y = []
H = np.array() #Initialize Homography to something ?

#Calculating the Homography Matrix using the candidate points 
for i in range(N):
    Q_pred = getQpred(i) #3x1 vector (Q_pred) [x2, y2, z2]
    Q_gt = getQgt(i) #3x1 vector (q_gt) [x1, y1, z1]
    Q_pred_tilde = Q_pred
    Q_pred_tilde[2] = 1
    Q_gt_tilde = Q_gt
    Q_gt_tilde[2] = 1
    a_x = np.array([-Q_gt_tilde[0], -Q_gt[1], -1, 0, 0, 0, Q_pred_tilde[0], Q_pred_tilde[0], Q_pred_tilde[0]])
    a_x = np.transpose(a_x)
    a_y = np.array([0, 0, 0, -Q_gt_tilde[0], -Q_gt_tilde[1], -1, Q_pred_tilde[1], Q_pred_tilde[1], Q_pred_tilde[1]])
    a_y = np.transpose(a_y)
    a_x_y.append(np.transpose(a_x))
    a_x_y.append(np.transpose(a_y))

A = np.array(a_x_y)
h = np.array([H[0][0] H[0][1], H[0][2], H[1][0], H[1][1], H[1][2], H[2][0], H[2][1], H[2][2]])
h = np.transpose(h)

predictions = np.dot(h, Q_gt)
target = Q_gt
homo_loss = nn.SmoothL1Loss(beta=1.0)
output = homo_loss(predictions, target)

#Homo_loss will tell us how much the Homography Matrix should change and since, Q_pred_tilde = H*qgt. The homography matrix will only change when Q_pred changes.
#So indirectly Homo_loss tells us how much should the Q_pred_tilde change