# -*- coding: utf-8 -*-

import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- horizontal and vertical finite defference operators
def diffh(x):
    x_diffh = torch.zeros(x.shape)
    x_diffh[:,:-1] = x[:,1:] - x[:,0:-1]
    return x_diffh

def diffv(x):
    x_diffv = torch.zeros(x.shape)
    x_diffv[:-1, :] = x[1:, :] - x[0:-1,:]
    return x_diffv.T

# --- Total variation norm
def TVnorm(x):
    y = torch.sum(torch.sqrt(diffh(x)**2 + diffv(x)**2))
    return y


# def Grad_Image(x):

#     with torch.no_grad():

#         x = x.to(device).clone()
#         x_temp = x[1:, :] - x[0:-1,:]
#         dux = torch.cat((x_temp.T,torch.zeros(x_temp.shape[1],1,device=device)),1).to(device)
#         dux = dux.T
#         x_temp = x[:,1:] - x[:,0:-1]
#         duy = torch.cat((x_temp,torch.zeros((x_temp.shape[0],1),device=device)),1).to(device)
#         return  torch.cat((dux,duy),dim=0).to(device)