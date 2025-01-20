# -*- coding: utf-8 -*-

import numpy as np
import torch
from Grad_Image import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def tv(x):
    
    with torch.no_grad():

        Dx=Dx.view(-1)
        N = len(Dx)
        Dux = Dx[:int(N/2)]
        Dvx = Dx[int(N/2):N]
        tv = torch.sum(torch.sqrt(Dux**2 + Dvx**2))
        
        return tv