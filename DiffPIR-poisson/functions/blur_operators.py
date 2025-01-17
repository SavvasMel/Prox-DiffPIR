import torch
import numpy as np
from functions.max_eigenval import max_eigenval

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def blur_operators(kernel_len, size, type_blur, var = None):

    nx = size[0]
    ny = size[1]
    if type_blur=='uniform':
        h = torch.zeros(nx,ny).to(device)
        lx = kernel_len[0]
        ly = kernel_len[1]
        h[0:lx,0:ly] = 1/(lx*ly)
        c =  np.ceil((np.array([ly,lx])-1)/2).astype("int64")
    if type_blur=='gaussian':
        if var != None:
            [x,y] = torch.meshgrid(torch.arange(-ny/2,ny/2),torch.arange(-nx/2,nx/2)).to(device)
            h = torch.exp(-(x**2+y**2)/(2*var))
            h = h/torch.sum(h)
            c = np.ceil(np.array([nx,ny])/2).astype("int64") 
        else:
            print("Choose a variance for the Gaussian filter.")

    H_FFT = torch.fft.fft2(torch.roll(h, shifts = (-c[0],-c[1]), dims=(0,1)))
    HC_FFT = torch.conj(H_FFT)

    # A forward operator
    def A(x):
        a = torch.zeros((256,256,3)).to(device)
        for i in range(3):
            a[:,:,i] = torch.fft.ifft2(torch.multiply(H_FFT,torch.fft.fft2(x[:,:,i]))).real.reshape(x[:,:,i].shape)
        return a

    # A backward operator
    def AT(x):
        at = torch.zeros((256,256,3)).to(device)
        for i in range(3):
            at[:,:,i] = torch.fft.ifft2(torch.multiply(HC_FFT,torch.fft.fft2(x[:,:,i]))).real.reshape(x[:,:,i].shape)
        return at

    AAT_norm = max_eigenval(A, AT, nx, 1e-4, int(1e4), 0)

    return A, AT, AAT_norm