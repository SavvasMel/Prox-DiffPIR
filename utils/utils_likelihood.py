import numpy as np
from numpy.linalg import norm

'''
MIT License

Copyright (c) 2025 Savvas Melidonis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

# def forward_operator(x, task_current, M = None, FB=None, shape_in=None):

#     if task_current == "blurring":
#         return np.real(np.fft.ifftn(FB.cpu().numpy() * (np.fft.fftn(x.reshape(shape_in), axes=(-2,-1))), axes=(-2, -1))).ravel()
    
#     elif task_current == "masking":
#         return M.ravel()*x
    
#     elif task_current == "denoising":
#         return x

# def backward_operator(x, task_current, M = None, FBC=None, shape_in=None):

#     if task_current == "blurring":
#         return np.real(np.fft.ifftn(FBC.cpu().numpy() * (np.fft.fftn(x.reshape(shape_in), axes=(-2,-1))), axes=(-2, -1))).ravel()
    
#     elif task_current == "masking":
#         return M.ravel()*x
    
#     elif task_current == "denoising":
#         return x

class Objectives():

    def __init__(self, noise_type, task_current, x_hat, shape_in, measurement, alpha, beta, rho, FB = None, M = None, t=None):

        self.noise_type = noise_type
        self.task_current = task_current

        self.x_hat = np.asarray(x_hat.cpu().numpy())
        self.measurement = np.asarray(measurement.cpu())
        self.alpha = alpha
        self.beta = np.asarray(beta.cpu())
        self.rho = np.asarray(rho.cpu())

        if not t == None:
            self.t = np.asarray(t.cpu())

        if self.task_current == "deblurring":
            self.FB = FB.cpu().numpy()
            self.K = lambda x: np.real(np.fft.ifftn(self.FB * (np.fft.fftn(x.reshape(shape_in), axes=(-2,-1))), axes=(-2, -1))).ravel()
            return

        elif self.task_current == "masking":
            self.M = M
            self.K = lambda x: (self.M*x.reshape(shape_in)).ravel()
            return 

        elif self.task_current == "denoising":
            self.K = lambda x: x
            return 
        else:
            raise TypeError("Inverse problems that are accepted: deblurring, masking or denoising")

    def objective(self, x):

        if self.noise_type == "Poisson":
            return np.sum(-self.measurement * np.log(self.alpha * self.K(x)+ self.beta) + self.alpha * self.K(x) + self.beta) + self.rho * norm(x - self.x_hat, 2)**2
        elif self.noise_type == "binomial":
            calc = (self.alpha*self.K(x) + self.beta)
            return np.sum(-self.measurement*np.log(1-np.exp(-calc)) + (self.t - self.measurement) * calc) + self.rho * norm(x - self.x_hat, 2)**2
        elif self.noise_type == "geometric":
            calc = (self.alpha*self.K(x) + self.beta)
            return np.sum(- np.log(1 - np.exp(-calc)) + calc * (self.measurement-1)) + self.rho * norm(x - self.x_hat, 2)**2
        else:
            raise TypeError("Accepted noise types: Poisson, binomial or geometric")

class Derivatives():

    def __init__(self, noise_type, task_current, x_hat, shape_in, measurement, alpha, beta, rho, FB = None, FBC = None, M = None, t=None):
        
        self.noise_type = noise_type
        self.task_current = task_current

        self.x_hat = np.asarray(x_hat.cpu().numpy())
        self.measurement = np.asarray(measurement.cpu())
        self.alpha = alpha
        self.beta = np.asarray(beta.cpu())
        self.rho = np.asarray(rho.cpu())

        if not t == None:
            self.t = np.asarray(t.cpu())

        if self.task_current == "deblurring":
            self.FBC = FBC.cpu().numpy()
            self.FB = FBC.cpu().numpy()
            self.K = lambda x: np.real(np.fft.ifftn(self.FB * (np.fft.fftn(x.reshape(shape_in), axes=(-2,-1))), axes=(-2, -1))).ravel()
            self.KT = lambda x: np.real(np.fft.ifftn(self.FBC * (np.fft.fftn(x.reshape(shape_in), axes=(-2,-1))), axes=(-2, -1))).ravel()
            return

        elif self.task_current == "masking":
            self.M = M
            self.K = lambda x: (self.M*x.reshape(shape_in)).ravel()
            self.KT = lambda x: (self.M*x.reshape(shape_in)).ravel()
            return 

        elif self.task_current == "denoising":
            self.K = lambda x: x
            self.KT = lambda x: x
            return 
        else:
            raise TypeError("Inverse problems that are accepted: deblurring, masking or denoising")

    def derivative(self, x):

        if self.noise_type == "Poisson":
            return  self.alpha * (1 - self.KT(self.measurement * (1 / (self.alpha * self.K(x) + self.beta)))) + 2 * self.rho * (x - self.x_hat)
        elif self.noise_type == "binomial":
            exp_calc = np.exp(-(self.alpha * self.K(x) + self.beta))
            return self.alpha *  self.KT(self.t - self.measurement) - self.KT(self.alpha * self.measurement * (exp_calc) / (1-exp_calc)) + 2 * self.rho * (x - self.x_hat)
        elif self.noise_type == "geometric":
            exp_calc = np.exp(-(self.alpha * self.K(x) + self.beta))
            return self.alpha * self.KT(self.measurement-1) - self.KT(self.alpha * exp_calc / (1-exp_calc)) + 2 * self.rho * (x - self.x_hat)
        else:
            raise TypeError("Accepted noise types: Poisson, binomial or geometric")

