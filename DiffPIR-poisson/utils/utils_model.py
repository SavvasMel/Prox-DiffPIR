# -*- coding: utf-8 -*-
import numpy as np
import torch
import scipy
from numpy.linalg import norm
from utils import utils_image as util
# from utils.utils_deblur import G, Gt
from functools import partial

from guided_diffusion.script_util import add_dict_to_argparser
import argparse

'''
modified by Kai Zhang (github: https://github.com/cszn)
03/03/2019
'''


def test_mode(model_fn, model_diffusion, L, mode=0, refield=32, min_size=256, sf=1, modulo=1, noise_level=0, vec_t=None, \
        model_out_type='pred_xstart', diffusion=None, ddim_sample=False, alphas_cumprod=None):
    '''
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Some testing modes
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # (0) normal: test(model, L)
    # (1) pad: test_pad(model, L, modulo=16)
    # (2) split: test_split(model, L, refield=32, min_size=256, sf=1, modulo=1)
    # (3) x8: test_x8(model, L, modulo=1)
    # (4) split and x8: test_split_x8(model, L, refield=32, min_size=256, sf=1, modulo=1)
    # (4) split only once: test_onesplit(model, L, refield=32, min_size=256, sf=1, modulo=1)
    # ---------------------------------------
    '''

    model = partial(model_fn, model_diffusion=model_diffusion, diffusion=diffusion, ddim_sample=False, alphas_cumprod=alphas_cumprod)
    
    if mode == 0:
        E = test(model, L, noise_level, vec_t, model_out_type)
    elif mode == 1:
        E = test_pad(model, L, modulo, noise_level, vec_t, model_out_type)
    elif mode == 2:
        E = test_split(model, L, refield, min_size, sf, modulo, noise_level, vec_t, model_out_type)
    elif mode == 3:
        E = test_x8(model, L, modulo, noise_level, vec_t, model_out_type)
    elif mode == 4:
        E = test_split_x8(model, L, refield, min_size, sf, modulo, noise_level, vec_t, model_out_type)
    elif mode == 5:
        E = test_onesplit(model, L, refield, min_size, sf, modulo, noise_level, vec_t, model_out_type)
    return E


'''
# ---------------------------------------
# normal (0)
# ---------------------------------------
'''


def test(model, L, noise_level=15, vec_t=None, model_out_type='pred_xstart'):
    E = model(L, noise_level, vec_t=vec_t, model_out_type=model_out_type)
    return E


'''
# ---------------------------------------
# pad (1)
# ---------------------------------------
'''


def test_pad(model, L, modulo=16, noise_level=15, vec_t=None, model_out_type='pred_xstart'):
    h, w = L.size()[-2:]
    paddingBottom = int(np.ceil(h/modulo)*modulo-h)
    paddingRight = int(np.ceil(w/modulo)*modulo-w)
    L = torch.nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(L)
    E = model(L, noise_level, vec_t=vec_t, model_out_type=model_out_type)
    E = E[..., :h, :w]
    return E


'''
# ---------------------------------------
# split (function)
# ---------------------------------------
'''


def test_split_fn(model, L, refield=32, min_size=256, sf=1, modulo=1, noise_level=15, vec_t=None, model_out_type='pred_xstart'):
    '''
    model:
    L: input Low-quality image
    refield: effective receptive filed of the network, 32 is enough
    min_size: min_sizeXmin_size image, e.g., 256X256 image
    sf: scale factor for super-resolution, otherwise 1
    modulo: 1 if split
    '''
    h, w = L.size()[-2:]
    if h*w <= min_size**2:
        L = torch.nn.ReplicationPad2d((0, int(np.ceil(w/modulo)*modulo-w), 0, int(np.ceil(h/modulo)*modulo-h)))(L)
        E = model(L, noise_level, vec_t=vec_t, model_out_type=model_out_type)
        E = E[..., :h*sf, :w*sf]
    else:
        top = slice(0, (h//2//refield+1)*refield)
        bottom = slice(h - (h//2//refield+1)*refield, h)
        left = slice(0, (w//2//refield+1)*refield)
        right = slice(w - (w//2//refield+1)*refield, w)
        Ls = [L[..., top, left], L[..., top, right], L[..., bottom, left], L[..., bottom, right]]

        if h * w <= 4*(min_size**2):
            Es = [model(Ls[i], noise_level, vec_t=vec_t, model_out_type=model_out_type) for i in range(4)]
        else:
            Es = [test_split_fn(model, Ls[i], refield=refield, min_size=min_size, sf=sf, modulo=modulo, noise_level=noise_level, vec_t=vec_t, model_out_type=model_out_type) for i in range(4)]

        b, c = Es[0].size()[:2]
        E = torch.zeros(b, c, sf * h, sf * w).type_as(L)

        E[..., :h//2*sf, :w//2*sf] = Es[0][..., :h//2*sf, :w//2*sf]
        E[..., :h//2*sf, w//2*sf:w*sf] = Es[1][..., :h//2*sf, (-w + w//2)*sf:]
        E[..., h//2*sf:h*sf, :w//2*sf] = Es[2][..., (-h + h//2)*sf:, :w//2*sf]
        E[..., h//2*sf:h*sf, w//2*sf:w*sf] = Es[3][..., (-h + h//2)*sf:, (-w + w//2)*sf:]
    return E



def test_onesplit(model, L, refield=32, min_size=256, sf=1, modulo=1, noise_level=15, vec_t=None, model_out_type='pred_xstart'):
    '''
    model:
    L: input Low-quality image
    refield: effective receptive filed of the network, 32 is enough
    min_size: min_sizeXmin_size image, e.g., 256X256 image
    sf: scale factor for super-resolution, otherwise 1
    modulo: 1 if split
    '''
    h, w = L.size()[-2:]

    top = slice(0, (h//2//refield+1)*refield)
    bottom = slice(h - (h//2//refield+1)*refield, h)
    left = slice(0, (w//2//refield+1)*refield)
    right = slice(w - (w//2//refield+1)*refield, w)
    Ls = [L[..., top, left], L[..., top, right], L[..., bottom, left], L[..., bottom, right]]
    Es = [model(Ls[i],noise_level,vec_t=vec_t,model_out_type=model_out_type) for i in range(4)]
    b, c = Es[0].size()[:2]
    E = torch.zeros(b, c, sf * h, sf * w).type_as(L)
    E[..., :h//2*sf, :w//2*sf] = Es[0][..., :h//2*sf, :w//2*sf]
    E[..., :h//2*sf, w//2*sf:w*sf] = Es[1][..., :h//2*sf, (-w + w//2)*sf:]
    E[..., h//2*sf:h*sf, :w//2*sf] = Es[2][..., (-h + h//2)*sf:, :w//2*sf]
    E[..., h//2*sf:h*sf, w//2*sf:w*sf] = Es[3][..., (-h + h//2)*sf:, (-w + w//2)*sf:]
    return E



'''
# ---------------------------------------
# split (2)
# ---------------------------------------
'''


def test_split(model, L, refield=32, min_size=256, sf=1, modulo=1, noise_level=15, vec_t=None, model_out_type='pred_xstart'):
    E = test_split_fn(model, L, refield=refield, min_size=min_size, sf=sf, modulo=modulo, noise_level=noise_level, vec_t=vec_t, model_out_type=model_out_type)
    return E


'''
# ---------------------------------------
# x8 (3)
# ---------------------------------------
'''


def test_x8(model, L, modulo=1, noise_level=15, vec_t=None, model_out_type='pred_xstart'):
    E_list = [test_pad(model, util.augment_img_tensor(L, mode=i), modulo=modulo, noise_level=noise_level, vec_t=vec_t, model_out_type=model_out_type) for i in range(8)]
    for i in range(len(E_list)):
        if i == 3 or i == 5:
            E_list[i] = util.augment_img_tensor(E_list[i], mode=8 - i)
        else:
            E_list[i] = util.augment_img_tensor(E_list[i], mode=i)
    output_cat = torch.stack(E_list, dim=0)
    E = output_cat.mean(dim=0, keepdim=False)
    return E


'''
# ---------------------------------------
# split and x8 (4)
# ---------------------------------------
'''


def test_split_x8(model, L, refield=32, min_size=256, sf=1, modulo=1, noise_level=15, vec_t=None, model_out_type='pred_xstart'):
    E_list = [test_split_fn(model, util.augment_img_tensor(L, mode=i), refield=refield, min_size=min_size, sf=sf, modulo=modulo, noise_level=noise_level, vec_t=vec_t, model_out_type=model_out_type) for i in range(8)]
    for k, i in enumerate(range(len(E_list))):
        if i==3 or i==5:
            E_list[k] = util.augment_img_tensor(E_list[k], mode=8-i)
        else:
            E_list[k] = util.augment_img_tensor(E_list[k], mode=i)
    output_cat = torch.stack(E_list, dim=0)
    E = output_cat.mean(dim=0, keepdim=False)
    return E


# ----------------------------------------
# wrap diffusion model
# ----------------------------------------

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def model_fn(x, noise_level, model_diffusion, vec_t=None, model_out_type='pred_xstart', \
        diffusion=None, ddim_sample=False, alphas_cumprod=None, **model_kwargs):

    sqrt_alphas_cumprod     = torch.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod  = torch.sqrt(1. - alphas_cumprod)
    reduced_alpha_cumprod   = torch.div(sqrt_1m_alphas_cumprod, sqrt_alphas_cumprod)        # equivalent noise sigma on image

    # time step corresponding to noise level
    if not torch.is_tensor(vec_t):
        t_step = find_nearest(reduced_alpha_cumprod,(noise_level/255.))
        vec_t = torch.tensor([t_step] * x.shape[0], device=x.device)
        # timesteps = torch.linspace(1, 1e-3, num_train_timesteps, device=device)
        # t = timesteps[t_step]
    if not ddim_sample:
        out = diffusion.p_sample(
            model_diffusion,
            x,
            vec_t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=model_kwargs,
        )
    else:
        out = diffusion.ddim_sample(
            model_diffusion,
            x,
            vec_t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=model_kwargs,
            eta=0,
        )

    if model_out_type == 'pred_x_prev_and_start':
        return out["sample"], out["pred_xstart"]
    elif model_out_type == 'pred_x_prev':
        out = out["sample"]
    elif model_out_type == 'pred_xstart':
        out = out["pred_xstart"]
    elif model_out_type == 'epsilon':
        alpha_prod_t = alphas_cumprod[int(t_step)]
        beta_prod_t = 1 - alpha_prod_t
        out = (x - alpha_prod_t ** (0.5) * out["pred_xstart"]) / beta_prod_t ** (0.5)
    elif model_out_type == 'score':
        alpha_prod_t = alphas_cumprod[int(t_step)]
        beta_prod_t = 1 - alpha_prod_t
        out = (x - alpha_prod_t ** (0.5) * out["pred_xstart"]) / beta_prod_t ** (0.5)
        out = - out / beta_prod_t ** (0.5)
            
    return out



'''
# ^_^-^_^-^_^-^_^-^_^-^_^-^_^-^_^-^_^-^_^
# _^_^-^_^-^_^-^_^-^_^-^_^-^_^-^_^-^_^-^_
# ^_^-^_^-^_^-^_^-^_^-^_^-^_^-^_^-^_^-^_^
'''


'''
# ---------------------------------------
# print
# ---------------------------------------
'''


# -------------------
# print model
# -------------------
def print_model(model):
    msg = describe_model(model)
    print(msg)


# -------------------
# print params
# -------------------
def print_params(model):
    msg = describe_params(model)
    print(msg)


'''
# ---------------------------------------
# information
# ---------------------------------------
'''


# -------------------
# model inforation
# -------------------
def info_model(model):
    msg = describe_model(model)
    return msg


# -------------------
# params inforation
# -------------------
def info_params(model):
    msg = describe_params(model)
    return msg


'''
# ---------------------------------------
# description
# ---------------------------------------
'''


# ----------------------------------------------
# model name and total number of parameters
# ----------------------------------------------
def describe_model(model):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    msg = '\n'
    msg += 'models name: {}'.format(model.__class__.__name__) + '\n'
    msg += 'Params number: {}'.format(sum(map(lambda x: x.numel(), model.parameters()))) + '\n'
    msg += 'Net structure:\n{}'.format(str(model)) + '\n'
    return msg


# ----------------------------------------------
# parameters description
# ----------------------------------------------
def describe_params(model):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    msg = '\n'
    msg += ' | {:^6s} | {:^6s} | {:^6s} | {:^6s} || {:<20s}'.format('mean', 'min', 'max', 'std', 'param_name') + '\n'
    for name, param in model.state_dict().items():
        if not 'num_batches_tracked' in name:
            v = param.data.clone().float()
            msg += ' | {:>6.3f} | {:>6.3f} | {:>6.3f} | {:>6.3f} || {:s}'.format(v.mean(), v.min(), v.max(), v.std(), name) + '\n'
    return msg

# ----------------------------------------
# load model
# ----------------------------------------

def create_argparser(model_config):
    defaults = dict(
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path='',
        diffusion_steps=1000,
        noise_schedule='linear',
        num_head_channels=64,
        resblock_updown=True,
        use_fp16=False,
        use_scale_shift_norm=True,
        num_heads=4,
        num_heads_upsample=-1,
        use_new_attention_order=False,
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        channel_mult="",
        learn_sigma=True,
        class_cond=False,
        use_checkpoint=False,
        image_size=256,
        num_channels=128,
        num_res_blocks=1,
        attention_resolutions="16",
        dropout=0.1,
    )
    defaults.update(model_config)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def grad_and_value(operator, x, x_hat, measurement):
    difference = measurement - operator(x_hat)
    norm = torch.linalg.norm(difference)
    norm_grad = torch.autograd.grad(outputs=norm, inputs=x)[0]
    return norm_grad,  norm

def PIDAL(x_hat, FB, FBC, F2B, measurement, mu, rho, alpha_pois):
    
    NbIt = 5000
    N_dim = x_hat.ravel().shape[0]
    normx = torch.zeros(NbIt) 

    u_1 = x_hat.clone()
    u_2 = x_hat.clone()  
    u_3 = x_hat.clone()  

    d_1 = torch.zeros_like(x_hat)
    d_2 = torch.zeros_like(x_hat)
    d_3 = torch.zeros_like(x_hat)

    z_aprox = u_3.clone()

    mu = mu.float()
    rho = rho.float()

    def K(z):
        return torch.real(torch.fft.ifftn(FB.mul(torch.fft.fftn(z, dim=(-2,-1))), dim=(-2, -1)))

    def KT(z):
        return torch.real(torch.fft.ifftn(FBC.mul(torch.fft.fftn(z, dim=(-2,-1))), dim=(-2, -1)))

    for it in range(0, NbIt):

        zold = z_aprox.clone()
        
        zeta_1 = u_1 + d_1
        zeta_2 = u_2 + d_2
        zeta_3 = u_3 + d_3
        
        gamma = KT(zeta_1) + zeta_2 + zeta_3

        z = torch.real(torch.fft.ifftn(torch.fft.fftn(gamma, dim=(-2,-1))/(F2B+2), dim=(-2, -1)))
    
        Kz = K(z)
        
        nu_1 = Kz - d_1
        
        u_1_old = u_1.clone()
        u_1 = 1/2 * (nu_1 - alpha_pois/mu+ torch.sqrt((nu_1-alpha_pois/mu)**2+4*measurement/mu))
        
        nu_2 = z-d_2
        
        u_2_old = u_2.clone()
        u_2 = (mu*nu_2 + 2*rho*x_hat)/(mu+2*rho)
        
        nu_3 = z - d_3
        
        u_3_old = u_3.clone()
        u_3 = torch.maximum(nu_3, torch.tensor(0))
        
        d_1 = d_1-Kz+u_1
        d_2 = d_2-z+u_2
        d_3 = d_3-z+u_3
        
        z_aprox = u_3.clone()

        # r_res = torch.norm(torch.vstack((Kz-u_1,z-u_2,z-u_3)).ravel(),2).cpu()
        # s_res = torch.norm((-mu*(KT(u_1-u_1_old)+u_2-u_2_old+u_3-u_3_old)).ravel(),2).cpu()
        # print(r_res)
        # print(s_res)

        # eps_abs = torch.tensor(1e-6).cpu()
        # eps_rel = torch.tensor(1e-6).cpu()
        # norm_1 = torch.norm(torch.vstack((Kz,z,z)).ravel(),2).cpu()
        # norm_2 = torch.norm(torch.vstack((u_1,u_2,u_3)).ravel(),2).cpu()
        # eps_pri = np.sqrt(3*N_dim**2)*eps_abs + eps_rel*torch.max(torch.stack((norm_1,norm_2,torch.tensor(0)))).cpu()
        # eps_dual = np.sqrt(N_dim**2)*eps_abs + mu*eps_rel*torch.norm((KT(d_1)+d_2+d_3).ravel(),2).cpu()
        normx[it] = torch.norm((z_aprox-zold).ravel(),2)/torch.norm(zold.ravel(),2) 

        # if r_res<=eps_pri and s_res<=eps_dual:
        #     break

        Stop_norm = 1e-4
        if it > 10 and normx[it] < Stop_norm:
          
            break

    normx = normx[:it+1]

    # print(r_res)
    # print(s_res)
    # print("Norm of x = ", normx[-1])
    print("End of iterations:", it+1)
    # print("primal residual = ", r_res)
    # print("dual residual = ", s_res)

    return z_aprox

# class obj_der():

#     def __init__(self, noise_type, K, KT, x_hat, measurement, alpha, beta, rho, t=None):
#         self.noise_type = noise_type
#         self.K = K
#         self.KT = KT
#         self.x_hat = x_hat
#         self.measurement =measurement
#         self.alpha = alpha
#         self.beta = beta
#         self.rho = rho
#         self.t = t

#     def objective(self, x):
        
#         if self.noise_type == "Poisson":
#             return np.sum(-self.measurement * np.log(self.alpha * self.K(x)+ self.beta) + self.alpha * self.K(x) + self.beta) + self.rho * norm(x - self.x_hat, 2)**2
#         elif self.noise_type == "binomial":
#             calc = (self.alpha*self.K(x) + self.beta)
#             return np.sum(-self.measurement*np.log(1-np.exp(-calc)) + (self.t - self.measurement) * calc) + self.rho * norm(x - self.x_hat, 2)**2
#         elif self.noise_type == "geometric":
#             calc = (self.alpha*self.K(x) + self.beta)
#             return np.sum(- np.log(1 - np.exp(-calc)) + calc * (self.measurement-1)) + self.rho * norm(x - self.x_hat, 2)**2
#         else:
#             raise TypeError("Accepted noise types: Poisson, binomial or geometric")
        
#     def derivative(self, x):

#         if self.noise_type == "Poisson":
#             return  self.alpha * (1 - self.KT(self.measurement * (1 / (self.alpha * self.K(x) + self.beta)))) + 2 * self.rho * (x - self.x_hat)
#         elif self.noise_type == "binomial":
#             exp_calc = np.exp(-(self.alpha * self.K(x) + self.beta))
#             return self.alpha *  self.KT(self.t - self.measurement) - self.KT(self.alpha * self.measurement * (exp_calc) / (1-exp_calc)) + 2 * self.rho * (x - self.x_hat)
#         elif self.noise_type == "geometric":
#             exp_calc = np.exp(-(self.alpha * self.K(x) + self.beta))
#             return self.alpha * self.KT(self.measurement-1) - self.KT(self.alpha * exp_calc / (1-exp_calc)) + 2 * self.rho * (x - self.x_hat)
#         else:
#             raise TypeError("Accepted noise types: Poisson, binomial or geometric")

# def objective(x, x_hat, rho):
#     return np.sum(-measurement * np.log(alpha_pois * G(x, k, shape_in) + beta) + alpha_pois * G(x, k, shape_in) + beta) + rho * norm(x - x_hat, 2)**2

# def derivative(x, x_hat, rho):

#     Kz = alpha_pois * G(x, k, shape_in)
#     inv_Kz = 1 / (Kz + beta)
#     return  alpha_pois * (1 - Gt(measurement * inv_Kz, k, shape_in)) + 2 * rho * (x - x_hat)

def LBFGSB(obj, der, x_hat, shape_in, tol):

    bounds = (((0., 1.),) * x_hat.shape[0] )
    result = scipy.optimize.minimize(obj, x_hat.cpu().numpy(), method='L-BFGS-B', jac=der, bounds = bounds, tol = tol) #options = {"tol" : 1e-2}
    print("finished")
    print(np.max(result["x"]))
    return result["x"].reshape(shape_in)

# def LBFGSB(x_hat, FB, FBC, shape_in, noise_type, measurement, alpha, beta, rho, tol, t = None):

    
#     if not t == None:
#         t = np.asarray(t.cpu())
#     beta = np.asarray(beta.cpu())
#     rho = np.asarray(rho.cpu())
#     x_hat = np.asarray(x_hat.cpu())
#     measurement = np.asarray(measurement.cpu())
    
#     obj_de = obj_der(noise_type, K, KT, x_hat, measurement, alpha, beta, rho, t)
#     obj = obj_de.objective
#     der = obj_de.derivative

#     bounds = (((0., None),) * x_hat.shape[0] )
#     result = scipy.optimize.minimize(obj, x_hat, method='L-BFGS-B', jac=der, bounds = bounds, tol = tol) #options = {"tol" : 1e-2}
#     print("finished")
#     return result["x"].reshape(shape_in)

if __name__ == '__main__':

    class Net(torch.nn.Module):
        def __init__(self, in_channels=3, out_channels=3):
            super(Net, self).__init__()
            self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)

        def forward(self, x):
            x = self.conv(x)
            return x

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    model = Net()
    model = model.eval()
    print_model(model)
    print_params(model)
    x = torch.randn((2,3,400,400))
    torch.cuda.empty_cache()
    with torch.no_grad():
        for mode in range(5):
            y = test_mode(model, x, mode)
            print(y.shape)
