import os.path
import cv2
import logging

import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime, date
from collections import OrderedDict
import hdf5storage

from tqdm.auto import tqdm
from utils import utils_model
from utils import utils_logger
from utils import utils_sisr as sr
from utils import utils_image as util
from utils import utils_welford
from utils.save_plots import save_plots
from utils.save_progress import save_progress
from utils.utils_deblur import MotionBlurOperator, GaussialBlurOperator, MaskingOperator, DenoisingOperator
from scipy import ndimage
from utils.utils_likelihood import Objectives, Derivatives #, forward_operator, backward_operator
from skimage.metrics import structural_similarity as ssim

# from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)

def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------
    today = date.today()
    rate = 0.0001
    tol = 7e-6
    alpha_geo = 255 * rate
    model_name              = 'diffusion_ffhq_10m'  # diffusion_ffhq_10m, 256x256_diffusion_uncond; set diffusino model
    testset_name            = 'ffhq_val'            # set testing set,  'imagenet_val' | 'ffhq_val'
    num_train_timesteps     = 1000
    iter_num                = 100               # set number of iterations
    iter_num_U              = 1                 # set number of inner iterations, default: 1
    skip                    = num_train_timesteps//iter_num     # skip interval

    total_samples = 3
    save_samples = 3
    thinning_step = np.int64(total_samples/save_samples)

    show_img                = False             # default: False
    save_L                  = True             # save LR image
    save_E                  = True              # save estimated image
    save_LEH                = False             # save zoomed LR, E and H images
    save_progressive        = False             # save generation process
    border                  = 0
	
    lambda_                 = 1.0               # key parameter lambda
    
    log_process             = False
    ddim_sample             = False             # sampling method
    model_output_type       = 'pred_xstart'     # model output type: pred_x_prev; pred_xstart; epsilon; score
    generate_mode           = 'DiffPIR'         # DiffPIR; DPS; vanilla
    skip_type               = 'quad'            # uniform, quad
    eta                     = 0.0               # eta for ddim sampling
    zeta                    = 0.1  
    guidance_scale          = 1.0   

    calc_LPIPS              = True
    use_DIY_kernel          = True
    blur_mode               = 'Gaussian'          # Gaussian; motion      
    kernel_size             = 5
    kernel_std              = 3.0 if blur_mode == 'Gaussian' else 0.5
    prob = 0.5

    sf                      = 1
    task_current            = "masking" #"denoising" #'deblur'          
    n_channels              = 3                 # fixed
    cwd                     = ''  
    model_zoo               = os.path.join(cwd, 'model_zoo')    # fixed
    testsets                = os.path.join(cwd, 'testsets')     # fixed
    results                 = os.path.join(cwd, str(today) + '_results')      # fixed

    beta_start              = 0.1 / 1000
    beta_end                = 20 / 1000

    result_name             = f'{testset_name}_{task_current}_{generate_mode}_{model_name}_rate{rate}_NFE{iter_num}_task_current{task_current}_geometric_tol{tol}_bound_at_1_new_gs'
    model_path              = os.path.join(model_zoo, model_name+'.pt')
    device                  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # noise schedule 
    betas                   = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
    betas                   = torch.from_numpy(betas).to(device)
    alphas                  = 1.0 - betas
    alphas_cumprod          = np.cumprod(alphas.cpu(), axis=0)
    sqrt_alphas_cumprod     = torch.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod  = torch.sqrt(1. - alphas_cumprod)
    reduced_alpha_cumprod   = torch.div(sqrt_1m_alphas_cumprod, sqrt_alphas_cumprod)        # equivalent noise sigma on image

    noise_model_t           = 0
    
    noise_inti_img          = 50 / 255
    t_start                 = utils_model.find_nearest(reduced_alpha_cumprod, 2 * noise_inti_img) # start timestep of the diffusion process
    t_start                 = num_train_timesteps - 1              

    
    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------

    L_path = os.path.join(testsets, testset_name) # L_path, for Low-quality images
    E_path = os.path.join(results, result_name)   # E_path, for Estimated images
    util.mkdir(E_path)

    for img_dir in ['input', 'label', 'recon_n']:
        os.makedirs(os.path.join(E_path, img_dir), exist_ok=True)

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    # ----------------------------------------
    # load model
    # ----------------------------------------

    model_config = dict(
            model_path=model_path,
            num_channels=128,
            num_res_blocks=1,
            attention_resolutions="16",
        ) if model_name == 'diffusion_ffhq_10m' \
        else dict(
            model_path=model_path,
            num_channels=256,
            num_res_blocks=2,
            attention_resolutions="8,16,32",
        )
    args = utils_model.create_argparser(model_config).parse_args([])
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()))
    # model.load_state_dict(
    #     dist_util.load_state_dict(args.model_path, map_location="cpu")
    # )
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.eval()
    if generate_mode != 'DPS_y0':
        # for DPS_yt, we can avoid backward through the model
        for k, v in model.named_parameters():
            v.requires_grad = False
    model = model.to(device)

    logger.info('model_name:{}, Poisson rate:{:.3f}'.format(model_name, rate))
    logger.info('eta:{:.3f}, zeta:{:.3f}, lambda:{:.3f}, guidance_scale:{:.2f} '.format(eta, zeta, lambda_, guidance_scale))
    logger.info('start step:{}, skip_type:{}, skip interval:{}, skipstep analytic steps:{}'.format(t_start, skip_type, skip, noise_model_t))
    logger.info('Model path: {:s}'.format(model_path))
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)
    
    if calc_LPIPS:
        import lpips
        loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    
    def test_rho(lambda_=lambda_, zeta=zeta, model_output_type=model_output_type):
        logger.info('eta:{:.3f}, zeta:{:.3f}, lambda:{:.3f}, guidance_scale:{:.2f}'.format(eta, zeta, lambda_, guidance_scale))
        test_results = OrderedDict()
        test_results['psnr'] = []
        if calc_LPIPS:
            test_results['lpips'] = []

        for idx, img in enumerate(L_paths):

            model_out_type = model_output_type

            # --------------------------------
            # (1) get img_H
            # --------------------------------

            img_name, ext = os.path.splitext(os.path.basename(img))
            fname = str(idx).zfill(5) + '.png'
            img_H = util.imread_uint(img, n_channels=n_channels)
            img_H = util.modcrop(img_H, 8)  # modcrop

            # --------------------------------
            # (2) get img_L
            # --------------------------------

            np.random.seed(seed=123)
            torch.manual_seed(123)

            kernel = MaskingOperator(kernel_size=256, p=prob)  
            mask = kernel.get_kernel()
            # kernel = DenoisingOperator(kernel_size=256)  
            # mask = kernel.get_kernel()

            img_L = kernel.forward(img_H.transpose(2,0,1)) # np.copy(img_H)

            # Create synthetic data

            np.random.seed(seed=0)
            torch.manual_seed(0)

            img_L = util.uint2single(img_L)
            img_L = img_L.clip(0, 1)

            print("MIV: ", np.mean(img_L * alpha_geo))
            b_geo =  torch.tensor(np.mean(img_L * alpha_geo)*0.01).to(device)
            img_L = np.random.geometric(1-np.exp(-(img_L*alpha_geo + b_geo.cpu().numpy())))
            
            y = torch.from_numpy(np.ascontiguousarray(img_L)).float().unsqueeze(0).to(device)
            img_L = img_L.transpose(1,2,0)
            # y = util.single2tensor4(img_L).to(device)
            print(y.shape)
            print(img_L.shape)
            # --------------------------------
            # (2) get rhos and sigmas
            # --------------------------------

            sigmas = []
            sigma_ks = []
            rhos = []
            for i in range(num_train_timesteps):
                sigmas.append(reduced_alpha_cumprod[num_train_timesteps-1-i])
                if model_out_type == 'pred_xstart' and generate_mode == 'DiffPIR':
                    sigma_ks.append((sqrt_1m_alphas_cumprod[i]/sqrt_alphas_cumprod[i]))
                #elif model_out_type == 'pred_x_prev':
                else:
                    sigma_ks.append(torch.sqrt(betas[i]/alphas[i]))
                rhos.append(lambda_/(sigma_ks[i]**2))    
            rhos, sigmas, sigma_ks = torch.tensor(rhos).to(device), torch.tensor(sigmas).to(device), torch.tensor(sigma_ks).to(device)
            
            # --------------------------------
            # (3) initialize x, and pre-calculation
            # --------------------------------

            # x = torch.randn_like(y) #y.clone()
            # t_y = 200
            # sqrt_alpha_effective = sqrt_alphas_cumprod[t_start] / sqrt_alphas_cumprod[t_y]
            # x = sqrt_alpha_effective * (2*y/torch.max(y)-1) + torch.sqrt(sqrt_1m_alphas_cumprod[t_start]**2 - \
            #         sqrt_alpha_effective**2 * sqrt_1m_alphas_cumprod[t_y]**2) * torch.randn_like(y)                    

            # --------------------------------
            # (4) main iterations
            # --------------------------------

            progress_img = []
            # create sequence of timestep for sampling
            if skip_type == 'uniform':
                seq = [i*skip for i in range(iter_num)]
                if skip > 1:
                    seq.append(num_train_timesteps-1)
            elif skip_type == "quad":
                seq = np.sqrt(np.linspace(0, num_train_timesteps**2, iter_num))
                seq = [int(s) for s in list(seq)]
                seq[-1] = seq[-1] - 1
            progress_seq = seq[::max(len(seq)//10,1)]
            if progress_seq[-1] != seq[-1]:
                progress_seq.append(seq[-1])

            chain_x = []
            count = 0

            for k_s in tqdm(range(0,total_samples)):

                x = torch.randn_like(y) 

                # reverse diffusion for one image from random noise
                for i in range(len(seq)):
                    curr_sigma = sigmas[seq[i]].cpu().numpy()
                    # time step associated with the noise level sigmas[i]
                    t_i = utils_model.find_nearest(reduced_alpha_cumprod,curr_sigma)
                    # skip iters
                    if t_i > t_start:
                        continue
                    for u in range(iter_num_U):
                        # --------------------------------
                        # step 1, reverse diffsuion step
                        # --------------------------------

                        # solve equation 6b with one reverse diffusion step
                        x0 = utils_model.model_fn(x, noise_level=curr_sigma*255, model_out_type=model_out_type, \
                                model_diffusion=model, diffusion=diffusion, ddim_sample=ddim_sample, alphas_cumprod=alphas_cumprod)

                        # --------------------------------
                        # step 2, FFT
                        # --------------------------------

                        if seq[i] != seq[-1]:
                            if generate_mode == 'DiffPIR':
                                
                                x0_p = x0 / 2 + 0.5
                                shape_x0 = list(x0_p.size())
                                # K = forward_operator(x, task_current, M = mask, shape_in=shape_x0)
                                # KT = backward_operator(x, task_current, M = mask, shape_in=shape_x0)

                                # obj = Objectives("geometric", K, KT, x0_p.ravel(), y.ravel(), alpha_geo, b_geo, rhos[t_i], task_current)
                                # der = Derivatives("geometric", K, KT, x0_p.ravel(), y.ravel(), alpha_geo, b_geo, rhos[t_i], task_current)
                                # b_geo_model =  torch.tensor(0.0001).to(device)
                                obj = Objectives("geometric", task_current, x0_p.ravel(), shape_x0, y.ravel(), alpha_geo, b_geo, rhos[t_i], M = mask)
                                der = Derivatives("geometric", task_current, x0_p.ravel(), shape_x0, y.ravel(), alpha_geo, b_geo, rhos[t_i], M = mask)
                                # obj = Objectives("geometric", task_current, x0_p.ravel(), shape_x0, y.ravel(), alpha_geo, b_geo, rhos[t_i])
                                # der = Derivatives("geometric", task_current, x0_p.ravel(), shape_x0, y.ravel(), alpha_geo, b_geo, rhos[t_i])
                                x0_p = utils_model.LBFGSB(obj.objective, der.derivative, x0_p.ravel(), shape_x0, tol)
                                x0_p = torch.from_numpy(x0_p * 2 - 1).float().to(device)
                                x0 = x0 + guidance_scale * (x0_p-x0)

                                pass                               

                        if (generate_mode == 'DiffPIR' and model_out_type == 'pred_xstart') and not (seq[i] == seq[-1] and u == iter_num_U-1):
                            #x = sqrt_alphas_cumprod[t_i] * (x0) + (sqrt_1m_alphas_cumprod[t_i]) *  torch.randn_like(x)
                            
                            t_im1 = utils_model.find_nearest(reduced_alpha_cumprod,sigmas[seq[i+1]].cpu().numpy())
                            # calculate \hat{\eposilon}
                            eps = (x - sqrt_alphas_cumprod[t_i] * x0) / sqrt_1m_alphas_cumprod[t_i]
                            eta_sigma = eta * sqrt_1m_alphas_cumprod[t_im1] / sqrt_1m_alphas_cumprod[t_i] * torch.sqrt(betas[t_i])
                            x = sqrt_alphas_cumprod[t_im1] * x0 + np.sqrt(1-zeta) * (torch.sqrt(sqrt_1m_alphas_cumprod[t_im1]**2 - eta_sigma**2) * eps \
                                        + eta_sigma * torch.randn_like(x)) + np.sqrt(zeta) * sqrt_1m_alphas_cumprod[t_im1] * torch.randn_like(x)
                        else:
                            # x = x0
                            pass
                            
                        # set back to x_t from x_{t-1}
                        if u < iter_num_U-1 and seq[i] != seq[-1]:
                            # x = torch.sqrt(alphas[t_i]) * x + torch.sqrt(betas[t_i]) * torch.randn_like(x)
                            sqrt_alpha_effective = sqrt_alphas_cumprod[t_i] / sqrt_alphas_cumprod[t_im1]
                            x = sqrt_alpha_effective * x + torch.sqrt(sqrt_1m_alphas_cumprod[t_i]**2 - \
                                    sqrt_alpha_effective**2 * sqrt_1m_alphas_cumprod[t_im1]**2) * torch.randn_like(x)

                    # save the process
                    x_0 = (x/2+0.5)

                if k_s == 0:
                    post_meanvar = utils_welford.welford(x_0)
                else:
                    post_meanvar.update(x_0)

                if count == thinning_step-1:
                    chain_x.append(x_0.detach().cpu().numpy())
                    count = 0
                else:
                    count += 1 

            # --------------------------------
            # (3) img_E
            # --------------------------------

                if k_s >= 0:

                    img_E = util.tensor2uint(post_meanvar.get_mean())
                    psnr = util.calculate_psnr(img_E, img_H, border=border)  # change with your own border
                    test_results['psnr'].append(psnr)
            
                    if calc_LPIPS:
                        img_H_tensor = np.transpose(img_H, (2, 0, 1))
                        img_H_tensor = torch.from_numpy(img_H_tensor)[None,:,:,:].to(device)
                        img_H_tensor = img_H_tensor / 255 * 2 -1
                        lpips_score = loss_fn_vgg(post_meanvar.get_mean().detach().detach()*2-1, img_H_tensor)
                        lpips_score = lpips_score.cpu().detach().numpy()[0][0][0][0]
                        test_results['lpips'].append(lpips_score)
                        ssim_score = ssim(img_E, img_H, full=False, multichannel=True)
                        logger.info('{:->4d}, iter {}/{}--> {:>10s} PSNR: {:.4f}dB SSIM: {:.4f} LPIPS: {:.4f} ave LPIPS: {:.4f}'.format(idx+1, k_s, total_samples, img_name+ext, psnr, ssim_score, lpips_score, sum(test_results['lpips']) / len(test_results['lpips'])))
                    else:
                        logger.info('{:->4d}--> {:>10s} PSNR: {:.4f}dB'.format(idx+1, img_name+ext, psnr))

                    if n_channels == 1:
                        img_H = img_H.squeeze()

                    if save_E:
                        # util.imsave(img_E, os.path.join(E_path, img_name+'_'+model_name+ext))
                        util.imsave(img_E, os.path.join(E_path, 'recon_n', fname))

                # if  k_s > 0:
                #     img_H_tensor = torch.from_numpy(np.transpose(img_H, (2, 0, 1))/255)[:,:,:].to(device)
                #     save_plots(img_H_tensor, x_0, post_meanvar, img_name, E_path)
                    
                # if (k_s+1)%20==0:
                #     save_progress(x_0, post_meanvar, chain_x, img_name, E_path)
                                                                            
            # --------------------------------
            # (4) img_LEH
            # --------------------------------

            if save_LEH:
                img_L = img_L/np.max(img_L)#np.log(img_L+1)/np.max(np.log(img_L+1))
                img_L = util.single2uint(img_L)
                k_v = k/np.max(k)*1.0
                k_v = util.single2uint(np.tile(k_v[..., np.newaxis], [1, 1, 3]))
                k_v = cv2.resize(k_v, (3*k_v.shape[1], 3*k_v.shape[0]), interpolation=cv2.INTER_NEAREST)
                img_I = cv2.resize(img_L, (sf*img_L.shape[1], sf*img_L.shape[0]), interpolation=cv2.INTER_NEAREST)
                img_I[:k_v.shape[0], -k_v.shape[1]:, :] = k_v
                img_I[:img_L.shape[0], :img_L.shape[1], :] = img_L
                util.imshow(np.concatenate([img_I, img_E, img_H], axis=1), title='LR / Recovered / Ground-truth') if show_img else None
                util.imsave(np.concatenate([img_I, img_E, img_H], axis=1), os.path.join(E_path, img_name+'_LEH'+ext))

            if save_L:
                img_L = img_L/np.max(img_L)
                # util.imsave(util.single2uint(img_L), os.path.join(E_path, img_name+'_LR'+ext))
                util.imsave(util.single2uint(img_L), os.path.join(E_path, 'input', fname))
                util.imsave(img_H, os.path.join(E_path, 'label', fname))

        # --------------------------------
        # Average PSNR and LPIPS
        # --------------------------------
        
        if total_samples == 1 :

            ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
            logger.info('------> Average PSNR of ({}): {:.4f} dB'.format(testset_name, ave_psnr))


            if calc_LPIPS:
                ave_lpips = sum(test_results['lpips']) / len(test_results['lpips'])
                logger.info('------> Average LPIPS of ({}): {:.4f}'.format(testset_name, ave_lpips))

    
    # experiments
    
    lambdas = [lambda_*i for i in range(2,3)]  # 3 imagenet, 2 ffhq
    zetas = [0.7]#[zeta*i for i in range(0.65)]   # 0.73 imagenet, 0.7 ffhq
    for lambda_ in lambdas:
        for zeta_i in zetas:
            test_rho(lambda_, zeta=zeta_i, model_output_type=model_output_type)

if __name__ == '__main__':

    main()
