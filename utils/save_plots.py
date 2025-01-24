import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

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

def save_plots(orig, x, post_meanvar, img_name, path):

    def rgb(t): return (np.clip((t[0] if len(t.shape) == 4 else t).detach().cpu().numpy().transpose([1, 2, 0]), 0, 1) * 255).astype(np.uint8)

    var_np = post_meanvar.get_var()[0].detach().cpu().numpy()
    var_grayscale = np.zeros([x.shape[1], x.shape[2]])
    res_np = (post_meanvar.get_mean().cpu().numpy()[0]-orig.cpu().numpy())**2
    res_grayscale = np.zeros([x.shape[1], x.shape[2]])
        
    Image.fromarray(rgb(x)).save(os.path.join(path, "sample_{}.png".format(img_name)))
    Image.fromarray(rgb(post_meanvar.get_mean())).save(os.path.join(path, "post_mean_{}.png".format(img_name)))

    res_grayscale = 0.299*res_np[0] + 0.587*res_np[1] + 0.114*res_np[2]
    norm_mean = matplotlib.colors.Normalize(vmin=np.min(np.sqrt(res_grayscale)), vmax=np.max(np.sqrt(res_grayscale)))

    fig, ax = plt.subplots()
    im = ax.imshow(np.sqrt(res_np.transpose([1,2,0])))
    plt.title("Residuals (squared-root)")
    plt.colorbar(im)
    plt.savefig(os.path.join(path, "residuals_{}.png".format(img_name)), bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots()
    im = ax.imshow(np.sqrt(res_grayscale), cmap = "gray", norm = norm_mean)
    plt.title("Residuals (squared-root, grayscaled)")
    plt.colorbar(im)
    plt.savefig(os.path.join(path, "residuals_gray_{}.png".format(img_name)), bbox_inches='tight')
    plt.close()

    var_grayscale = 0.299*var_np[0] + 0.587*var_np[1] + 0.114*var_np[2]
    norm_mean = matplotlib.colors.Normalize(vmin=np.min(np.sqrt(var_grayscale)), vmax=np.max(np.sqrt(var_grayscale)))

    fig, ax = plt.subplots()
    im = ax.imshow(np.clip(np.sqrt(var_np.transpose([1,2,0]))*4,0,1))
    plt.title("Posterior st. deviation (scaled by 4)")
    plt.colorbar(im)
    plt.savefig(os.path.join(path, "post_stdev_{}_scaled_4.png".format(img_name)), bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots()
    im = ax.imshow(np.sqrt(var_np.transpose([1,2,0])))
    plt.title("Posterior st. deviation")
    plt.colorbar(im)
    plt.savefig(os.path.join(path, "post_stdev_{}.png".format(img_name)), bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots()
    im = ax.imshow(np.sqrt(var_grayscale), cmap = "gray", norm = norm_mean)
    plt.title("Posterior st. deviation (grayscaled)")
    plt.colorbar(im)
    plt.savefig(os.path.join(path, "post_stdev_gray_{}.png".format(img_name)), bbox_inches='tight')
    plt.close()
