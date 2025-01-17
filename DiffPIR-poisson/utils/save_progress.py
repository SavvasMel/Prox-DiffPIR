import joblib
import torch
import numpy as np

def save_progress(x, post_meanvar, chain, img_name, path):

    data = {
    "sample": x.cpu().numpy(),
    "meanSamples": post_meanvar.get_mean().cpu().numpy(),
    "variance": post_meanvar.get_var().cpu().numpy()}

    with open(path + '/' + img_name + '_data_ddrm.joblib', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        joblib.dump(data, f)

    with open(path + '/' + img_name + '_chain_ddrm.joblib', 'wb') as f:
        joblib.dump({'MC_chain' : chain}, f)