"utils"
from glob import glob
import os
import matplotlib.pyplot as plt

from scipy.stats import entropy
import numpy as np
import torch as T
import pandas as pd
import h5py
import yaml
import json
from omegaconf import OmegaConf


#internal
from src.PICNN.PICNN import PICNN
# from tools.flows import Flow

STYLE_TARGET = {"marker":"o", "color":"black", "label":"Base",
                "linewidth":0, "markersize":4}
STYLE_SOURCE = {"linestyle": "dotted","color":"blue", "label":"Source",
                "drawstyle":"steps-mid"}
STYLE_TRANS = {"linestyle": "dashed", "color":"red", "label":"Transport",
                "drawstyle":"steps-mid"}

replace_symbols = {":": "", " ": "", "$": "", "-":"_", ",":"_", "\\": ""}

def load_yaml(path, hydra_bool:bool=True):
    if hydra_bool:
        data = OmegaConf.load(path)
        OmegaConf.set_struct(data, False)
    else:
        with open(path, 'r') as stream:
            data = yaml.safe_load(stream)
    return data

def load_hdf(filename):
    return h5py.File(filename,'r')

def load_json(name):
    with open(name, "r") as fp:
        data = json.load(fp)
    return data

def save_yaml(dict, name, hydra=True):
    """save data as yaml    

    Parameters
    ----------
    dict : data
        data that should be saved
    name : str
        path.yml where it should be saved
    """
    if hydra:
        # dumps to file:
        with open(name, "w") as f:
            OmegaConf.save(dict, f)
    else:
        with open(name, 'w') as outfile:
            yaml.dump(dict, outfile, default_flow_style=False)


def logit(prob):
    if isinstance(prob, T.Tensor):
        return T.log(prob/(1-prob+10e-10))
    else:
        return np.log(prob/(1-prob+10e-10))

def probsfromlogits(logitps: np.ndarray) -> np.ndarray:
    """reverse transformation from logits to probs

    Parameters
    ----------
    logitps : np.ndarray
        arrray of logit

    Returns
    -------
    np.ndarray
        probabilities from logit
    """
    norm=1
    if isinstance(logitps, T.Tensor):
        ps_value = 1.0 / (1.0 + T.exp(-logitps))
        if (ps_value.shape[-1]>1) and (len(ps_value.shape)>1):
            norm = T.sum(ps_value, axis=1)
            norm = T.stack([norm] * logitps.shape[1]).T
    else:
        ps_value = 1.0 / (1.0 + np.exp(-logitps))
        if (ps_value.shape[-1]>1) and (len(ps_value.shape)>1):
            norm = np.sum(ps_value, axis=1)
            norm = np.stack([norm] * logitps.shape[1]).T
    return ps_value / norm



def load_flow_model(path:str, device= "cpu"):
    if len(glob(f"{path}/*"))==0:
        raise ValueError(f"{path} is empty")
    try:
        model_args = load_yaml(glob(f"{path}/flow*.yaml")[0])
    except:
        model_args = load_yaml(glob(f"{path}/model*.yaml")[0])
    train_args = load_yaml(f"{path}/train_config.yaml")
    # model_args = load_yaml(f"{path}/flow_config.yaml")
    model_args.do_lu =model_args.xz_dim>2
    model_args.logit =model_args.xz_dim>2
    try:
        flow = Flow(flow_config=model_args,
                    train_config=train_args,
                    add_time_to_dir=False,
                    device=device)
        flow.load_old_model(path)
    except:
        model_args.end_do_lu = True
        flow = Flow(flow_config=model_args,
                    train_config=train_args,
                    add_time_to_dir=False,
                    device=device)
        flow.load_old_model(path)
    return flow.flow, model_args

def load_ot_model(path:str, verbose=True, device= "cpu"):
    "load ot model given a path to model"
    model_args = load_yaml(f"{path}/model_config.yml")
    train_args = load_yaml(f"{path}/train_config.yml")
    log = load_json(glob(f"{path}/log.*")[-1])
    eval_col =  None #"AUC" if "source" in path else "log_likelihood_eval"#"source_average_wasserstein"#
    if (eval_col is not None) & (eval_col in log):
        performance_metric = log[eval_col]
        model_index =np.argmin(performance_metric)
    else:
        model_index =-1

    model_args["device"]=device
    model_args["cvx_dim"]=train_args["cvx_dim"]
    model_args["noncvx_dim"]=train_args["noncvx_dim"]
    model_args["verbose"] = verbose
    w_disc = PICNN(**model_args)
    generator = PICNN(**model_args)
    path_to_best_model = sorted(glob(f"{path}/training_setup/*"),
                                key=os.path.getmtime)[model_index]
    if verbose:
        print(path_to_best_model.split("/")[-1])
    parameters = T.load(path_to_best_model, map_location=model_args["device"])
    w_disc.load_state_dict(parameters["f_func"])
    generator.load_state_dict(parameters["g_func"])
    return w_disc, generator

def softmax(x):
    exp = np.clip(np.exp(x), -20, 20)
    return exp/np.sum(exp,1).reshape(-1,1).astype(float)

def JSD(hist1, hist2):
    return 0.5 * (entropy(hist1, 0.5 * (hist1 + hist2)) + entropy(hist2, 0.5 * (hist1 + hist2)))

def find_threshold(L, mask, x_frac):
    """
    From sam: 
    Calculate c such that x_frac of the array is less than c.

    Parameters
    ----------
    L : Array
        The array where the cutoff is to be found
    mask : Array,
        Mask that returns L[mask] the part of the original array over which it is desired to calculate the threshold.
    x_frac : float
        Of the area that is lass than or equal to c.

    returns c (type=L.dtype)
    """
    max_x = mask.sum()
    x = int(np.round(x_frac * max_x))
    L_sorted = np.sort(L[mask.astype(bool)])
    return L_sorted[x]

def get_disc_func(sig_label):
    def discriminant(dist):
        if sig_label==0:
            return dist[:,0]/(dist[:,1]+dist[:,2])
        elif sig_label==1:
            return dist[:,1]/(dist[:,0]+dist[:,2])
        elif sig_label==2:
            return dist[:,2]/(dist[:,1]+dist[:,0])
    return discriminant

def check_intersection(vectors):
    # Extract the start and end coordinates of the vectors
    vector_start = vectors[:, 0:2]
    vector_end = vectors[:, 2:4]
    vector_check_start = vectors[:, 4:6]
    vector_check_end = vectors[:, 6:8]

    # Calculate the directions of the vectors
    vector_dir = vector_end - vector_start
    vector_check_dir = vector_check_end - vector_check_start
    
    # Calculate the cross product of the vectors
    cross_product = np.cross(vector_dir,vector_check_dir)

    # Check if the cross product is zero (parallel vectors)
    parallel_mask = cross_product < 1e-6

    # Calculate the intersection point for non-parallel vectors
    vector_diff = vector_check_start - vector_start
    t = np.cross(vector_diff, vector_check_dir) / cross_product
    intersection_point = vector_start + t[:, np.newaxis] * vector_dir

    
    # Check if the intersection point lies within both sets of boundary points
    within_boundaries_mask = (intersection_point >= np.minimum(vector_start, vector_end)) \
                           & (intersection_point <= np.maximum(vector_start, vector_end)) \
                           & (intersection_point >= np.minimum(vector_check_start, vector_check_end)) \
                           & (intersection_point <= np.maximum(vector_check_start, vector_check_end))

    
    is_subset_mask = parallel_mask & (
        np.all(vector_start == vector_check_start, axis=1) |
        np.all(vector_end == vector_check_end, axis=1)
    )

    dot_product = np.sum(vector_dir * vector_check_dir, axis=1)
    cosine_sim = np.arccos(dot_product / (np.sqrt(np.sum(vector_dir**2, axis=1)) * np.sqrt(np.sum(vector_check_dir**2, axis=1))))

    return parallel_mask, within_boundaries_mask.all(axis=-1), is_subset_mask, cosine_sim

def save_fig(fig, save_path, title:str=None, close_fig=True,
             save_args={}, replace_symbols_bool=True, **kwargs):
    if replace_symbols_bool:
        save_path = save_path.split("/")
        for i,j in replace_symbols.items():
            save_path[-1] = save_path[-1].replace(i, j)
        save_path = "/".join(save_path)
    if title is not None:
        plt.title(title)
    if kwargs.get("tight_layout", True):
        plt.tight_layout()
    plt.savefig(save_path, dpi=500, facecolor='white', transparent=False, **save_args)
    if close_fig:
        plt.close(fig)

def save_config(outdir: str, values: dict, drop_keys: list, file_name: str):
    """save the model and training config files and helps with creating folders

    Parameters
    ----------
    outdir : str
        path to the output folder
    values : dict
        config values in dict form - also the function will remove all drop_keys
    drop_keys : list
        drop_keys should all non-saveable parameters in values
    file_name : str
        Name of the new saved file
    """
    for folder in ["", "models", "plots"]:
        if not os.path.exists(outdir + "/" + folder):
            os.mkdir(outdir + "/" + folder)
    for drop in drop_keys:
        values.pop(drop, None)

    save_yaml(values, f"{outdir}/{file_name}.yml", hydra=True)
