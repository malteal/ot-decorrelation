"Flow decorrelation"
import os
import sys
from tqdm import tqdm
from datetime import datetime
from glob import glob

import pandas as pd
import numpy as np
import torch as T
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

import pipeline as pl
import utils

from otcalib.torch.utils import utils, loader
from otcalib.torch.utils.utils import save_config
from tools.schedulers import get_scheduler
from tools import misc, scalers

from tools.flows import Flow

style_target = {"marker":"o", "color":"black", "label":"Uniform",
                "linewidth":0, "markersize":4}
style_source = {"linestyle": "dotted","color":"blue", "label":"Source",
                "drawstyle":"steps-mid"}
style_trans = {"linestyle": "dashed", "color":"red", "label":"Transport",
                "drawstyle":"steps-mid"}

@hydra.main(config_path="config", config_name="flow_config", version_base=None)
def main(config: DictConfig):
    #  %matplotlib widget

    # define data
    output = pl.load_data(config.model_config.xz_dim,
                          multi_clf=config.get("multi_clf", None),
                          path_to_clf=config.path_to_clf)
    
    # init model
    config.model_config.xz_dim = (config.model_config.xz_dim-config.model_config.drop_dim)
    flow = Flow(flow_config=config.model_config,
                train_config=config.train_config,
                save_path=config.save_path,
                add_time_to_dir=False,
                device=config.device)

    print(f"Saving at {config.save_path}")

    # preprocess data
    if (config.path_to_clf is not None) and (not config.multi_cls):
        bkg_mask = output["labels"]==0
        output = {i: output[i][bkg_mask] for i in output}
    if config.model_config.drop_dim>0:
        output["encodings"] = output["encodings"][:,:-config.model_config.drop_dim]
    if config.logit:
        output["encodings"] = utils.logit(output["encodings"])
        output["encodings"] = np.clip(output["encodings"], -15, 15)
    else:
        output["encodings"] = np.clip(output["encodings"], 0,1)
        if config.model_config.xz_dim>1:
            output["encodings"] =  output["encodings"]/np.sum( output["encodings"],1)[:,None]
        output["encodings"] = output["encodings"][:,:config.model_config.xz_dim]

    if config.conds_trans=="log":
        output["mass"] = np.log(output["mass"])

    data = np.c_[output["mass"], output["encodings"]]
    if config.maxevents is not None:
        data = data[np.random.choice(np.arange(len(data)), config.maxevents, replace=False)]
    

    # scaler
    bounds = config.model_config.rqs_kwargs.get("tail_bound",None)
    if bounds is not None:
        minmax = scalers.norm_type(name="minmax", args={"feature_range":[-bounds,bounds]})
        data = T.tensor(minmax.fit_transform(data), requires_grad=True).float()
        scalers.save_scaler(minmax, f"{config.save_path}/minmax_scaler.pkl")
    else:
        minmax = scalers.norm_type(name="minmax", args={"feature_range":[0,1],
                                                        "clip":True})
        data = T.tensor(minmax.fit_transform(data), requires_grad=True).float()
        scalers.save_scaler(minmax, f"{config.save_path}/minmax_scaler.pkl")


    if isinstance(config.valid_size, str):
        train, test = train_test_split(data.detach(), test_size=200_000)
    else:
        train, test = data[config.valid_size:], data[:config.valid_size]

    print(f"Training size {train.shape}")
    print(f"Test size {test.shape}")
    
    # sys.exit()
    # dataloaders
    train_dataset = T.utils.data.DataLoader(train.cpu().detach(), batch_size=512,
                                            shuffle=True, pin_memory=True)
    test_dataset = T.utils.data.DataLoader(test.cpu().detach(), batch_size=512,
                                           shuffle=True, pin_memory=True)

    flow.create_plotting(test[:,1:], test[:,:1])
    # sys.exit()
    #init scheduler
    sch_attr = {"T_max":flow.train_config["n_epochs"]*len(train_dataset),
                "eta_min": 0.000001}
    flow.scheduler = get_scheduler("singlecosine", flow.optimizer, attr=sch_attr)

    # plotting the training data
    training_data = T.concat([i for i in train_dataset], 0).detach().numpy()
    for i in range(training_data.shape[1]):
        fig = plt.figure()
        plt.hist(training_data[:,i], bins=75, histtype="step", lw=2, density=True)
        misc.save_fig(fig, f"{config.save_path}/figures/training_{i}.png")

    # train flow
    flow.train(train_dataloader=train_dataset,
               valid_dataloader=test_dataset)

if __name__ == "__main__":
    main()
