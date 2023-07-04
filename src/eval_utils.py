"plotting utils"
import os
import numpy as np
import matplotlib.pyplot as plt

import torch as T

import src.utils as utils
FIG_SIZE = (6, 6)
FONTSIZE = 16
LABELSIZE = 16
LEGENDSIZE = 14

def plot_training_setup(source_values:iter, target_values:iter, outdir:str,
                        eval_data:dict=None, generator=None,
                        **kwargs):
    "What a mess!"

    # get iterator data for training
    os.makedirs(f"{outdir}/training_dist", exist_ok=True)
    if not isinstance(source_values, T.Tensor):
        source_values = T.concat([T.concat(next(source_values),1).cpu()
                                        for _ in range(kwargs.get("n_samples", 200))],0)
    if not isinstance(target_values, T.Tensor):
        target_values = T.concat([T.concat(next(target_values),1).cpu()
                                        for _ in range(kwargs.get("n_samples", 200))],0)
    
    # get transport if generator defined
    mask_sig_source = source_values[:,-1] ==1
    mask_sig_target = target_values[:,-1] ==1
    if generator is not None:
        cvx_dim = generator.weight_zz[0].shape[-1]
        noncvx_dim = generator.weight_uutilde[0].shape[-1]
        transport = generator.chunk_transport(
            source_values[:,:noncvx_dim],
            source_values[:,noncvx_dim:noncvx_dim+cvx_dim],
            sig_mask = mask_sig_source, 
            n_chunks=2, scaler=kwargs.get("scaler", None))

        transport = T.concat([source_values[:,:noncvx_dim].cpu(),
                                transport,
                                source_values[:,-1:].cpu()], 1)
        
        transport = transport.cpu().detach().numpy()
    
    mask_sig_source = mask_sig_source.cpu().detach().numpy()
    mask_sig_target = mask_sig_target.cpu().detach().numpy()
    source_values = source_values.cpu().detach().numpy()
    target_values = target_values.cpu().detach().numpy()

    # training iterator data
    style = {"bins": kwargs.get("n_bins", 40), "histtype": "step", "density":True, "stacked":True}
    plot_var = kwargs.get("plot_var", ["pT", "pb", "pc", "pu"])
    dist_labels = kwargs.get("dist_labels", ["Source", "Target", "Transport"])
    train_ranges = []
    for nr, i in enumerate(plot_var):
        fig = plt.figure()
        x_range = np.percentile(target_values[:,nr], [0, 100])
        train_ranges.append(x_range)
        style["range"] = x_range
        if generator is not None:
            _, bins , _ = plt.hist([transport[:,nr][~mask_sig_source],
                                    transport[:,nr][mask_sig_source]],
                                    label=[f"{dist_labels[2]} background",
                                           f"{dist_labels[2]} signal"],
                                    **style)
        _, bins , _ = plt.hist([source_values[:,nr][~mask_sig_source],
                                source_values[:,nr][mask_sig_source]],
                                label=[f"{dist_labels[0]} background",
                                       f"{dist_labels[0]} signal"],
                                **style)
        _, bins , _ = plt.hist([target_values[:,nr][~mask_sig_target],
                                target_values[:,nr][mask_sig_target]],
                                label=[f"{dist_labels[1]} background",
                                       f"{dist_labels[1]} signal"], **style)
        plt.xlabel(f"Trainable {i}")
        plt.ylabel("#")
        plt.legend(frameon=False)
        plt.tight_layout()
        # plt.yscale("log")
        save_fig(f"{outdir}/training_dist/training_{i}.png")

    if isinstance(eval_data, dict):
        # plot eval distributions
        for name, values in eval_data.items():
            nr=0
            for keys_to_plot in ["conds", "transport"]:
                shape = values[list(values.keys())[0]][keys_to_plot].shape[1]
                for col in range(shape):
                    fig = plt.figure()
                    style = {"bins": kwargs.get("n_bins", 40), "histtype": "step",
                             "density":True, "lw": 1.4}
                    style["range"] = train_ranges[nr]
                    for sub_keys in values.keys():
                        col_values = values[sub_keys][keys_to_plot][:,col].detach().numpy()
                        sig_mask = values[sub_keys]["sig_mask"].detach().numpy()==1
                        if "truth" in sub_keys:
                            style["ls"] = "dashed"
                            style["zorder"] = 100
                        else:
                            style.pop("ls", None)
                            style.pop("zorder", None)
                        _, bins , _ = plt.hist([col_values[~sig_mask],
                                                col_values[sig_mask]],
                                                label=[f"{sub_keys} background",
                                                       f"{sub_keys} signal"],
                                                stacked=True,  **style)
                        style["bins"] = bins
                    if col == 0:
                        plt.xlabel(f"log(pT)")
                    else:
                        plt.xlabel(f"{plot_var[nr]}")
                    plt.ylabel("#")
                    plt.legend(frameon=False)
                    plt.tight_layout()
                    # plt.yscale("log")
                    utils.save_fig(f"{outdir}/training_dist/eval_{name}_{plot_var[nr]}.png")
                    nr+=1

