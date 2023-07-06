"plotting utils"
import os
import numpy as np
import matplotlib.pyplot as plt

import torch as T

import src.utils as utils
import src.plotting as plot

FIG_SIZE = (6, 6)
FONTSIZE = 16
LABELSIZE = 16
LEGENDSIZE = 14

def hist_in_bins(*distributions, conds, binning, style={"range": [0,1], "bins": 20},
                 legend_title=None):
    fig, ax = plt.subplots(1,len(distributions), figsize=(8*len(distributions),6))
    ax = np.ravel(ax)
    dist_styles = [{"alpha":0.5,
                    "lw": 2,
                    }]
    mask_conds = (binning[0]<conds)
    label = rf"${binning[0]} < m$"
    # full dist
    for nr, dist in enumerate(distributions):
        dist_styles[0]["label"]= label
        dist_styles[0]["color"]= "black"
        counts, _ = plot.plot_hist(dist[mask_conds], ax=ax[nr],
                                    dist_styles=dist_styles,
                                    style=style)
        style["bins"] = counts["bins"]
    dist_styles[0].pop("color")

    # slices in conds
    for low_bin, high_bin in zip(binning[:-1], binning[1:]):
        mask_conds = (low_bin<=conds) & (conds<high_bin)
        label = rf"${low_bin} < m$"
        if high_bin!=binning[-1]:
            label+=rf"$ < {high_bin}$"
        for nr, dist in enumerate(distributions):
            dist_styles[0]["label"]= label
            counts, _ = plot.plot_hist(dist[mask_conds], ax=ax[nr],
                                       dist_styles=dist_styles,
                                       style=style,
                                       legend_kwargs={"title": legend_title,
                                                      "loc": "center"})
            style["bins"] = counts["bins"]

    for i in range(len(distributions)):
        ax[i].set_xlabel("Probability", fontsize=plot.FONTSIZE)
    return fig

class EvalauteFramework:
    "evaluation metric for decorrelation"
    def __init__(self, conds, bkg_label=0, sig_label=1, **kwargs) -> None:
        self.plot_bool=kwargs.get("plot_bool", False)
        self.sig_label=sig_label
        self.bkg_label=bkg_label
        self.conds = conds
        self.save_path = kwargs.get("save_path", None)
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
        self.verbose = kwargs.get("verbose", True)
        self.fig_type = kwargs.get("fig_type", ".pdf")
        self.generator=None
        self.w_dist=None


    def run_ot(self, ot_path, col_name="predictions", device="cpu",
               reverse_bool=False):
        del self.generator
        del self.w_dist
        self.w_dist, self.generator = utils.load_ot_model(ot_path, device=device,
                                                             )

        # init transport
        conds = T.tensor(np.log(self.conds), requires_grad=True).float().reshape(-1,1)

        if len(self.output[self.clf_col].shape)==1:
            totransport = self.output[self.clf_col].reshape(-1,1)
        else:
            totransport = self.output[self.clf_col]
        totransport = utils.logit(T.tensor(totransport, requires_grad=True).float())

        self.output[col_name] = self.ot_decorrelate(totransport, conds,
                                                    reverse_bool)

    def run_flow(self, flow_path, col_name="predictions", device="cpu",
                 reverse_bool=False):
        
        self.flow, flow_config = utils.load_flow_model(flow_path, device=device)
        path_scaler = glob(f"{flow_path}/mi*")[0]
        scaler = scalers.load_scaler(path_scaler)
        # init transport
        if flow_config.logit:
            data = np.c_[np.log(self.conds), utils.logit(self.output[self.clf_col])]
        else:
            if flow_config.xz_dim>1:
                data = np.c_[np.log(self.conds), self.output[self.clf_col][:,:flow_config.xz_dim]]
            else:
                data = np.c_[np.log(self.conds), self.output[self.clf_col]]
        data_scaled = scaler.transform(data)
        if not flow_config.logit:
            data_scaled = np.clip(data_scaled, 0, 1)
            
        data_scaled = T.tensor(data_scaled).float()
        n_chunks = len(data_scaled)//10_000
        n_chunks+= 1 if n_chunks==0 else 0
        if reverse_bool:
            # samples, logabsdet = self._transform.inverse(noise, context=embedded_context)
            proba = [self.flow._transform.inverse(
                j.to(device),i.to(device)
                )[0].cpu().detach().numpy() for i,j in zip(data_scaled[:,:1].chunk(n_chunks),
                                                        data_scaled[:,1:].chunk(n_chunks))]
        else:
            proba = [self.flow.transform_to_noise(
                j.to(device), i.to(device)
                ).cpu().detach().numpy() for i,j in zip(data_scaled[:,:1].chunk(n_chunks),
                                                        data_scaled[:,1:].chunk(n_chunks))]
        proba = np.concatenate(proba,0)
        if flow_config.logit:
            proba = np.exp(proba)/np.sum(np.exp(proba),1)[:, None]
        elif flow_config.xz_dim>1: # create addtional columns if one is removed
            # should pbaly be removed
            last_column = 1-np.sum(proba, 1)
            last_column[last_column<0] = 0
            proba = np.c_[proba, last_column]
            proba = proba/np.sum(proba, 1)[:,None]
        self.output[col_name] = proba
            
    def load_decorrelation(self, path):
        self.output = np.load(f"{path}/decorrelated_output.npy", allow_pickle=True).item()
        self.test = np.load(f"{path}/test_output.npy", allow_pickle=True)
        self.conds=pl.inverse_mass_transform(self.test[:,0])
        
    def redefine_output(self, output:dict, clf_col:str, test:np.ndarray=None,
                            conds:np.ndarray=None):
        "if you want to change the output and test dict and array"
        self.output = output
        self.mask_bkg = self.output["labels"]==self.bkg_label            
        self.clf_col=clf_col
        # self.output["index_low_to_high"] = np.argsort(output[self.clf_col])
        if test is not None:
            self.test = test
        if conds is not None:
            self.conds = conds
        self.jsd_style={"bins": 30, "histtype": "step", "density": False,
               "range": [
                   self.conds.min(),
                   self.conds.max()
                         ]}

    def ot_decorrelate(self, totransport, conds, reverse_bool=False):
        "run OT inference"
        n_chunks = len(conds)//100_000
        n_chunks+= 1 if n_chunks==0 else 0
        if reverse_bool:
            transport = self.w_dist.chunk_transport(conds,totransport,
                                                    T.ones_like(conds).bool(),
                                                    n_chunks).detach().numpy()
        else:
            transport = self.generator.chunk_transport(conds,totransport,
                                                    T.ones_like(conds).bool(),
                                                    n_chunks).detach().numpy()
        
        proba = utils.probsfromlogits(transport)
        return proba
    
    def plot_ruc(self, predictions, truth, name=""):
        "Calculating the roc/AUC of classification"
        tpr, fpr, auc, fig = plot.plot_roc_curve(np.ravel(truth),
                                    predictions,
                                    label=f"{name} ",plot_bool=False
                                    )
        roc_dict = {"tpr":tpr, "fpr":fpr, "auc":auc}
        return roc_dict

    def calculate_binned_roc(self, conds_bins, truth_key="vDNN", output_dict=None,
                             **kwargs):
        "calculate auc binned in mass quantiles"
        if output_dict is None:
            output_dict = self.output

        auc_lst = {}
        for key, values in output_dict.items():
            if (key == "labels") or (key == "index_low_to_high"):
                continue
            if key not in auc_lst:
                auc_lst[key]=[]
            for low_bin, high_bin in zip(conds_bins[:-1], conds_bins[1:]):
                mask_conds = (low_bin<=self.conds) & (self.conds<high_bin)
                roc_dict = self.plot_ruc(values[mask_conds], output_dict["labels"][mask_conds])
                auc_lst[key].extend([roc_dict["auc"]])
        auc_lst["bins"] = conds_bins

        self.auc_lst=auc_lst

        if self.plot_bool:
            return self.plot_binned_roc(truth_key=truth_key, **kwargs)
        
    def plot_binned_roc(self, colors=plot.COLORS, truth_key="vDNN", **kwargs):
        "plot auc of the mass quantiles"
        fig, (ax_1, ax_2) = plt.subplots(
            2, 1, gridspec_kw={"height_ratios": [3, 1]}, figsize=(8, 6), sharex="col"
            )
        
        auc_counts = {key:{"counts": [val]} for key,val in self.auc_lst.items()}
        auc_counts["bins"] = self.auc_lst["bins"]
        dist_styles=[{"marker": ".", "ls":"none", "label": key,
                      "color":col } for key, col in zip(auc_counts, colors)]
        
        counts, ax_1 = plot.plot_hist_1d(auc_counts, normalise=False, ax=ax_1,
                                        dist_styles=dist_styles,
                                        xerr_on_errorbar=True)
        ax_2 = plot.plot_ratio(auc_counts, truth_key=truth_key, ax=ax_2,
                               ylim=kwargs.get("ylim", [0.995,1.005]),
                               normalise=False,
                               styles=[{"color": col} for col in colors],
                               zero_line_unc=False
                               )
        ax_1.set_ylabel("AUC", fontsize=plot.FONTSIZE)
        ax_2.set_xlabel("Mass [GeV]", fontsize=plot.FONTSIZE)
        return fig


    def proba_plot_in_mass_bins(self,columns:list,mass_binning = [0, 70, 100, 1e8]):
        "Simple plot of the probabilities in mass bins"
        for i in columns:
            fig = hist_in_bins(self.output[i][self.mask_bkg], conds=self.conds[self.mask_bkg],
                         binning=mass_binning, legend_title=f"{i} [GeV]")
            if self.save_path is not None:
                utils.save_fig(fig, f"{self.save_path}/prob_correlation_{i}{self.fig_type}")
    
    def ideal_calculation(self, background_rej = [0.5, 0.9, 0.95, 0.99], ax=None,
                          sample_times=50):
        bkg_mask = np.ravel(self.output["labels"]==self.bkg_label)
        counts_truth, _ = plot.plot_hist(self.conds[bkg_mask],
                                            style=self.jsd_style, normalise=True,
                                            # dist_styles=[
                                            # {"label": f"total bkg"},
                                            #     ],
                                            names=["labels"],
                                            plot_bool=False)
        jsd_lst = {"std":[], "mean":[]}
        sig_eff = []
        for bkg_rej in background_rej:
            _jsd_lst=[]
            for nr in range(sample_times):
                resampled_masses = np.random.choice(self.conds[bkg_mask],
                                                    int(len(self.conds[bkg_mask])
                                                        *(1-bkg_rej)),
                                                    replace=False)
                counts_dict, _ = plot.plot_hist(resampled_masses,
                                                style=self.jsd_style,
                                                normalise=True,
                                                plot_bool=False)
                counts_dict["labels"] = counts_truth["labels"].copy()

                _jsd_lst.append(1/utils.JSD(
                    counts_dict["labels"]["counts"][0]/counts_dict["labels"]["counts"][0].sum(),
                    counts_dict["dist_0"]["counts"][0]/counts_dict["dist_0"]["counts"][0].sum())
                                        )
            jsd_lst["std"].append(np.std(_jsd_lst))
            jsd_lst["mean"].append(np.mean(_jsd_lst))
        jsd_lst["mean"] = np.array(jsd_lst["mean"])
        jsd_lst["std"] = np.array(jsd_lst["std"])
        if ax is not None:
            ax.errorbar(background_rej, jsd_lst["mean"],
                            yerr=jsd_lst["std"],
                            label = "Ideal",color="grey", ls="dotted")
        return jsd_lst, sig_eff, background_rej
        

    def bkg_rej_calculation(self,column,background_rej = [0.5, 0.9, 0.95, 0.99],
                            proba_flat=None, legend_kwargs={}):
        "calculate JSD at different bkg rejections"
        
        bkg_mask = self.output["labels"]==self.bkg_label
        sig_labels = np.unique(self.output["labels"])
        sig_labels = sig_labels[sig_labels!=self.bkg_label]
        if proba_flat is None:
            proba_flat = np.ravel(self.output[column])

        probabilities_bins = np.quantile(proba_flat[np.ravel(bkg_mask)], background_rej)
        # style={"bins": 25, "histtype": "step", "density": False,
        #        "range": [self.conds.min(),self.conds.max()], "lw": 2}
        if self.plot_bool:
            fig, (ax_1, ax_2) = plt.subplots(
                2, 1, gridspec_kw={"height_ratios": [3, 1]}, figsize=(8, 6),
                sharex="col")
            ax_2.set_xlabel("Mass [GeV]", fontsize=plot.FONTSIZE)
        else:
            ax_1=None
        
        counts_truth, ax_1 = plot.plot_hist(self.conds[np.ravel(bkg_mask)],
                                            style=self.jsd_style, ax=ax_1, normalise=True,
                                            dist_styles=[
                                            {"label":"Background distribution", "lw": 2,
                                             "color": "black"},], names=["labels"],
                                            plot_bool=self.plot_bool)
        jsd_lst = []
        sig_eff = []
        for nr, prob_bin in enumerate(probabilities_bins):
            # sig_eff
            mask_proba = (proba_flat>=prob_bin) & (np.ravel(bkg_mask))
            sig_eff.append([np.sum(self.output["labels"][proba_flat>=prob_bin]==label)
                           /np.sum(self.output["labels"]==label)
                           for label in sig_labels])

            counts_dict, ax_1 = plot.plot_hist(self.conds[mask_proba],style=self.jsd_style,
                                               ax=ax_1, normalise=True,
                                            dist_styles=[
                                                {"label": f"Background rejection at {background_rej[nr]}", "lw": 2}
                                                ], plot_bool=self.plot_bool,
                                                legend_kwargs=legend_kwargs)
            counts_dict["labels"] = counts_truth["labels"].copy()
            if self.plot_bool:
                ax_2 = plot.plot_ratio(counts_dict, truth_key="labels", ax=ax_2,
                                       ylim=[0.5,1.5], zero_line=nr==0,
                                       legend_bool=False)

            jsd_lst.append(utils.JSD(counts_dict["labels"]["counts"][0]/counts_dict["labels"]["counts"][0].sum(),
                                counts_dict["dist_0"]["counts"][0]/counts_dict["dist_0"]["counts"][0].sum()))

        jsd_lst = np.array(jsd_lst)
        sig_eff = np.array(sig_eff)
        if (self.save_path is not None) & (self.plot_bool):
            utils.save_fig(fig, f"{self.save_path}/bkg_dist_at_diff_rejcts_{legend_kwargs.get('title', '')}{self.fig_type}")
        return jsd_lst, sig_eff, background_rej
 


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
            n_chunks=2
            )

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

