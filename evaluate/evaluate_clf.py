"evaluate"

# global
import sys
import os
from datetime import datetime
from glob import glob

#package
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch as T
import matplotlib.pyplot as plt

#private
import pipeline as pl
import utils


# internal packages
from otcalib.torch.models.PICNN import PICNN
from otcalib.torch.utils import utils as ot_utils
from tools.discriminator import DenseNet
from tools import misc, scalers
from tools.visualization import general_plotting as plot
FIG_TYPE=".pdf"

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
    def __init__(self, conds, bkg_label=0, sig_label=1, **kwargs) -> None:
        self.plot_bool=kwargs.get("plot_bool", False)
        self.sig_label=sig_label
        self.bkg_label=bkg_label
        self.conds = conds
        self.save_path = kwargs.get("save_path", None)
        self.verbose = kwargs.get("verbose", True)
        self.generator=None
        self.w_dist=None


    def run_ot(self, ot_path, col_name="predictions", device="cpu",
               reverse_bool=False, missing_kwargs={"cvx_dim":1,"noncvx_dim":1}):
        del self.generator
        del self.w_dist
        # self.w_dist, self.generator = utils.load_ot_model(ot_path, verbose=self.verbose, device=device)
        self.w_dist, self.generator = ot_utils.load_ot_model(ot_path, device=device,
                                                             missing_kwargs=missing_kwargs)
        standard_path = glob(f"{ot_path}/stand*")
        # init transport
        conds = T.tensor(np.log(self.conds), requires_grad=True).float().reshape(-1,1)

        if len(standard_path)>1:
            standardization = misc.load_yaml(standard_path[0])
            conds = (conds-standardization["conds_mean"])/standardization["conds_std"]
        if len(self.output[self.clf_col].shape)==1:
            totransport = self.output[self.clf_col].reshape(-1,1)
        else:
            totransport = self.output[self.clf_col]
        totransport = ot_utils.logit(T.tensor(totransport, requires_grad=True).float())

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
            
    def save_decorrelation(self, path):
        np.save(f"{path}/decorrelated_output", self.output)
        np.save(f"{path}/test_output", self.test)

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

    def calculate_efficiencies(self, pred, target):
        preds = np.array(pred.tolist()).flatten()
        targets = np.array(target.tolist()).flatten()
        acc = (preds.round() == targets).sum() / targets.shape[0]
        signal_efficiency = ((preds.round() == targets) & (targets == 0)).sum() / (targets == 1).sum()
        background_efficiency = ((preds.round() == targets) & (targets == 1)).sum() / (targets == 0).sum()

        c = utils.find_threshold(preds, (targets == 0), 0.5)
        R50 = 1 / ((preds[targets == 1] < c).sum() / (targets == 1).sum())
                    
        mass = np.array(self.conds.tolist()).flatten()[targets == 1]
        bkg_preds = preds[targets == 1]
        hist1, bins = np.histogram(mass[bkg_preds > c], bins=50, density=True)
        hist2, _ = np.histogram(mass[bkg_preds < c], bins=bins, density=True)
        # Mask out the bad
        mx = (hist1 > 0) & (hist2 > 0)
        hist1 = hist1[mx]
        hist2 = hist2[mx]
        JSD_val = utils.JSD(hist1, hist2)

        n1 = np.sum((targets == 1) & (preds > c))
        n_ones = np.sum(targets == 1)
        JSDs = []
        for i in range(10):
            index = np.random.permutation(n_ones)
            hist1, bins = np.histogram(mass[index[:n1]], bins=50, density=True)
            hist2, _ = np.histogram(mass[index[n1:]], bins=bins, density=True)
            JSDs.extend([utils.JSD(hist1, hist2)])

        result = {"acc": acc,
                  "signalE": signal_efficiency,
                  "backgroundE": background_efficiency,
                  "signalE": signal_efficiency,
                  "JSDs": JSDs,
                  "JSD_val": JSD_val,
                  "R50": R50,
                  }

        return result

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
        
        proba = ot_utils.probsfromlogits(transport)
        return proba
    
    def plot_ruc(self, predictions, truth, name=""):
        "Calculating the roc/AUC of classification"
        try:
            tpr, fpr, auc, fig = plot.plot_roc_curve(np.ravel(truth),
                                        predictions,
                                        label=f"{name} ",plot_bool=False
                                        )
        except:
            tpr=fpr=auc=0
        roc_dict = {"tpr":tpr, "fpr":fpr, "auc":auc}
        return roc_dict
        
    def calculate_binned_eff(self, conds_bins):
        "calculate auc binned in mass quantiles"

        self.sig_eff = {}
        self.bkg_eff = {}
        for key, values in self.output.items():
            if (key == "labels") or (key == "index_low_to_high"):
                continue
            if key not in self.bkg_eff:
                self.bkg_eff[key]=[]
                self.sig_eff[key]=[]
            for low_bin, high_bin in zip(conds_bins[:-1], conds_bins[1:]):
                mask_conds = (low_bin<=self.conds) & (self.conds<high_bin)
                mask_sig_eff = np.ravel(values>0.5) & mask_conds
                mask_bkg_eff = np.ravel(values<0.5) & mask_conds
                
                sig_ef = (np.sum(evaluate.output["labels"][mask_sig_eff]==evaluate.sig_label)
                        /np.sum(evaluate.output["labels"][mask_conds]==evaluate.sig_label))
                bkg_ef = (np.sum(evaluate.output["labels"][mask_bkg_eff]==evaluate.bkg_label)
                        #/np.sum(evaluate.output["labels"][mask_conds]==evaluate.bkg_label)
                        )
                self.sig_eff[key].append(sig_ef)
                self.bkg_eff[key].append(bkg_ef)


    def calculate_binned_roc(self, conds_bins, truth_key="vDNN", **kwargs):
        "calculate auc binned in mass quantiles"

        auc_lst = {}
        for key, values in self.output.items():
            if (key == "labels") or (key == "index_low_to_high"):
                continue
            if key not in auc_lst:
                auc_lst[key]=[]
            for low_bin, high_bin in zip(conds_bins[:-1], conds_bins[1:]):
                mask_conds = (low_bin<=self.conds) & (self.conds<high_bin)
                roc_dict = self.plot_ruc(values[mask_conds], self.output["labels"][mask_conds])
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
                misc.save_fig(fig, f"{self.save_path}/prob_correlation_{i}{FIG_TYPE}")
    
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
            ax_bkg.errorbar(background_rej, jsd_lst["mean"],
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
            misc.save_fig(fig, f"{self.save_path}/bkg_dist_at_diff_rejcts_{legend_kwargs.get('title', '')}{FIG_TYPE}")
        return jsd_lst, sig_eff, background_rej
 

if __name__ == "__main__":
    # %matplotlib widget
    device="cuda"
    run_ot_bool = True
    run_flow_bool = True
    style_bkg_rej = {"ls":"dashed", "lw": 2}
    save_path = "figures/1d"
    # save_path = None
    internal_plot_bool=True
    size=None
    vDNN_label = r"$\mathcal{D}_\mathrm{VB}$"
    # classifier model
    # model_name = "01_22_2023_08_57_22_391332/"

    flow_paths=glob(
        # "/home/users/a/algren/scratch/trained_networks/decorrelation/flow/1d/*",
        # "/home/users/a/algren/scratch/trained_networks/decorrelation/flow/1d/new_flow_2023_06_13_16_23_42_265353_uniform_1_1",
        "/home/users/a/algren/scratch/trained_networks/decorrelation/flow/1d/new_below_300_flow_2023_06_19_13_53_04_513457_uniform_1_1",
    )[:1]

    ot_paths = [
        # "/home/users/a/algren/scratch/trained_networks/decorrelation/OT/1d/gridsearch/OT_2023_05_01_14_03_40_375033_source_1_1/",
        # "/home/users/a/algren/scratch/trained_networks/decorrelation/OT/1d/gridsearch/OT_2023_05_01_14_03_40_819725_base_1_1/",
        
        #1d multi
        # "/home/users/a/algren/scratch/trained_networks/decorrelation/OT/1d/gridsearch_w_disc/OT_2023_06_13_12_06_14_011145_source_1_1/",

        # "/home/users/a/algren/scratch/trained_networks/decorrelation/OT/1d/gridsearch_w_disc/OT_2023_06_15_09_38_54_980630_base_uniform_1_1/",

        #1d below 300 multi
        # "/home/users/a/algren/scratch/trained_networks/decorrelation/OT/1d/gridsearch_w_disc/OT_2023_06_19_13_49_49_214505_source_1_1/",

        # "/home/users/a/algren/scratch/trained_networks/decorrelation/OT/1d/gridsearch_w_disc/OT_2023_06_19_13_49_49_215935_base_uniform_1_1/",
        # # "/home/users/a/algren/scratch/trained_networks/decorrelation/OT/1d/gridsearch_w_disc/OT_2023_06_19_13_49_49_213300_base_uniform_1_1/",

        #1d below 450 multi
        "/home/users/a/algren/scratch/trained_networks/decorrelation/OT/1d/gridsearch_450/OT_2023_06_20_23_03_39_622025_source_1_1/",

        "/home/users/a/algren/scratch/trained_networks/decorrelation/OT/1d/gridsearch_450/OT_2023_06_20_23_03_39_625815_base_uniform_1_1/",
        # "/home/users/a/algren/scratch/trained_networks/decorrelation/OT/1d/gridsearch_w_disc/OT_2023_06_19_13_49_49_213300_base_uniform_1_1/",
        ]

    # model ot model
    evaluate = EvalauteFramework("eval/", plot_bool=internal_plot_bool, save_path=save_path)
    colors = ["blue", "red", "green", "darkorange", "black"]
    path_to_results = [
        # "/home/users/a/algren/scratch/trained_networks/decorrelation/discriminators/decorrelated_inputs_05_16_2023_11_23_17_090815/results_df.h5",
        # "/home/users/a/algren/scratch/trained_networks/decorrelation/discriminators/decorrelated_inputs_05_16_2023_21_56_47_936064/results_df.h5",
        
        # "/home/users/a/algren/scratch/trained_networks/decorrelation/discriminators/dense_04_02_2023_16_49_19_533453/results_df.h5",
        ""
        ]
    n_plots=0
    if True:
        fig_sig, ax_sig = plt.subplots(1,1,figsize=(8,6))
        fig_bkg, ax_bkg = plt.subplots(1,1,figsize=(8,6))
        ax_bkg.set_yscale("log")
        for i, color in zip(path_to_results, colors):
            label = vDNN_label #"vDNN" 
            # if (len(path_to_results)>=1):
            #     label+=" OT-normal" if n_plots==0 else " OT-source" if n_plots==1 else ""
            if i=="":
                # output, data = pl.load_multi_cls("/srv/beegfs/scratch/groups/rodem/anomalous_jets/taggers/supervised/transformer_multiclass/outputs/qcd_scores.h5")
                output, data = pl.load_multi_cls("/srv/beegfs/scratch/groups/rodem/anomalous_jets/taggers/supervised/transformer_multiclass/outputs/combined_scores.h5", upper_mass_cut=450)
                data["w_score"] = 2*(1/(1+np.exp(-utils.get_disc_func(2)(output["encodings"]))))-1
                # output = pd.DataFrame.from_dict(output)
                data = data[data.label!=1]
                results = data[["mass", "label", "w_score"]]
                results = results.rename(columns={"w_score": label, "label": "labels"})
                results["labels"][results["labels"]!=2] = 0
                results["labels"][results["labels"]==2] = 1
                # results = results[results["mass"]< 300]
            else:
                results = pd.read_hdf(i, key='df', index=False).iloc[:size]
                results = results.rename(columns={"encodings": label})
                model_name = i.split("/")[-2].split("_")[-1]
            output = {i: results[i].values for i in results.columns}
            conds = output.pop("mass")
            evaluate.redefine_output(output=output,conds=conds, clf_col=label)
            (jsd_lst_disco, sig_eff_disco,
             background_rej) = evaluate.bkg_rej_calculation(label,
                                                            legend_kwargs={"title":vDNN_label})
            evaluate.proba_plot_in_mass_bins(
                [i for i in evaluate.output.keys() if i not in ["labels", "index_low_to_high"]]
                )
            ax_sig.plot(background_rej, sig_eff_disco, label = label,
                    color=colors[n_plots], ls="solid")
            ax_bkg.plot(background_rej, 1/jsd_lst_disco, label = label,
                    color=colors[n_plots], ls="solid")
            n_plots+=1
        jsd_lst_ideal, _, background_rej = evaluate.ideal_calculation(ax=ax_bkg)
        if run_ot_bool:
            for i in ot_paths:
                name =f"OT-{vDNN_label} {i.split('/')[-2].split('_')[-3]}"
                name = name.replace("uniform", "uniform[0,1]")
                data_dict={"sig_eff":[], "JSD":[]}
                evaluate.run_ot(ot_path=i, device=device, col_name=name)

                jsd_lst_disco, sig_eff, background_rej = evaluate.bkg_rej_calculation(
                    name, legend_kwargs={"title":name})
                data_dict["sig_eff"].append(sig_eff)
                data_dict["JSD"].append(1/jsd_lst_disco)
                evaluate.proba_plot_in_mass_bins([name])
                style_bkg_rej["color"] = colors[n_plots]
                ax_sig.errorbar(background_rej, 
                               np.mean(data_dict["sig_eff"],0),
                            # yerr=np.std(data_dict["sig_eff"]),
                            label = name, **style_bkg_rej)
                
                ax_bkg.errorbar(background_rej,
                               np.mean(data_dict["JSD"],0),
                            # yerr=np.std(data_dict["JSD"],0),
                            label = name,
                            **style_bkg_rej)
                n_plots+=1
                
        T.cuda.empty_cache()
        if run_flow_bool:
            for i in flow_paths:
                if len(flow_paths)==1:
                    name = f"cf-{vDNN_label} uniform[0,1]"
                else:
                    name=f"Flow {i.split('/')[-1].split('_')[-4]}"
                
                data_dict={"sig_eff":[], "JSD":[]}
                evaluate.run_flow(flow_path=i, device=device, col_name=name)

                jsd_lst_disco, sig_eff, background_rej = evaluate.bkg_rej_calculation(name,legend_kwargs={"title":name})
                data_dict["sig_eff"].append(sig_eff)
                data_dict["JSD"].append(1/jsd_lst_disco)
                evaluate.proba_plot_in_mass_bins([name])
                style_bkg_rej["color"] = colors[n_plots]
                ax_sig.errorbar(background_rej, np.mean(data_dict["sig_eff"],0),
                            # yerr=np.std(data_dict["sig_eff"],0),
                            label = name, **style_bkg_rej)
                
                ax_bkg.errorbar(background_rej, np.mean(data_dict["JSD"],0),
                            # yerr=np.std(data_dict["JSD"],0),
                            label = name,
                            **style_bkg_rej)
                n_plots+=1
        for ax,i,j in zip(
            [ax_sig, ax_bkg],
            ["Background rejection", "Background rejection"],
            ["Signal Efficiency", "1/JSD"]):
            ax.set_xlabel(i, fontsize=plot.FONTSIZE)
            ax.set_ylabel(j, fontsize=plot.FONTSIZE)
            ax.legend(prop={"size": plot.LEGENDSIZE},frameon=False,
                            title_fontsize=plot.LEGENDSIZE,
                            loc="center",
                            bbox_to_anchor=(0.5, 0.4)
                            )

            ax.tick_params(axis="both", which="major", labelsize=plot.LABELSIZE)
        plt.tight_layout()
        if save_path is not None:
            misc.save_fig(fig_bkg, f"{save_path}/1d_bkg_rej{FIG_TYPE}")
            misc.save_fig(fig_sig, f"{save_path}/1d_sig_eff{FIG_TYPE}")

        #### ROC in bins
        # bins = [evaluate.conds.min(), evaluate.conds.max()]
        bins = list(np.linspace(evaluate.conds.min(), 250, 18))
        bins.append(evaluate.conds.max())
        # bins.extend([240, evaluate.conds.max()])
        fig = evaluate.calculate_binned_roc(np.array(bins), colors=colors,
                                             truth_key=vDNN_label)
        if save_path is not None:
            misc.save_fig(fig, f"{save_path}/binned_aucs{FIG_TYPE}")

        fig = evaluate.calculate_binned_roc(np.array([bins[0], bins[-1]]),
                                            colors=colors,truth_key=vDNN_label,
                                            ylim=[0.95, 1.00])
            
        # evaluate.calculate_binned_eff(bins)
        
        # plt.figure()
        # for key, val in evaluate.sig_eff.items():
        #     plt.plot(np.array(bins[:-1])+np.diff(bins)/2, val, label=key)
        # plt.legend()
        # plt.figure()
        # for key, val in evaluate.bkg_eff.items():
        #     plt.plot(np.array(bins[:-1])+np.diff(bins)/2, val, label=key)
        # plt.legend()
        # plt.yscale("log")
        

