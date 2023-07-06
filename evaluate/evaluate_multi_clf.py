"evaluate multi class tagger"
import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import torch as T
from tqdm import tqdm

import src.pipeline as pl
import src.eval_utils as eval_utils
import src.plotting as plot
import src.utils as utils
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


class Metrics(eval_utils.EvalauteFramework):
    "Project the 3d classifier scores down to 1d and evaluate same as 1d"
    def __init__(self, distribution:dict, conds, label, bkg_label=0, sig_label=1,
                 save_path=None, disc_func=None, n_bins=11, plot_bool=False,device="cpu",
                 verbose=False) -> None:
        self.disc_func = disc_func
        self.distribution=distribution
        self.conds=conds
        self.device=device
        self.plot_bool=plot_bool
        self.label=np.array(label)
        self.n_bins=n_bins
        self.mass_percentiles = np.round(np.linspace(np.percentile(self.conds, 0.1),
                                                     np.percentile(self.conds, 99),
                                                     n_bins),3)
        super().__init__(conds, bkg_label, sig_label, verbose=verbose,
                         save_path=save_path, plot_bool=plot_bool)
        if disc_func is not None:
            self.project_distribution(self.disc_func)
        self.redefine_output(self.distribution, clf_col="mDNN")

    def project_distribution(self, disc_func=None, sig_label=None):
        self.disc_func = disc_func        
        self.projected_distributions = {key: self.disc_func(i)
                                        for key, i in self.distribution.items()
                                        if key!="labels"}
        self.projected_distributions["labels"] = self.label
        if sig_label is not None:
            self.sig_label=sig_label

    def run_ot(self, ot_path, col_name=None, reverse_bool=False):
        if col_name is None:
            col_name = f"OT_{ot_path.split('/')[-1].split('_')[-4]}"
        super().run_ot(ot_path, col_name, self.device, reverse_bool)
        self.projected_distributions[col_name] = self.disc_func(self.output[col_name])
        
    def run_flow(self, flow_path, col_name=None, reverse_bool=False):
        if col_name is None:
            col_name = f"flow_{flow_path.split('/')[-1].split('_')[-4]}"
        super().run_flow(flow_path, col_name, self.device, reverse_bool)
        self.projected_distributions[col_name] = self.disc_func(self.output[col_name])


if __name__ =="__main__":
    FIG_TYPE = ".pdf"
    style_bkg_rej = {"ls":"dashed", "lw": 2}
    
    save_fig_path = "figures/3d/"
    plot_bool=True
    device="cpu"
    MULTI_CLF = "data/combined_scores.h5"
    
    OT_PATH = [
        "outputs/OT/3d/OT_2023_07_05_15_39_07_997447_base_uniform_3_1",
        ]

    FLOW_PATH = [
        ]

    bkg_rej = [0.5, 0.9, 0.95, 0.99]

    truth, data = pl.load_multi_cls(MULTI_CLF, upper_mass_cut=450)
    prob_distributions = {"mDNN": truth["encodings"],
                          "labels": np.array(truth["labels"])}
    data = data.to_numpy()
    mass = data[:,0]
    metric_cls = Metrics(prob_distributions, conds=mass,
                         label=np.array(truth["labels"]),
                         bkg_label=0,
                         n_bins=5, plot_bool=plot_bool,
                         verbose=False, device=device,
                         save_path=save_fig_path
                         )

    disc_name = [r"$\mathcal{D}_\mathrm{QCD}$",
                 "$\mathcal{D}_\mathrm{T}$",
                 "$\mathcal{D}_\mathrm{VB}$"]
    
    # running the different projections
    for sig_label, legend_title in enumerate(disc_name):
        metric_cls.project_distribution(utils.get_disc_func(sig_label), sig_label)

        #calculating OT projections
        for path in OT_PATH:
            name =f"OT-mDNN {path.split('/')[-1].split('_')[-3]}"
            name = name.replace("uniform", "Dir(1,1,1)")
            metric_cls.run_ot(path, col_name=name)

        #calculating flow projections
        for nr, path in enumerate(FLOW_PATH):
            name=None
            if len(FLOW_PATH)==1:
                    name = "cf-mDNN"
            metric_cls.run_flow(path, col_name=name)

        fig, ax = plt.subplots(1,2,figsize=(8*2,6))
        nr=0
        for i in metric_cls.projected_distributions:
            if i == "labels":
                continue
            
            # calculate 1/JSD
            jsd_lst_disco, sig_eff, background_rej = metric_cls.bkg_rej_calculation(
                i, proba_flat = metric_cls.projected_distributions[i],
                legend_kwargs={"title": f"{i}: {legend_title}"},
                background_rej=bkg_rej
                )
            jsd_lst_disco = 1/jsd_lst_disco
            color = plot.COLORS[nr]
            
            if len(sig_eff.shape)>1:
                ax[0].plot(background_rej, sig_eff[:,0],
                            label = i, color=color,
                            **style_bkg_rej)

                ax[0].plot(background_rej, sig_eff[:,1],
                           color=color,lw=style_bkg_rej["lw"], ls="solid")
            
            ax[1].plot(background_rej, jsd_lst_disco,
                        label = i,color=color,
                        **style_bkg_rej)
            

            #### ROC over conditional distribution
            bins = list(np.linspace(metric_cls.conds.min(), 250, 18))
            bins.append(metric_cls.conds.max())
            output_dict = copy.deepcopy(metric_cls.projected_distributions)
            mask_bkg=output_dict["labels"]!=sig_label
            mask_sig=output_dict["labels"]==sig_label
            output_dict["labels"][mask_bkg] = 0
            output_dict["labels"][mask_sig] = 1
            fig = metric_cls.calculate_binned_roc(np.array(bins),
                                                  output_dict=output_dict,
                                                  colors=plot.COLORS,
                                                    truth_key="mDNN")
            nr+=1
        #calculate the ideal
        jsd_lst_ideal, _, background_rej = metric_cls.ideal_calculation(
            background_rej=bkg_rej, sample_times=20)

        ax[1].errorbar(background_rej, jsd_lst_ideal["mean"], yerr=jsd_lst_ideal["std"],
                    label = "Ideal",color="grey",lw=style_bkg_rej["lw"],ls="dotted")
        ax[1].set_yscale("log")
        for nr, (i,j) in enumerate(zip(["Background rejection", "Background rejection"],
                    ["Signal Efficiency", "1/JSD"])):
            ax[nr].set_xlabel(i, fontsize=plot.FONTSIZE)
            ax[nr].set_ylabel(j, fontsize=plot.FONTSIZE)
            ax[nr].tick_params(axis="both", which="major", labelsize=plot.LABELSIZE)
        plt.tight_layout()
        for nr_ in range(2):
            legend_kwargs = {"prop":{'size': 12}, "frameon": False, "title_fontsize":12}
            if nr_==1:
                legend_kwargs["bbox_to_anchor"] = (0.5, 0.47)
                legend_kwargs["loc"] = "center"
            if ("QCD" in legend_title) and (nr_==0):
                legend_kwargs["loc"] = "center right"
                legend_kwargs["bbox_to_anchor"] = (1, 0.60)
            legend_kwargs["title"] = legend_title
            ax[nr_].legend(**legend_kwargs)
        
        # top/WZ legend
        ax2 = ax[0].twinx()
        ax2.plot([], [],color="black",lw=style_bkg_rej["lw"], ls="dashed", label="Top")
        ax2.plot([], [],color="black",lw=style_bkg_rej["lw"], ls="solid", label="VB")
        ax2.get_yaxis().set_visible(False)
        ax2.legend(**{"prop":{'size': 14}, "frameon": False,"title_fontsize":14})

        if save_fig_path is not None:
            utils.save_fig(fig, f"{save_fig_path}/bkg_reject_label_{sig_label}{FIG_TYPE}")

