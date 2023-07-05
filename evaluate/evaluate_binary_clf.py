"evaluate"
import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

#package
import numpy as np
import torch as T
import matplotlib.pyplot as plt

#private
import src.pipeline as pl
import src.utils as utils
import src.plotting as plot
import src.eval_utils as eval_utils


if __name__ == "__main__":
    FIG_TYPE=".pdf"
    # %matplotlib widget
    device="cpu"
    run_ot_bool = True
    run_flow_bool = True
    style_bkg_rej = {"ls":"dashed", "lw": 2}
    save_path = "figures/1d"

    internal_plot_bool=True
    size=None
    vDNN_label = r"$\mathcal{D}_\mathrm{VB}$"

    FLOW_PATHS=[]

    OT_PATHS = [
        "/home/malteal/local_work/ot-decorrelation/outputs/OT/1d/OT_2023_07_05_14_42_07_841701_base_uniform_1_1/"
        ]
    eval_data_path = "/home/malteal/unige/combined_scores.h5"

    # model ot model
    evaluate = eval_utils.EvalauteFramework("eval/", plot_bool=internal_plot_bool, save_path=save_path, fig_type=FIG_TYPE)
    colors = ["blue", "red", "green", "darkorange", "black"]
    n_plots=0
    fig_sig, ax_sig = plt.subplots(1,1,figsize=(8,6))
    fig_bkg, ax_bkg = plt.subplots(1,1,figsize=(8,6))
    ax_bkg.set_yscale("log")
    
    label = vDNN_label #"vDNN" 

    output, data = pl.load_multi_cls(eval_data_path, upper_mass_cut=450)
    data["w_score"] = 2*(1/(1+np.exp(-utils.get_disc_func(2)(output["encodings"]))))-1

    data = data[data.label!=1]

    results = data[["mass", "label", "w_score"]]

    results = results.rename(columns={"w_score": label, "label": "labels"})

    results["labels"][results["labels"]!=2] = 0
    results["labels"][results["labels"]==2] = 1

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
        for i in OT_PATHS:
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
        for i in FLOW_PATHS:
            if len(FLOW_PATHS)==1:
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
        utils.save_fig(fig_bkg, f"{save_path}/1d_bkg_rej{FIG_TYPE}")
        utils.save_fig(fig_sig, f"{save_path}/1d_sig_eff{FIG_TYPE}")

    #### ROC in bins
    bins = list(np.linspace(evaluate.conds.min(), 250, 18))
    bins.append(evaluate.conds.max())

    fig = evaluate.calculate_binned_roc(np.array(bins), colors=colors,
                                            truth_key=vDNN_label)
    if save_path is not None:
        utils.save_fig(fig, f"{save_path}/binned_aucs{FIG_TYPE}")


