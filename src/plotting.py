"general plotting"
import matplotlib.pyplot as plt
import numpy as np
import copy
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib.colors as mcolors

import src.utils as utils
# from plot_utils import get_atlas_internal_str

np.seterr(invalid='ignore')

FIG_SIZE = (8, 6)
FONTSIZE = 16
LABELSIZE = 16
LEGENDSIZE = 14
KWARGS_LEGEND={"prop":{'size': LEGENDSIZE}, "frameon": False,
               "title_fontsize":LEGENDSIZE}
COLORS = list(mcolors.TABLEAU_COLORS)

def generate_figures(nrows, ncols=2, **kwargs):
    heights = [3, 1]*nrows
    gs_kw={"height_ratios": heights}
    fig, ax =  plt.subplots(ncols=ncols, nrows=nrows*2,gridspec_kw=gs_kw,
                            figsize=(8*ncols,6*nrows),
                            **kwargs)
    ax = np.reshape(ax, (2*nrows, ncols))
    return fig, ax

def plot_stairs(counts, bins,sigma=None, total_sigma=None,total_sigma_label=None,
                ax=None, normalise=True, style={}):
    if normalise:
        counts = counts/np.sum(counts)
    if ax is None: fig, ax = plt.subplots(1,1, figsize = (15,7))
    ax.stairs(counts, bins, **style)
    if sigma is not None:
        ax.stairs(counts+sigma, bins, alpha=0.30,
                    color=style.get("color", None),
                    # label=style.get("label", None),
                    baseline=counts-sigma, fill=True)
    if total_sigma is not None:
        ax.stairs(counts+total_sigma, bins, alpha=0.15,
                    color=style.get("color", None),
                    label=total_sigma_label if total_sigma_label is not None else None,
                    baseline=counts-total_sigma, fill=True)
    return ax

def hist_uncertainty(arr, weights, bins, stacked_bool=False):
    if isinstance(weights[0], np.ndarray):
        weights = [i**2 for i in weights]
        counts, _, _ = plt.hist(arr, weights=weights, bins=bins, stacked=stacked_bool)
    else:
        counts, _, _ = plt.hist(arr, weights=weights**2, bins=bins, stacked=stacked_bool)
    plt.close()
    if stacked_bool:
        return np.sqrt(counts[-1])
    else:
        return np.sqrt(counts)
    
def binned_errorbar(*dist, bins, **kwargs):
    if kwargs.get("ax", True):
        fig, ax = plt.subplots(1,1)
        
    xerr = np.abs((bins[:-1]-bins[1:])/2)
    xs_value = (bins[:-1]+bins[1:]) / 2


    for i in dist:
        eb1 = ax.errorbar(
            xs_value,i,
            xerr=xerr,
            ls='none',
            **kwargs.get("style", {})
            )
    # eb1[-1][0].set_linestyle("solid")
    return ax

def plot_hist(*args, **kwargs) -> plt.Figure:  # pylint: disable=too-many-locals
    """plot histogram

    Parameters
    ----------
    target : np.ndarray
        target distribution
    source : np.ndarray
        source distribution
    trans : np.ndarray
        transported source distribution
    binrange : tuple, optional
        range of the x axis, by default (-2, 2)

    Returns
    -------
    plt.Figure
        output figure
    """

    style = kwargs.setdefault("style", {"bins": 20, "histtype":"step"})
    mask_sig=None
    if kwargs.get("remove_placeholder", False):
        if (-999 in args[0]) & (len(args)>1): #remove -999 as a placeholder i args
            if ("mask_sig" in kwargs):
                mask_sig = [j[np.array(np.ravel(i))!=-999] for i, j in zip(args, kwargs["mask_sig"])]
            args = tuple([np.array(i)[np.array(i)!=-999] for i in args])
        

    if (not "range" in style) and (isinstance(style.get("bins", 20), int)):
        percentile_lst = kwargs.get("percentile_lst", [0.05,99.95])
        # just select percentile from first distribution
        if len(args[0])==2: # stakced array
            style["range"] = np.percentile(args[0][-1], percentile_lst)
        else:
            style["range"] = np.percentile(args[0], percentile_lst)
        

    names = kwargs.get("names", [f"dist_{i}" for i in range(len(args))])
    weights = kwargs.get("weights", [np.ones(len(i)) for i in args])
    
    # kw_weights = kwargs.get("weights", [None]*len(args))
    # weights=[]
    # for nr, w in enumerate(kw_weights):
    #     if w is None:
    #         weights.append([np.ones(len(args[nr]))])
    #     else:
    #         weights.append(w)

    if mask_sig is None:
        mask_sig = kwargs.get("mask_sig", [np.ravel(np.ones_like(i)==1) for i in args])
    
    uncertainties = kwargs.get("uncertainties", [None]*len(args))
    # if not "mask_sig" in kwargs:
    #     mask_sig = [np.ones_like(i)==1 for i in args]
   
    counts_dict = {}
    for nr, (name, dist, uncertainty) in enumerate(zip(names,args, uncertainties)):
        fig = plt.figure()
        weight = weights[nr]
        if not all(mask_sig[nr]):
            mask = mask_sig[nr]
            counts, bins, _ = plt.hist([dist[~mask], dist[mask]], stacked=True,
                                        weights=[weight[~mask], weight[mask]],
                                        **style) # stacked doesnt work for np.histogram
            unc = [None, hist_uncertainty([dist[~mask], dist[mask]], weights=[weight[~mask], weight[mask]],
                                          bins=bins, stacked_bool=True)
                   if uncertainty is None else uncertainty]
            if kwargs.get("norm", False):
                counts = list(counts/np.sum(counts,1)[:,None])
                
            else:
                counts = list(counts)
            weight = [weight, weight]
        else:
            counts, bins, _ = plt.hist(dist, weights=weight, **style)
            if weight is None:
                unc = [np.sqrt(counts)if uncertainty is None else uncertainty]
                weight = np.ones((len(dist)))
            else:
                unc = [hist_uncertainty(dist, weights=weight, bins=bins,
                                        stacked_bool=False)
                    if uncertainty is None else uncertainty]
            # if not style.get("stacked", False):
            counts = [counts]

            weight = [weight]


        counts_dict[name] = {"counts": counts, "unc": unc, "weight": weight}

        plt.close(fig)

    counts_dict["bins"] = bins
    if kwargs.get("plot_bool", True):
        return plot_hist_1d(counts_dict, **kwargs)
    else:
        return counts_dict, None
    
def plot_hist_integral(distributions, truth_key, var_names, conds_bins,
                       plot_kwargs={}, conds_names=None, save_path=None,
                       **kwargs):
    "still requires some work"

    if isinstance(conds_bins, (list, np.ndarray)):
        conds_bins = np.round(np.percentile(distributions[truth_key]["conds"],
                                                    conds_bins, 0),3)
        ax = plot_hist_integration_over_bins(distributions, bins=conds_bins,
                                             plot_kwargs=plot_kwargs,
                                             var_names=var_names, **kwargs)
    else:
        if conds_names is None:
            raise ValueError("When giving conds_bins as dict, conds_names is required!")

        for key, item in conds_bins.items():
            n_bins = len(item)

            fig, ax_all = generate_figures(n_bins-1, len(var_names))
            for n_row in range(n_bins):
                bins = np.c_[[items[n_row:n_row+2].T for i, items in conds_bins.items()]].T
                # ax = ax_all[2*n_row: 2*n_row+2, 0]
                plot_hist_integration_over_bins(distributions, bins=bins,
                                                    plot_kwargs=plot_kwargs,
                                                    ax=ax_all[2*n_row: 2*n_row+2],
                                                    var_names=var_names,
                                                    conds_col_nr=np.argmax(key==np.array(conds_names)), 
                                                    **kwargs)

            if save_path is not None:
                utils.save_fig(
                    fig,
                    f"{save_path}/{var_names}_{key}_{item}.{kwargs.get('format', 'png')}"
                    )

def plot_hist_integration_over_bins(distributions, bins, plot_kwargs, legend_title,
                                    var_names, ax=None, conds_col_nr=None,
                                    save_path=None,**kwargs):
    legend_title = kwargs.get("legend_title", "")
    for (low, high) in zip(bins[:-1], bins[1:]):
        dists = []
        labels=[]
        for key, values in distributions.items():
            if conds_col_nr is None:
                mask_conds =  np.all([(low[i]<=values["conds"][:,i]) &
                                    (high[i]>=values["conds"][:,i])
                                    for i in range(len(low))], 0)
            else:
                mask_conds =  ((low[0]<=values["conds"][:,conds_col_nr]) &
                                    (high[0]>=values["conds"][:,conds_col_nr]))
            
            dists.append(values["dist"][mask_conds])
            labels.append(values["labels"][mask_conds]==1)

        for col_nr in range(len(var_names)):
            plot_args = copy.deepcopy(plot_kwargs)
            title_str = f"{legend_title} pT: [{', '.join(map(str, low))}: {', '.join(map(str, high))})"

            if ax is None:
                fig, ax = generate_figures(1, 1, sharex="col")
            # histogram
            counts, _ = plot_hist(*[i[:,col_nr] for i in dists],ax=ax[0, col_nr], 
                                        # style=kwargs.get("style", {"bins":utils.default_bins()}),
                                        mask_sig=labels,
                                        # dist_styles=copy.deepcopy(dist_styles),
                                        legend_kwargs={"title":title_str},
                                        **plot_args)
            if kwargs.get("log_y", False):
                ax[0].set_yscale("log")

            # ratio plot
            plot_ratio(counts, truth_key="dist_0", ax=ax[1, col_nr],
                            # ylim=kwargs.get("ylim", [0.95, 1.05]),
                            # styles=copy.deepcopy(dist_styles),
                            zero_line_unc=True,
                            **plot_args)

            ax[1, col_nr].set_xlabel(var_names[col_nr])
            if "ylim" in plot_args:
                ax[1, col_nr].set_ylim(plot_args["ylim"])
            if (save_path is not None) & (not kwargs.get("single_figure", False)):
                utils.save_fig(
                    fig,
                    f"{save_path}/{var_names}_{low}_{high}_{col_nr}.{kwargs.get('format', 'png')}"
                    )
    # return ax

    

def plot_hist_1d(counts_dict:dict, **kwargs):
    """Plot histogram

    Parameters
    ----------
    counts : dict
        Should contain:   bins:[], truth:{counts:[], unc:[], weights: []},
                          different_counts:{counts:[],unc:[], weights: []}

    Returns
    -------
    dict, ax
        return counts and ax
    """
    if "bins" in counts_dict:
        bins = counts_dict["bins"]
    else:
        raise ValueError("counts_dict missing bins")
    
    ax = kwargs.get("ax", None)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))


    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='minor', length=4)
    
    # define bins
    xs_value = (bins[:-1]+bins[1:]) / 2
    # fixing the bound of the x axis because of steps
    if kwargs.get("xerr_on_errorbar", False):
        xerr = np.abs((bins[:-1]-bins[1:])/2)
    else:
        # xerr[1:-1] = np.zeros_like(xerr[1:-1])
        xerr = np.zeros_like(bins[:-1])
    
    for nr, (name, counts) in enumerate(counts_dict.items()):
        if name == "bins":
            continue
        
        # if uncertainties or weight are missing
        counts.setdefault("unc", [np.zeros_like(counts["counts"])[0]])
        counts.setdefault("weight", [np.ones_like(counts["counts"])[0]])
        # this type of loop is need if hist(stacked=True)
        for nr_count, (count, unc, weight) in enumerate(zip(counts["counts"],
                                                            counts["unc"],
                                                            counts["weight"])):
            style = copy.deepcopy(kwargs.get("dist_styles", [{}]*(nr+1))[nr])
            if (len(counts["counts"])==2) & (nr_count ==0):
                style["label"] = "Background"
                style["alpha"] = 0.5
            else:
                style["alpha"] = 1

            style.pop("drawstyle", None)
            if not isinstance(weight[0], float):
                sum_weight = sum([np.sum(i) for i in weight])
            else:
                sum_weight = np.sum(weight)


            yerr=(None if unc is None else
                    unc/sum_weight if kwargs.get("normalise", True) else
                    unc)
            hist_counts = count/sum_weight if kwargs.get("normalise", True) else count

            if "marker" in style:
                # style["linestyle"]="none"
                eb1 = ax.errorbar(
                    xs_value,
                    hist_counts,
                    xerr=xerr,
                    yerr=(None if unc is None else
                          unc/sum_weight if kwargs.get("normalise", True) else
                          unc),
                    **style
                    )
                # eb1[-1][0].set_linestyle(style.get("linestyle", "solid"))
            else:
                yerr = 0 if yerr is None else yerr
                # ax.stairs(count/np.sum(weight) if kwargs.get("normalise", True) else count, bins, **style)

                # eb1 = ax.errorbar(
                #     xs_value,
                #     count/np.sum(weight) if kwargs.get("normalise", True) else count,
                #     marker=".",
                #     linewidth=0,
                #     yerr=None if unc is None else unc/np.sum(weight),
                #     )
                # print(eb1[-1])
                # eb1[-1][0].set_linestyle(style.get("linestyle", "solid"))
                if not isinstance(weight[0],  (int, float,  np.float32,  np.float64)): # only for stacked=True
                    ax.stairs(hist_counts[-1], bins, baseline=hist_counts[-1]-yerr[-1],fill=True, alpha=0.1, color = style.get("color", None))
                    ax.stairs(hist_counts[-1], bins, baseline=hist_counts[-1]+yerr[-1],fill=True, alpha=0.1, color = style.get("color", None))
                    for i, label in zip(hist_counts, kwargs["stacked_labels"]): # need to have style be list
                        ax.stairs(i, bins, **style, label=label)
                else:
                    ax.stairs(hist_counts, bins, baseline=hist_counts-yerr,fill=True, alpha=0.1, color = style.get("color", None))
                    ax.stairs(hist_counts, bins, baseline=hist_counts+yerr,fill=True, alpha=0.1, color = style.get("color", None))
                    ax.stairs(hist_counts, bins, **style)
    legend_kwargs = KWARGS_LEGEND.copy()
    legend_kwargs.update(kwargs.get("legend_kwargs", {}))
    ax.legend(**legend_kwargs)

    ax.tick_params(axis="both", which="major", labelsize=LABELSIZE)
    ylabel = "Normalised counts" if kwargs.get("normalise", True) else "#"
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    return counts_dict, ax

def plot_ratio(counts:dict, truth_key:str, **kwargs):
    """Ratio plot

    Parameters
    ----------
    counts : dict
        Counts from plot_hist
    truth_key : str
        Which key in counts are the truth
    kwargs:
        Add ylim if you want arrow otherwise they are at [0,2]
    Returns
    -------
    _type_
        _description_
    """
    ax = kwargs.get("ax", None)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(11, 8))
    bins = counts["bins"]

    styles = copy.deepcopy(kwargs.get("styles", [{} for _ in counts.keys()]))
    ylim = kwargs.get("ylim", [0,2])
    xs_value = (bins[:-1]+bins[1:]) / 2
    xerr = (bins[:-1]-bins[1:])/2
    if "counts" in counts[truth_key]:
        counts_truth = counts[truth_key]["counts"][-1]
    else:
        counts_truth = counts[truth_key]
    counts_truth=np.array(counts_truth)
    if kwargs.get("zero_line", True):
        ax.plot(bins, np.ones_like(bins),
                    label="Zero line",
                    color="black",
                    linewidth=2,
                    linestyle="dashed",
                    zorder=-1,
                    )
        if kwargs.get("zero_line_unc", False):
            if np.sum(counts[truth_key]["unc"][-1])!=0:
                unc = np.divide(counts[truth_key]["unc"][-1],counts[truth_key]["counts"][-1])
            else:
                unc = np.sqrt(counts_truth)/counts_truth
            ax.fill_between(xs_value, np.ones_like(xs_value)-np.abs(unc), 
                            np.ones_like(xs_value)+np.abs(unc), color='black',
                            alpha=0.1, zorder=-1)
    nr = 0
    for (name, count), style in zip(counts.items(), styles):
        if (name == "bins") or (name == truth_key):
            continue

        if "counts" in count:
            if not isinstance(count.get("counts",None), list):
                # not isinstance(count.get("unc",None), list)
                raise TypeError("Counts in dict has to be a list")
            count_pr_bins = np.array(count["counts"][-1])
        else:
            count_pr_bins = np.array(count)
            
        if "unc" in count:
            if not isinstance(count.get("unc",None), list):
                # not isinstance(count.get("unc",None), list)
                raise TypeError("Counts in dict has to be a list")
            menStd = np.array(count["unc"][-1])
            if len(menStd.shape)>1:
                menStd = menStd[-1]
        else:
            menStd = np.zeros_like(count_pr_bins)

        if kwargs.get("normalise", True):
            y_counts = (count_pr_bins/count_pr_bins.sum())/(counts_truth/counts_truth.sum())
            yerr_relative = (menStd/count_pr_bins.sum())/(counts_truth/counts_truth.sum())
        else:
            y_counts = count_pr_bins/counts_truth
            yerr_relative = menStd/counts_truth

        # up or down error if outside of ylim
        yerr_relative = np.nan_to_num(yerr_relative, nan=10, posinf=10)
        mask_down = ylim[0]>=y_counts
        mask_up = ylim[1]<=y_counts

        style["linestyle"]="none"
        # yerr_relative[mask_down | mask_up] = 0
        # y_counts[mask_down | mask_up] = -1
        
        #additional uncertainties
        if "total_unc" in count:
            total_unc = np.array(count["total_unc"][-1])
            if kwargs.get("normalise", True):
                total_unc = ((total_unc/count_pr_bins.sum())
                                 /(counts_truth/counts_truth.sum()))
            else:
                total_unc = total_unc/counts_truth
            
            ax.stairs(y_counts+total_unc, bins, alpha=0.3,
                    color=style.get("color", None),
                    baseline=y_counts-total_unc, fill=True)
            

        # plotting
        ax.errorbar(xs_value, y_counts, yerr=yerr_relative,
                        xerr = np.abs(xerr), **style
                        )
        #marker up
        if ("label" not in style) & kwargs.get("legend_bool", True):
            style["label"] = name
        style.update({"marker": '^', "s": 35, "alpha": 1})
        style.pop("linestyle")
        ax.scatter(xs_value[mask_up],
                    np.ones(mask_up.sum())*(ylim[1]-ylim[1]/100),
                    **style)
        
        #marker down
        style["marker"] = 'v'
        ax.scatter(xs_value[mask_down],
                    np.ones(mask_down.sum())*(ylim[0]+ylim[0]/100),
                    **style)

        nr+=1

    ax.set_ylabel("Ratio", fontsize=FONTSIZE)
    ax.set_ylim(ylim)
    ax.tick_params(axis="both", which="major", labelsize=LABELSIZE)
    return ax

def plot_roc_curve(truth, pred, weights=None, label="",
                   fig=None, uncertainty=False,
                   **kwargs):
    fpr, tpr, _ = roc_curve(truth, pred, sample_weight=weights, drop_intermediate=False
                            )
    plot_bool = kwargs.get("plot_bool", True)
    
    auc = roc_auc_score(truth, pred, sample_weight=weights)
    if plot_bool:
        if (fig is None) and plot_bool:
            fig = plt.figure(figsize=(8,8))
        auc_str = str(round(auc,4))

        label += f"AUC: {auc_str}"
        plt.plot(fpr, tpr, linewidth=2, label=label, color=kwargs.get("color", "blue"))
        plt.plot([0, 1], [0, 1], 'k--')
        
        if uncertainty:
            N = len(tpr) if weights is None else np.sum(weights)
            uncertainty = np.sqrt(tpr*(1-tpr)/N)/tpr
            plt.fill_between(fpr, tpr+uncertainty, tpr-uncertainty, alpha=0.5)

        plt.axis([0, 1, 0, 1])
        plt.xticks(np.arange(0,1, 0.1), rotation=90)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc='best', frameon=False)
        plt.tight_layout()
    return tpr, fpr, auc, fig


