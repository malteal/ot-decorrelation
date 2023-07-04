"evaluate multi class tagger"
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from sklearn import metrics
import torch as T
from tqdm import tqdm
import copy

import pipeline as pl
from evaluate_clf import EvalauteFramework

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, inset_axes, mark_inset

import utils 
from tools.visualization import general_plotting as plot
from otcalib.torch.utils import utils as ot_utils
from tools import misc, scalers
from tools.flows import Flow

figsize = (8,6)
columns = ["mass", "label", "q_score", "t_score", "w_score"]

def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

def sample_small_spherical_change(size):
    phi = np.linspace(0, np.pi, 100)
    theta = np.linspace(0, 2 * np.pi, 200)
    x = np.outer(np.sin(theta), np.cos(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.cos(theta), np.ones_like(phi))

    xi, yi, zi = sample_spherical(size)
    return np.c_[xi,yi,zi]

def plot_data(data):
    mass = data[:,0]
    for label in np.unique(data[:,1]):
        plt.figure(figsize=figsize)
        mask_label = data[:,1]==label
        plt.hist(data[:,0][mask_label], bins=50, range= [20, 400])
        plt.figure(figsize=figsize)
        style = {"histtype": "step", "bins": 50, "density": True}
        for i in range(3):
            style["label"] = columns[2+i]
            _, bins, _= plt.hist(data[:,2+i][mask_label], **style)
            style["bins"] = bins
        plt.legend()

def transport_map(ot_model, flow_model, size=301, save_fig_path=None,
                  image_format=".pdf", mass=127):

    x,y = np.meshgrid(np.linspace(1e-4,0.999, size), np.linspace(1e-4,0.999, size))
    x,y = np.ravel(x),np.ravel(y)
    meshgrid = np.stack([x,y],1)
    meshgrid = np.c_[meshgrid, np.abs(1-np.sum(meshgrid,1))]
    prob_norm = np.sum(meshgrid,1)==1
    x,y = x[prob_norm],y[prob_norm]
    meshgrid = meshgrid[prob_norm]
    # small s
    r_sq = sample_small_spherical_change(len(meshgrid))
    d_meshgrid = np.clip(meshgrid+r_sq/100, 0.00001, 0.99999)
    d_meshgrid = d_meshgrid/d_meshgrid.sum(1)[:,None]
    d_dummy_prob = ot_utils.logit(d_meshgrid)
    d_dummy_prob = T.tensor(d_dummy_prob,requires_grad=True).float().to(device)
    # d_meshgrid = meshgrid*(1+np.random.uniform(-0.01,0.01, (len(meshgrid),3)))
    #new
    r_sq_new = sample_small_spherical_change(len(meshgrid))
    d_meshgrid_new = np.clip(meshgrid+r_sq_new/1000, 0.0001, 0.9999)
    d_meshgrid_new = d_meshgrid_new/d_meshgrid_new.sum(1)[:,None]
    d_dummy_prob_new = ot_utils.logit(d_meshgrid_new)
    d_dummy_prob_new = T.tensor(d_dummy_prob_new,requires_grad=True).float().to(device)

    # dummy_prob = ot_utils.logit(np.random.dirichlet((1,1,1), 3000))
    dummy_prob = ot_utils.logit(meshgrid)
    dummy_prob = T.tensor(dummy_prob,requires_grad=True).float().to(device)
    proba_dummy_prob = ot_utils.probsfromlogits(dummy_prob.cpu().detach().numpy())

    
    dummy_conds = T.tensor([np.log(mass)]*len(dummy_prob)).float().view(-1,1).to(device)
    output={i:{} for i in ["regular", "delta"]}
    for i,dist in zip(["regular", "delta"], [dummy_prob, d_dummy_prob]):
        ot_output = ot_model.chunk_transport(dummy_conds, dist).cpu().detach().numpy()
        output_flow = [flow_model._transform.inverse(i, j)[0].cpu().detach().numpy() for i,j in zip(dist.chunk(20), dummy_conds.chunk(20))]
        output_flow = np.concatenate(output_flow, 0)
        ot_output = ot_utils.probsfromlogits(ot_output)
        output_flow = ot_utils.probsfromlogits(output_flow)
        output[i]["OT"] = ot_output-proba_dummy_prob
        output[i]["flow"] = output_flow-proba_dummy_prob
        
    cs_ot = np.sqrt(np.sum(((output["regular"]["OT"]-output["delta"]["OT"]))**2,1))+1e-8
    cs_flow = np.sqrt(np.sum(((output["regular"]["flow"]-output["delta"]["flow"]))**2,1))+1e-8
    # cs_ot = np.sqrt(np.sum(((output["regular"]["OT"]-output["delta"]["OT"])-
    #                         (output["regular"]["OT"]-output["delta_new"]["OT"]))**2,1))+1e-8
    # cs_flow = np.sqrt(np.sum(((output["regular"]["flow"]-output["delta"]["flow"])
    #                           - (output["regular"]["flow"]-output["delta_new"]["flow"]))**2,1))+1e-8
    # cs_flow = np.diag(cs_flow)
    
    
    
    if True:
        fig, ax = plt.subplots(1,1)
        print(np.max(cs_ot), np.max(cs_flow))
        counts, ax, = plot.plot_hist(np.ravel(cs_ot), np.ravel(cs_flow),
                                    style={
                                        "bins": 51, "range": [0, 1.05]
                                            },#percentile_lst=[0.1,100],
                                        ax=ax,
                                        dist_styles=[{"label":"OT-mDNN normal"},
                                                    {"label":"cf-mDNN"}],
                                        normalise=True, legend_kwargs={"loc": "upper right"})
        ax.set_yscale("log")
        ax.set_xlabel("|T-T'|")
        if save_fig_path is not None:
            misc.save_fig(fig, f"{save_fig_path}/hist_of_diff_{mass}_{image_format}")
    # mask_ot = cs_ot>np.max(cs_ot)
    mask_flow = cs_flow>np.max(cs_ot)
    print("Procentage above OT max: ", np.sum(mask_flow)/len(mask_flow))
    if size<101:
        labels=["OT-mDNN normal", "cf-mDNN"]
        for nr,(index,name) in enumerate(zip([[0,1], [1,2], [2,0]],
                                            [[r"$p_{QCD}$", r"$p_{Top}$"], [r"$p_{Top}$", r"$p_{VB}$"], [r"$p_{VB}$", r"$p_{QCD}$"]])):
            fig, ax = plt.subplots(1,2, figsize=(8*2*0.7,6*0.7), sharey=True)
            for nr_col,out in enumerate([output["regular"]["OT"],
                                        output["regular"]["flow"]
                                        ]):
                trans_cost = str(round(
                    np.abs(out).mean()+np.abs(out).mean(),
                    4))
                ax[nr_col].quiver(proba_dummy_prob[:,index[0]],
                        proba_dummy_prob[:,index[1]],
                        out[:,index[0]],
                        out[:,index[1]],
                        label=f"{labels[nr_col]}", 
                        # label=f"{labels[nr_col]}. Average transport cost: {trans_cost}", 
                        color=plot.COLORS[nr_col],
                        units='xy',
                        angles='xy',
                        alpha=0.3,
                        scale_units="xy",
                        scale=1,#1/np.sqrt(np.sum(out**2,1))
                        # scale=1/np.sqrt(out[:,index[0]]**2+out[:,index[1]]**2)
                        )
                if "cf" in labels[nr_col]:
                    ax[nr_col].quiver(proba_dummy_prob[:,index[0]][mask_flow],
                            proba_dummy_prob[:,index[1]][mask_flow],
                            out[:,index[0]][mask_flow],
                            out[:,index[1]][mask_flow],
                            # label=f"{labels[nr_col]}. Average transport cost: {trans_cost}", 
                            color="red",
                            units='xy',
                            angles='xy',
                            alpha=0.3,
                            scale_units="xy",
                            scale=1,#1/np.sqrt(np.sum(out**2,1))
                            # scale=1/np.sqrt(out[:,index[0]]**2+out[:,index[1]]**2)
                            )
                for box_loc, zoom_loc in zip([[0.4,0.6], [0.7, 0.35]],
                                            [
                                                [[0.0, 0.25], [0.7,0.9]],
                                                [[0.7, 0.9], [0.00,0.20]],
                                            ]):
                    axin1 = ax[nr_col].inset_axes(box_loc+[0.25, 0.25])
                    axin1.quiver(proba_dummy_prob[:,index[0]],
                            proba_dummy_prob[:,index[1]],
                            out[:,index[0]],
                            out[:,index[1]],
                            # label=f"{labels[nr_col]}. Average transport cost: {trans_cost}", 
                            color=plot.COLORS[nr_col],
                            scale_units="xy",
                            units='xy',
                            angles='xy',
                            alpha=0.3,
                            width=0.005,
                            scale=1,#1/np.sqrt(np.sum(out**2,1))
                            # scale=1/np.sqrt(out[:,index[0]]**2+out[:,index[1]]**2)
                            )
                    if "cf" in labels[nr_col]:
                        axin1.quiver(proba_dummy_prob[:,index[0]][mask_flow],
                                proba_dummy_prob[:,index[1]][mask_flow],
                                out[:,index[0]][mask_flow],
                                out[:,index[1]][mask_flow],
                                # label=f"{labels[nr_col]}. Average transport cost: {trans_cost}", 
                                color="red",
                                units='xy',
                                angles='xy',
                                alpha=0.8,
                                width=0.005,
                                scale_units="xy",
                                scale=1,#1/np.sqrt(np.sum(out**2,1))
                                # scale=1/np.sqrt(out[:,index[0]]**2+out[:,index[1]]**2)
                                )
                    axin1.set_xlim(*zoom_loc[0])
                    axin1.set_ylim(*zoom_loc[1])
                    axin1.set_xticks([])#visible=False)
                    axin1.set_yticks([])#visible=False)
                    ax[nr_col].indicate_inset_zoom(axin1, edgecolor='g')

                ax[nr_col].legend(frameon=False)
            ax[0].set_ylabel(name[-1])
            ax[0].set_xlabel(name[0])
            ax[1].set_xlabel(name[0])
            if save_fig_path is not None:
                misc.save_fig(fig, f"{save_fig_path}/displacement_plots_mass_{mass}_{name}{image_format}")

    fig_ot, ax_ot = plt.subplots(1,2, figsize=(8*2,6), sharey=True)
    style={
        # "vmax":np.log(cs_flow.max()), "vmin":np.log(cs_flow.min()),
        "vmax":cs_flow.max(), "vmin":cs_flow.min(),
            "levels":100}
    cntr2=ax_ot[0].tricontourf(proba_dummy_prob[:,1], 
                            proba_dummy_prob[:,2], 
                            # np.log(cs_ot),
                            cs_ot,
                            **style
                            )
    cntr1 = ax_ot[1].tricontourf(proba_dummy_prob[:,1], 
                            proba_dummy_prob[:,2],
                            # np.log(cs_flow),
                            cs_flow,
                            **style
                                )
    print(np.min(cs_ot),np.max(cs_ot))
    print(np.min(cs_flow), np.max(cs_flow))
    for i in range(2):
        ax_ot[i].legend(title="OT-mDNN normal" if i==0 else "cf-mDNN",
                        frameon=False, title_fontsize=plot.LEGENDSIZE+5)
        ax_ot[i].set_xlabel(r"$p_{Top}$", fontsize=plot.LABELSIZE)
        ax_ot[i].tick_params(axis="both", which="major", labelsize=plot.LABELSIZE)
    ax_ot[0].set_ylabel(r"$p_{VB}$", fontsize=plot.LABELSIZE)
    fig_ot.tight_layout()
    
    cbar=fig_ot.colorbar(cntr1, ax=ax_ot.ravel().tolist())
    
    cbar.ax.tick_params(labelsize=plot.LABELSIZE)
    cbar.set_label(label=r"|T-T'|", fontsize=plot.LABELSIZE)
    if save_fig_path is not None:
        misc.save_fig(fig_ot, f"{save_fig_path}/magnitude_differences_mass_{mass}{image_format}",
                      tight_layout=False)
    return proba_dummy_prob, output, cs_ot, cs_flow

def calculate_auc(label, pred, ax=None, legend:str=""):
    fpr, tpr, thresholds = metrics.roc_curve(label, pred)
    
    auc = metrics.auc(fpr, tpr)
    if ax is not None:
        ax.plot(fpr, tpr, label=f"{legend}, AUC: {np.round(auc,4)}")
    return auc

class Metrics(EvalauteFramework):
    def __init__(self, distribution:dict, conds, label, bkg_label, sig_label,
                 save_path=None, disc_func=None, n_bins=11, plot_bool=False,device="cpu",
                 verbose=False) -> None:
        self.sig_label=sig_label
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
        self.project_distribution(self.disc_func)
        super().__init__(conds, bkg_label, sig_label, verbose=verbose,
                         save_path=save_path, plot_bool=plot_bool)
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
        super().run_ot(ot_path, col_name, self.device, reverse_bool,
                       missing_kwargs={"cvx_dim":3,"noncvx_dim":1})
        self.projected_distributions[col_name] = self.disc_func(self.output[col_name])
        
    def run_flow(self, flow_path, col_name=None, reverse_bool=False):
        if col_name is None:
            col_name = f"flow_{flow_path.split('/')[-1].split('_')[-4]}"
        super().run_flow(flow_path, col_name, self.device, reverse_bool)
        self.projected_distributions[col_name] = self.disc_func(self.output[col_name])
    
    def plot_conds_distribution(self):
        style={"bins":51, "range":[20,300], "histtype": "step", "lw":1.5,
               "density": True}
        columns = ["q_score", "t_score", "w_score"]
         
        fig, ax = plt.subplots(1,len(self.distribution)+1, sharey=True,
                               figsize = (16,6))
        for i in np.unique(self.label):
            mask_mass = self.label==i
            ax[0].hist(self.conds[mask_mass], label=columns[int(i)], **style)
            for nr, (name, value) in enumerate(self.distribution.items()):
                mask_pred_label = np.argmax(value,1) == i
                ax[nr+1].hist(self.conds[mask_pred_label], label=columns[int(i)], **style)
        
        title=["Truth"]+list(self.distribution.keys())
        for i in range(len(ax)):
            ax[i].set_xlabel("Mass [GeV]")
            ax[i].legend(title=title[i], prop={'size': 15})
        ax[0].set_ylabel("Normalised entries")
        plt.tight_layout()

    def plot_rocs(self, one_v_all=False):
        for low,high in zip(self.mass_percentiles[:-1],
                            self.mass_percentiles[1:]):
            mask_mass_bin = (self.conds>=low) & (self.conds<high)
            if one_v_all:
                fig, ax = plt.subplots(1,len(np.unique(self.label)), figsize=(14,7))
                for i in np.int64(np.unique(self.label)):
                    for name, dist in self.distribution.items():
                        calculate_auc(self.label[mask_mass_bin]==i,
                                    dist[:,i][mask_mass_bin], ax=ax,
                                    legend=name)
            else:
                fig, ax = plt.subplots(1,1)
                for name, dist in self.projected_distributions.items():
                    if name=="labels":
                        continue
                    calculate_auc(self.output["labels"][mask_mass_bin],
                                  dist[mask_mass_bin], ax=ax,
                                  legend=name)
                    ax[i].legend(title=f"Label: {i}")
            fig.suptitle(f"Mass: {low}, {high}", fontsize=16)
    
    def probability_pr_bin(self, mass_bins):
        style={"bins":11, "range":[0,1], "histtype": "step", "lw":1.5,
               "density": True}
        fig, ax = plt.subplots(1,len(self.distribution), sharey=True,
                                figsize = (16,6))
        for low,high in zip(mass_bins[:-1],
                            mass_bins[1:]):
            mask_mass_bin = (self.conds>=low) & (self.conds<high)
            for nr, (name, value) in enumerate(self.distribution.items()):
                ax[nr].hist(value[:, 0][mask_mass_bin], label=f"mass: [{low}, {high}]",
                            **style)
        title=list(self.distribution.keys())
        for i in range(len(ax)):
            ax[i].set_xlabel("Mass [GeV]")
            ax[i].legend(title=title[i], prop={'size': 15})
        ax[0].set_ylabel("Normalised entries")
        plt.tight_layout()
        
                    
    def label_pr_bin(self):
        for name, dist in self.distribution.items():
            binned_events = []
            for low,high in zip(self.mass_percentiles[:-1],
                                self.mass_percentiles[1:]):
                mask_mass_bin = (self.conds>=low) & (self.conds<high)
                nevents = [np.mean(np.argmax(dist[mask_mass_bin], 1)==i)
                            for i in np.int64(np.unique(self.label))]
                binned_events.append(nevents)
            binned_events = np.array(binned_events)
            _, ax = plt.subplots(1,1, figsize = (15,7))
            for i in range(binned_events.shape[1]):
                ax = plot.plot_stairs(binned_events[:,i], self.mass_percentiles,
                                    normalise=False, ax=ax,
                                    style={"label": columns[2+i], "lw": 2})
            ax = plot.plot_stairs(binned_events.sum(1), self.mass_percentiles,
                                normalise=False, ax=ax,
                                style={"label": "Sum", "lw": 2, "color": "black",
                                       "ls": "dashed"})
            plt.legend(title=name)
            ax.set_xlabel("Probability")
            ax.set_ylabel("Predicted label")
        return binned_events
        
    def acc_efficiency(self, auc=False):
        for name, dist in self.distribution.items():
            binned_auc=[]
            predicted_label = np.argmax(dist,1)
            for low,high in zip(self.mass_percentiles[:-1],
                                self.mass_percentiles[1:]):
                mask_mass_bin = (self.conds>=low) & (self.conds<high)
                if auc:
                    binned_auc.append([
                        calculate_auc((self.label[mask_mass_bin]==0)*1,
                                      dist[:,0][mask_mass_bin]),
                        calculate_auc((self.label[mask_mass_bin]==1)*1,
                                      dist[:,1][mask_mass_bin]),
                        calculate_auc((self.label[mask_mass_bin]==2)*1,
                                      dist[:,2][mask_mass_bin])
                    ])
                else:
                    binned_auc.append([
                        np.mean((self.label[mask_mass_bin]==0) 
                                & (predicted_label[mask_mass_bin]==0)),
                        np.mean((self.label[mask_mass_bin]==1) 
                                & (predicted_label[mask_mass_bin]==1)),
                        np.mean((self.label[mask_mass_bin]==2) 
                                & (predicted_label[mask_mass_bin]==2))
                    ])
            binned_auc = np.array(binned_auc)
            
            _, ax = plt.subplots(1,1, figsize = (15,7))
            for i in range(len(binned_auc[0])):
                ax = plot.plot_stairs(binned_auc[:,i], self.mass_percentiles,
                                    normalise=False, ax=ax,
                                    style={"label": columns[2+i], "lw": 2})
            ax = plot.plot_stairs(binned_auc.sum(1)/3 if auc else binned_auc.sum(1),
                                  self.mass_percentiles,normalise=False, ax=ax,
                                style={"label": "norm sum", "lw": 2, "color": "black",
                                       "ls": "dashed"})
            if auc:
                ax.set_ylim([0,1])
                ax.set_ylabel("AUC")
            else:
                ax.set_ylim([0,1])
                ax.set_ylabel("Accuracy")
                print(f"{name} mean accuracy: {np.mean(binned_auc.sum(1))}")
            plt.legend(prop={'size': 12}, title=name)
        return binned_auc, self.mass_percentiles

    def eval_order_change(self, ot_paths, flow_paths, sig_label=None):
        old_disc_func = utils.get_disc_func(self.sig_label)
        original_conds = self.conds.copy()
        original_encodings = self.output["encodings"].copy()
        if sig_label is None:
            sig_label =self.sig_label
        else:
            self.sig_label = sig_label
            self.disc_func = utils.get_disc_func(sig_label)
        min_conds = original_conds[original_encodings[:,self.sig_label]>0.5].min()
        # make fake data
        grid = np.zeros((101, 3))
        grid[:,sig_label] = np.linspace(self.output["encodings"].min(0)[sig_label],
                                        self.output["encodings"].max(0)[sig_label],
                                        len(grid))
        grid_dummy_1 = ((1-grid[:,sig_label])/4)*3
        grid_dummy_2 = ((1-grid[:,sig_label])/4)
        grid_dummy=np.c_[grid_dummy_1,grid_dummy_1,grid_dummy_2]
        grid_dummy[:,sig_label]=0
        grid+=grid_dummy
        # x,y = np.meshgrid(np.linspace(0.01,0.99,11),
        #                   np.linspace(0.01,0.99,11))
        # values = np.stack((np.ravel(y),np.ravel(x)),1)
        # # values = np.exp(values)/np.sum(np.exp(values),1)[:,None]
        # values = values[np.lexsort((values[:,1], values[:,0]))]
        # grid = values[values.sum(1)<1]
        # grid = np.c_[grid, 1-np.sum(grid,1)]
        conds_lst = (
            # np.percentile(truth["mass"], [np.arange(0, 110, 10)])[0]
            np.linspace(min_conds, 300, 11)
            )
        order_change = {"mass": conds_lst}
        reverse_bool=False
        self.output["encodings"] = grid
        for nr,path in tqdm(enumerate(ot_paths), total=len(ot_paths)):
            col_name = f"OT_{path.split('/')[-1].split('_')[-4]}"
            for conds_val in conds_lst:
                self.conds = np.ones(len(grid))*conds_val
                self.run_ot(path, reverse_bool=reverse_bool)

                difference = (np.argsort(self.projected_distributions[col_name])-
                np.argsort(self.disc_func((grid))))
                # print(np.argsort(self.projected_distributions[col_name]))
                mae_diff = np.mean(np.abs(difference))
                if col_name not in order_change:
                    order_change[col_name] = []
                order_change[col_name].append(mae_diff)
                # print(mae_diff)
                # print(np.argsort(self.projected_distributions[col_name])[np.abs(difference)>0])
                # sys.exit()
        for nr,path in tqdm(enumerate(flow_paths), total=len(flow_paths)):
            col_name = f"flow_{path.split('/')[-1].split('_')[-4]}"
            for conds_val in conds_lst:
                self.conds = T.ones(len(grid))*conds_val
                self.run_flow(path, reverse_bool=reverse_bool)
                difference = (np.argsort(self.projected_distributions[col_name])-
                np.argsort(self.disc_func((grid))))
                mae_diff = np.mean(np.abs(difference))
                if col_name not in order_change:
                    order_change[col_name] = []
                order_change[col_name].append(mae_diff)
        self.conds = original_conds
        self.output["encodings"] = original_encodings
        self.disc_func = old_disc_func
        return order_change

if __name__ =="__main__":
    # %matplotlib widget
    save_fig_path = "figures/3d/"
    image_format = ".pdf"
    # image_format = ".png"
    save_fig_path = None #"figures/3d/"
    plot_bool=False
    sig_label=0
    size=None
    style_bkg_rej = {"ls":"dashed", "lw": 2}
    discriminant = utils.get_disc_func(sig_label)
    device="cuda"
    MULTI_CLF = "/srv/beegfs/scratch/groups/rodem/anomalous_jets/taggers/supervised/transformer_multiclass/outputs/combined_scores.h5"
    truth, data = pl.load_multi_cls(MULTI_CLF, upper_mass_cut=450)
    prob_distributions = {"mDNN": truth["encodings"][:size],
                          "labels": np.array(truth["labels"])}
    data = data.to_numpy()[:size]
    mass = data[:size,0]
    bkg_rej = np.round(np.linspace(0.5, 0.999, 11),4)
    bkg_rej = [0.5, 0.9, 0.95, 0.99]
    # bkg_rej = [0.5, 0.85, 0.95, 0.97, 0.99, 0.995]
    if False:
        style={"alpha": 0.2}
        for i in range(1,3):

            plt.plot(truth["encodings"][truth["labels"]==i][:10_000,1],truth["encodings"][truth["labels"]==i][:10_000,2], ".", **style)
        sys.exit()
    metric_cls = Metrics(prob_distributions, conds=mass,
                         disc_func=discriminant,
                         label=np.array(truth["labels"]),
                         sig_label=sig_label,bkg_label=0,
                         n_bins=5, plot_bool=plot_bool,
                         verbose=False, device=device,
                         save_path=save_fig_path
                         )
    OT_PATH = [
        #old
        # "/home/users/a/algren/scratch/trained_networks/decorrelation/OT/3d/gridsearch/OT_2023_05_01_22_54_41_336674_base_3_1",
        # "/home/users/a/algren/scratch/trained_networks/decorrelation/OT/3d/gridsearch/OT_2023_05_01_22_50_14_935605_source_3_1",
        # "/home/users/a/algren/scratch/trained_networks/decorrelation/OT/3d/gridsearch/OT_2023_05_02_21_05_42_997068_base_normal_3_1",

        #new below 300
        # "/home/users/a/algren/scratch/trained_networks/decorrelation/OT/3d/OT_2023_06_19_15_09_23_908285_source_3_1",
        # # # "/home/users/a/algren/scratch/trained_networks/decorrelation/OT/3d/OT_2023_06_19_15_09_24_676774_source_3_1",

        # # "/home/users/a/algren/scratch/trained_networks/decorrelation/OT/3d/OT_2023_06_19_15_09_24_358627_base_uniform_3_1",
        # "/home/users/a/algren/scratch/trained_networks/decorrelation/OT/3d/OT_2023_06_19_15_09_23_905987_base_uniform_3_1",

        # "/home/users/a/algren/scratch/trained_networks/decorrelation/OT/3d/OT_2023_06_19_15_09_26_269137_base_normal_3_1",
        
        # new
        # "/home/users/a/algren/scratch/trained_networks/decorrelation/OT/3d/gridsearch_450/OT_2023_06_20_13_30_15_363616_base_uniform_3_1",
        # "/home/users/a/algren/scratch/trained_networks/decorrelation/OT/3d/gridsearch_450/OT_2023_06_20_13_30_15_368723_base_uniform_3_1",
        # "/home/users/a/algren/scratch/trained_networks/decorrelation/OT/3d/gridsearch_450/OT_2023_06_20_13_30_37_982143_source_3_1",

        # "/home/users/a/algren/scratch/trained_networks/decorrelation/OT/3d/gridsearch_450/OT_2023_06_20_13_30_15_364128_base_normal_3_1",
        # "/home/users/a/algren/scratch/trained_networks/decorrelation/OT/3d/gridsearch_450/OT_2023_06_20_13_30_15_366291_base_normal_3_1",
        "/home/users/a/algren/scratch/trained_networks/decorrelation/OT/3d/gridsearch_450/OT_2023_06_22_12_55_49_533895_base_normal_3_1",
        ]

    FLOW_PATH = [
        # old
        # "/home/users/a/algren/scratch/trained_networks/decorrelation/flow/3d/gridsearch/flow_2023_05_07_18_38_54_487994_normal_3_1",
        # "/home/users/a/algren/scratch/trained_networks/decorrelation/flow/3d/gridsearch/flow_2023_05_07_18_38_55_425998_normal_3_1",
        # "/home/users/a/algren/scratch/trained_networks/decorrelation/flow/3d/gridsearch/flow_2023_05_07_18_38_48_211728_normal_3_1",
        # "/home/users/a/algren/scratch/trained_networks/decorrelation/flow/3d/gridsearch/flow_2023_05_07_18_38_54_788900_normal_3_1",

        # new below 450
        "/home/users/a/algren/scratch/trained_networks/decorrelation/flow/3d/gridsearch_450/new_below_300_flow_2023_06_20_13_31_29_325707_normal_3_1",

        #new below 300
        # "/home/users/a/algren/scratch/trained_networks/decorrelation/flow/3d/new_below_300_flow_2023_06_19_15_17_04_151026_normal_3_1",
        ]
    
    # FLOW_PATH+=glob("/home/users/a/algren/scratch/trained_networks/decorrelation/flow/3d/gridsearch_450_boot/*")

    # fig_order,ax_order = plt.subplots(1,3, figsize=(3*8, 6))
    # disc_name = ["QCD discriminate", "Top discriminate","W discriminate"]
    disc_name = [r"$\mathcal{D}_\mathrm{QCD}$",
                 "$\mathcal{D}_\mathrm{T}$",
                 "$\mathcal{D}_\mathrm{VB}$"]
    
    for sig_label, legend_title in enumerate(disc_name):
        metric_cls.project_distribution(utils.get_disc_func(sig_label), sig_label)
        # order_change = metric_cls.eval_order_change(OT_PATH, FLOW_PATH, sig_label=sig_label)
        # for key, value in order_change.items():
        #     if key is "mass":
        #         continue
        #     ax_order[sig_label].plot(order_change["mass"], value, label=key)
        #     ax_order[sig_label].set_ylabel("Mean index change")
        #     # ax_order[sig_label].set_xlabel("Mass GeV")
        # ax_order[sig_label].legend()
        # ax_order[sig_label].set_title(legend_title)
        # plt.tight_layout()
        # sys.exit()
        for path in OT_PATH:
            name =f"OT-mDNN {path.split('/')[-1].split('_')[-3]}"
            name = name.replace("uniform", "Dir(1,1,1)")
            metric_cls.run_ot(path, col_name=name)
        for nr, path in enumerate(FLOW_PATH):
            name=None
            if len(FLOW_PATH)==1:
                    name = "cf-mDNN"
            metric_cls.run_flow(path, col_name=name)
        # break
        # metric_cls.plot_rocs(one_v_all=False)
        fig, ax = plt.subplots(1,2,figsize=(8*2,6))
        nr=0
        for i in metric_cls.projected_distributions:
            if i == "labels":
                continue
            jsd_lst_disco, sig_eff, background_rej = metric_cls.bkg_rej_calculation(
                i, proba_flat = metric_cls.projected_distributions[i],
                # save_fig_path = f"{save_fig_path}/bkg_hist_{i}_{sig_label}{image_format}",
                legend_kwargs={"title": f"{i}: {legend_title}"},
                background_rej=bkg_rej
                )
            jsd_lst_disco = 1/jsd_lst_disco
            # if "flow" in i:
            #     color = "black"
            # else:
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
            
            nr+=1

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
            misc.save_fig(fig, f"{save_fig_path}/bkg_reject_label_{sig_label}{image_format}")

    # if plot_bool:
    #     misc.save_fig(fig_order, f"{save_fig_path}/order_preservation{image_format}")


    # Plot the grid warped by the conjugate potential's gradient

    # metric_cls.plot_conds_distribution()
    # metric_cls.probability_pr_bin([0,75,130, 250, 400])
    # metric_cls.probability_pr_bin()
    # metric_cls.plot_rocs()
    # binned_events = metric_cls.label_pr_bin()
    # metric_cls.acc_efficiency()
    # metric_cls.acc_efficiency(True)
    if True:
        mass=80.377
        # mass=127
        size = 31
        proba_dummy_prob, data, cs_ot, cs_flow = transport_map(metric_cls.generator,
                                                               metric_cls.flow, size=size,
                                                               save_fig_path=save_fig_path,
                                                               image_format=image_format,mass=mass)
        if size<102:
            data["init_loc"] = proba_dummy_prob
            # np.save('arrows.npy',output)
            # data2=np.load('arrows.npy', allow_pickle=True).item()
            key = 'flow'
            arrows_flow = np.concatenate([data['init_loc'][:,1:],data['init_loc'][:,1:]+data['regular'][key][:,1:]],axis=-1)
            arrows_to_arrows = np.concatenate([np.tile(arrows_flow[:,np.newaxis],[1,len(arrows_flow),1]),np.tile(arrows_flow[np.newaxis,:,:],(len(arrows_flow),1,1))],axis=-1)
            flow_mask = np.sqrt(np.sum(arrows_flow**2,axis=-1)) > -999 # 5e-2
            flow_para_c, flow_intersect_c, flow_subset_c, flow_angle_c = utils.check_intersection(arrows_to_arrows.reshape(-1,8))
            
            key = 'OT'
            arrows_ot = np.concatenate([data['init_loc'][:,1:],data['init_loc'][:,1:]+data['regular'][key][:,1:]],axis=-1)
            arrows_to_arrows = np.concatenate([np.tile(arrows_ot[:,np.newaxis],[1,len(arrows_ot),1]),np.tile(arrows_ot[np.newaxis,:,:],(len(arrows_ot),1,1))],axis=-1)
            ot_mask = np.sqrt(np.sum(arrows_ot**2,axis=-1)) > -999# 5e-2
            
            # OT_para, OT_intersect, OT_subset, OT_angle = utils.check_intersection(arrows_to_del.reshape(-1,8))
            OT_para_c, OT_intersect_c, OT_subset_c, OT_angle_c = utils.check_intersection(arrows_to_arrows.reshape(-1,8))
            # OT_angle = check_intersection_angle(arrows_to_arrows.reshape(-1,12))
                    
            OT_arr_c = np.nan_to_num((1-OT_para_c) * OT_intersect_c * OT_angle_c)
            flow_arr_c = np.nan_to_num((1-flow_para_c) * flow_intersect_c * flow_angle_c)
            OT_arr_c= OT_arr_c.reshape(len(arrows_ot),len(arrows_ot)) * (ot_mask.reshape(len(arrows_ot),1) | ot_mask.reshape(1,len(arrows_ot)))
            flow_arr_c = flow_arr_c.reshape(len(arrows_ot),len(arrows_ot)) * (flow_mask.reshape(len(arrows_ot),1) | flow_mask.reshape(1,len(arrows_ot)))
            
            fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(8*2,6), sharey=True)
            flow_lists = (flow_arr_c > np.pi/2).any(axis=-1) #flow_del_cross & (flow_angle > np.pi/32)#(flow_angle > np.pi/16)
            ot_lists = (OT_arr_c > np.pi/2).any(axis=-1) #OT_del_cross & (OT_angle > np.pi/32)#OT_del_cross#.all(axis=1)#(flow_angle > np.pi/4)#.reshape(len(arrows),len(arrows)).any(axis=1)#flow_del_cross#.all(axis=1)
            dim1 = 1
            dim2 = 2
            labels=["OT-mDNN normal", "cf-mDNN"]
            nr=0
            for a,key,sel in zip(ax,["OT", "flow"],[ot_lists,flow_lists]):
                a.quiver(data['init_loc'][~sel,dim1],
                        data['init_loc'][~sel,dim2],
                        data['regular'][key][~sel,dim1],
                        data['regular'][key][~sel,dim2], 
                        color=plot.COLORS[nr], 
                        units="xy", angles='xy', scale_units='xy', scale=1,alpha=0.3,label='No cross $> \pi/2$')

                a.quiver(data['init_loc'][sel,dim1],
                        data['init_loc'][sel,dim2],
                        data['regular'][key][sel,dim1],
                        data['regular'][key][sel,dim2], units="xy", angles='xy', scale_units='xy', scale=1,alpha=0.3,color='red',label='Cross $> \pi/2$')

                a.legend(title=f'{labels[nr]}, frac={np.sum(sel)/len(sel):.2f}',frameon=False)
                nr+=1
            ax[1].set_xlabel(r'$p_{VB}$')
            ax[0].set_xlabel(r'$p_{VB}$')
            ax[0].set_ylabel(r'$p_{Top}$')
            if save_fig_path is not None:
                misc.save_fig(fig, f"{save_fig_path}/angles_{mass}{image_format}",
                        tight_layout=True)
        

