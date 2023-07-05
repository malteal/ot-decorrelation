"plotting the order preservation plots"
import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)


import numpy as np
import src.pipeline as pl
import src.eval_utils as eval_utils
import src.plotting as plot
import src.utils as utils


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


def transport_map(ot_model, flow_model, size=301, save_fig_path=None,
                  FIG_TYPE=".pdf", mass=127):

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
    d_dummy_prob = utils.logit(d_meshgrid)
    d_dummy_prob = T.tensor(d_dummy_prob,requires_grad=True).float().to(device)
    # d_meshgrid = meshgrid*(1+np.random.uniform(-0.01,0.01, (len(meshgrid),3)))
    #new
    r_sq_new = sample_small_spherical_change(len(meshgrid))
    d_meshgrid_new = np.clip(meshgrid+r_sq_new/1000, 0.0001, 0.9999)
    d_meshgrid_new = d_meshgrid_new/d_meshgrid_new.sum(1)[:,None]
    d_dummy_prob_new = utils.logit(d_meshgrid_new)
    d_dummy_prob_new = T.tensor(d_dummy_prob_new,requires_grad=True).float().to(device)

    # dummy_prob = utils.logit(np.random.dirichlet((1,1,1), 3000))
    dummy_prob = utils.logit(meshgrid)
    dummy_prob = T.tensor(dummy_prob,requires_grad=True).float().to(device)
    proba_dummy_prob = utils.probsfromlogits(dummy_prob.cpu().detach().numpy())

    
    dummy_conds = T.tensor([np.log(mass)]*len(dummy_prob)).float().view(-1,1).to(device)
    output={i:{} for i in ["regular", "delta"]}
    for i,dist in zip(["regular", "delta"], [dummy_prob, d_dummy_prob]):
        ot_output = ot_model.chunk_transport(dummy_conds, dist).cpu().detach().numpy()
        output_flow = [flow_model._transform.inverse(i, j)[0].cpu().detach().numpy() for i,j in zip(dist.chunk(20), dummy_conds.chunk(20))]
        output_flow = np.concatenate(output_flow, 0)
        ot_output = utils.probsfromlogits(ot_output)
        output_flow = utils.probsfromlogits(output_flow)
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
            utils.save_fig(fig, f"{save_fig_path}/hist_of_diff_{mass}_{FIG_TYPE}")
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
                utils.save_fig(fig, f"{save_fig_path}/displacement_plots_mass_{mass}_{name}{FIG_TYPE}")

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
        utils.save_fig(fig_ot, f"{save_fig_path}/magnitude_differences_mass_{mass}{FIG_TYPE}",
                      tight_layout=False)
    return proba_dummy_prob, output, cs_ot, cs_flow


if __name__ == "__main__":
    mass=80.377
    # mass=127
    size = 31
    proba_dummy_prob, data, cs_ot, cs_flow = transport_map(metric_cls.generator,
                                                            metric_cls.flow, size=size,
                                                            save_fig_path=save_fig_path,
                                                            FIG_TYPE=FIG_TYPE,mass=mass)
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
            utils.save_fig(fig, f"{save_fig_path}/angles_{mass}{FIG_TYPE}",
                    tight_layout=True)
    