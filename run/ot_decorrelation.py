"OT decorrelation"
import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

from tqdm import tqdm
import logging
log = logging.getLogger(__name__)

import numpy as np
import hydra
from omegaconf import OmegaConf,DictConfig

import torch as T

from sklearn.model_selection import train_test_split

import src.pipeline as pl
import src.loaders as loaders
from src.PICNN.PICNN import PICNN
import src.utils as utils 
import src.eval_utils as eval_utils  

from otcalib.torch.run import Training
from otcalib.utils import plotutils


@hydra.main(config_path=f"{root}/configs", config_name="ot_config", version_base=None)
def main(config: DictConfig):
    OmegaConf.set_struct(config, False)

    # define data
    output = pl.load_data(config.cvx_dim, config.get("multi_clf", None))

    log.info(f"Saving at {config.save_path}")

    if config.model_args.logit:
        log.info("Transform to logit space")
        output["encodings"] = utils.logit(output["encodings"])
        output["encodings"] = np.clip(output["encodings"], -15, 15)

    data = T.tensor(np.c_[output["mass"], output["encodings"]],
                    requires_grad=True).float()

    train, test = train_test_split(data, test_size=0.1)
    
    log.info(f"Training size: {train.shape}")
    log.info(f"Test size: {test.shape}")

    # dataloaders
    source_loader = loaders.Dataset(train,
                                 nr_convex_dimensions=config.cvx_dim,
                                 nr_nonconvex_dimensions=config.noncvx_dim,
                                 batch_size=config.train_args.batch_size,
                                 device=config.device)


    base_distribution = loaders.get_base_distribution(config.cvx_dim,
                                                      config.target_distribution,
                                                      logit=config.model_args.logit, 
                                                      device=config.device)

    target_loader = loaders.BaseDistribution(base_distribution,
                                            device = config.device,
                                            batch_size=config.train_args.batch_size,
                                            dims=config.cvx_dim
                                            )
    # sample either from pdf or source distribution 
    if "base" in config.target_distribution:
        target_loader.sample(test[:,:config.noncvx_dim])
        truth = target_loader.data.clone()
        target_loader.sample(train[:,:config.noncvx_dim])
    elif config.target_distribution=="source":
        target_loader.sample(test[:,:config.noncvx_dim], test[:,config.noncvx_dim:])
        truth = target_loader.data.clone()
        target_loader.sample(train[:,:config.noncvx_dim], train[:,config.noncvx_dim:])
    else:
        raise ValueError("Unknown distribution")

    # eval data
    eval_data = utils.create_eval_dict(truth, test, names=["truth","source"],
                                       cvx_dim=config.cvx_dim,
                                       noncvx_dim=config.noncvx_dim,
                                    #    percentiles=[0,25,50,75,100] if config.target_distribution=="source" else None,
                                       with_total=True
                                       )
    # define networks
    w_disc = PICNN(**config.model_args)
    generator = PICNN(**config.model_args)
    w_disc.set_standard_parameters(eval_data["total"]["truth"]["transport"],
                                   eval_data["total"]["truth"]["conds"])
    generator.set_standard_parameters(eval_data["total"]["source"]["transport"],
                                      eval_data["total"]["source"]["conds"])

    # schedulers
    if config.train_args.learning_rate_scheduler:
        config.train_args["sch_args_f"] =  {
            "name": "CosineAnnealingLR",
            "args": {"T_max": config.train_args.nepochs*config.train_args["epoch_size"],
                     "eta_min": 5e-5}}
        config.train_args["sch_args_g"] =  {
            "name": "CosineAnnealingLR",
            "args": {"T_max": config.train_args.nepochs*config.train_args["epoch_size"],
                     "eta_min": 5e-5}}

    # train setup
    ot_training = Training(f_func=w_disc,g_func=generator,
                        outdir=config.save_path,
                        device=config.device,
                        distribution=base_distribution,
                        **config.train_args)

    pbar = tqdm(range(0, config.train_args.nepochs))
    # ot_training.pretrain_models(source_loader,
    #                             target_loader,)
    eval_utils.plot_training_setup(
            source_loader,
            target_loader,
            config.save_path,
            plot_var=["mass"]+[f"encodings_{i}" for i in range(generator.convex_layersizes[0])],
            eval_data=eval_data,
            generator=generator,
            DL1r=False
        )
    ot_training.distribution = target_loader
    if "base" in config.target_distribution:
        eval_kwargs = {
            "eval_performance_str": "log_likelihood_eval",
            "logp": True,
            "save_all":True,
        }
    else:
        eval_kwargs = {
            "eval_performance_str": "AUC",
            "discriminator_str": "source",
            "logp":False,
            "save_all":True,
        }
    for ep in pbar:
        ot_training.evaluate(eval_data.copy(), ep,
                             datatype=config.train_args.datatype,
                            plot_figures=config.cvx_dim == 10, run_metrics=True,
                            plot_style={"style_target": utils.STYLE_TARGET,
                                        "style_source": utils.STYLE_SOURCE,
                                        "style_trans": utils.STYLE_TRANS},
                            **eval_kwargs
                            )
        ot_training.step(sourcedset=source_loader,
                        targetdset=target_loader,
                        pbar=pbar)

        

if __name__ == "__main__":
    main()
