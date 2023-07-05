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
from src.trainer import Training



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

    train, test_source = train_test_split(data, test_size=0.1)
    
    log.info(f"Training size: {train.shape}")
    log.info(f"Test size: {test_source.shape}")

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
        target_loader.sample(test_source[:,:config.noncvx_dim])
        test_target = target_loader.data.clone()
        target_loader.sample(train[:,:config.noncvx_dim])
    elif config.target_distribution=="source":
        target_loader.sample(test_source[:,:config.noncvx_dim],
                             test_source[:,config.noncvx_dim:])
        test_target = target_loader.data.clone()
        target_loader.sample(train[:,:config.noncvx_dim], train[:,config.noncvx_dim:])
    else:
        raise ValueError("Unknown distribution")

    # define networks
    w_disc = PICNN(**config.model_args)
    generator = PICNN(**config.model_args)
    w_disc.set_standard_parameters(target_loader.data[:,config.noncvx_dim:],
                                   target_loader.data[:,:config.noncvx_dim])
    generator.set_standard_parameters(train[:,config.noncvx_dim:],
                                   train[:,:config.noncvx_dim])

    # schedulers
    if config.train_args.learning_rate_scheduler:
        config.train_args["sch_args_f"] =  {
            "T_max": config.train_args.nepochs*config.train_args["epoch_size"],
            "eta_min": 5e-5}
        config.train_args["sch_args_g"] =  {
            "T_max": config.train_args.nepochs*config.train_args["epoch_size"],
            "eta_min": 5e-5}

    # train setup
    ot_training = Training(f_func=w_disc,g_func=generator,
                        outdir=config.save_path,
                        device=config.device,
                        distribution=base_distribution,
                        **config.train_args)
    
    utils.save_config(
        outdir=config.save_path,  # save config%
        values=config.model_args,
        drop_keys=[],
        file_name="model_config",
    )

    pbar = tqdm(range(0, config.train_args.nepochs))

    for ep in pbar:
        ot_training.evaluate(test_source, test_target, ep,
                             datatype=config.train_args.datatype
                            )
        ot_training.step(sourcedset=source_loader,
                        targetdset=target_loader,
                        pbar=pbar)

        

if __name__ == "__main__":
    main()
