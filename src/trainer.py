"""This scripts will run the training loop for optimal transport"""
from time import time
import json
import numpy as np
import os
import logging
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import src.utils as utils
from src.dense_net import DenseNet



class Training:
    "Activate the training of OT"
    def __init__(  
        self,
        f_func: callable,
        g_func: callable,
        device:str = "cuda",
        verbose: bool = False,
        **kwargs
    ) -> None:
        """Initialize the start parameter of OT training

        Parameters
        ----------
        f_func : callable
            ML model
        g_func : callable
            ML model
        conds_bins : list
            list of conditional bins to evaluate in. Only works for 1d conds
        outdir : str, optional
            output folder , by default "tb"
        verbose : bool, optional
            How much to print, by default False
        """
        self.lr_f = kwargs.get("lr_f", 1e-4)
        self.lr_g = kwargs.get("lr_g", 1e-4)
        self.loss_li_ratio = kwargs.get("loss_li_ratio", 0)
        self.loss_wasser_ratio = kwargs.get("loss_wasser_ratio", 1)
        self.f_per_g = kwargs.get("f_per_g", 1)
        self.g_per_f = kwargs.get("g_per_f", 1)
        self.reverse_ratio = kwargs.get("reverse_ratio", 1)
        self.grad_clip = kwargs.get("grad_clip", {"f": 5, "g": 5})
        self.grad_norm = kwargs.get("grad_norm", {"f": 0, "g": 0})
        self.epoch_size = kwargs.get("epoch_size", 100)
        self.burn_in = kwargs.get("burn_in", 50)
        self.optimizer_args_f = kwargs.get("optimizer_args_f",{"lr": self.lr_f,
                                                               "betas":(0.0, 0.9),
                                                               })
        self.optimizer_args_g = kwargs.get("optimizer_args_g",{"lr": self.lr_g,
                                                               "betas":(0.0, 0.9),
                                                               })
        self.sch_args_g = kwargs.get("sch_args_g", None)
        self.sch_args_f = kwargs.get("sch_args_f", None)
        self.outdir = kwargs.get("outdir", "/")

        self.distribution = kwargs.get("distribution", None)
        self.log = kwargs.get("log", {"steps_f":[0],"steps_g":[0]})

        self.device = device
        self.verbose = verbose
        self.f_func = f_func
        self.g_func = g_func

        self.noncvx_dim = 0 if not hasattr(self.f_func, "weight_uutilde") else self.f_func.weight_uutilde[0].shape[1]
        self.cvx_dim = self.f_func.weight_zz[0].shape[1]
        self.distnames = kwargs.get("distnames", [f"dist_{i}" for i in range(self.cvx_dim)])
        self.condsnames = kwargs.get("condsnames", [f"conds_{i}" for i in range(self.noncvx_dim)])
        self.conds_bins = kwargs.get("conds_bins", {"":[-1000, 1000]})

        # init standard optimizer & scheduler
        self.f_func_optim= torch.optim.AdamW(
            self.f_func.parameters(), **self.optimizer_args_f
            )
        self.g_func_optim = torch.optim.AdamW(
            self.g_func.parameters(), **self.optimizer_args_g
            )
            # , self.sch_args_f
        self.f_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.f_func_optim,
                                                                      **self.sch_args_f)
        self.g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.g_func_optim,
                                                                      **self.sch_args_g)

        # define discriminator
        self.dense_net_args = {
            "input_dim": f_func.convex_layersizes[0]+f_func.nonconvex_layersizes[0],
            "N": 64,
            "n_layers": 4,
            "activation_str": "leaky_relu",
            "output_size": 1,
            "device": self.device,
            "sigmoid": False,
            "adam": True, #using SGD
            }

        self.discriminator = DenseNet(**self.dense_net_args)

        if not os.path.exists(self.outdir):
            self.save_config()

        # creating folders
        os.makedirs(self.outdir+"/training_setup", exist_ok = True)
        os.makedirs(self.outdir+"/plots", exist_ok = True)
        os.makedirs(self.outdir+"/plots/discriminator", exist_ok = True)
        
        # save setup
        utils.save_config(
            outdir=self.outdir,
            values=self.__dict__.copy(),
            drop_keys=["metric", "discriminator", "conds_bins", "distribution",
                       "f_func_optim", "f_scheduler", "f_func",
                       "g_func_optim", "g_scheduler", "g_func",
                       ],
            file_name="train_config",
        )

    def load_state(self, old_model, last_epoch):
        "load old state of optimizer and scheduler"
        self.f_func_optim.load_state_dict(torch.load(f"{old_model}/training_setup/f_func_optim_{last_epoch}.pth"))
        self.g_func_optim.load_state_dict(torch.load(f"{old_model}/training_setup/g_func_optim_{last_epoch}.pth"))
        self.f_scheduler.load_state_dict(torch.load(f"{old_model}/training_setup/f_func_scheduler_{last_epoch}.pth"))
        self.g_scheduler.load_state_dict(torch.load(f"{old_model}/training_setup/g_func_scheduler_{last_epoch}.pth"))


    def load_model(self,f_path:str, g_path:str):
        self.f_func.load(f_path)
        self.g_func.load(g_path)
    
    def evaluate( # pylint: disable=too-many-locals,too-many-arguments
        self,
        test_source,
        test_target,
        epoch: int,
        datatype: str="",
    ):
        """
        evaluate model performance

        Parameters
        ----------
        data : dict
            list of torch.Tensor of source, target, valid_source, valid_target
        epoch : int
            the epoch number
        datatype : str
            toy or cern
        Returns
        -------
        dict
            return the updated log dict
        """
        if self.verbose:
            logging.info("Evaluationing performance")
        self.g_func.eval()
        self.f_func.eval()

        # evaluate performance
        metric = {}

        # classification
        # Transport values
        n_chunks = len(test_source)//10_000
        transport = self.g_func.chunk_transport(test_source[:, :self.noncvx_dim],
                                                test_source[:, self.noncvx_dim:],
                                                n_chunks=n_chunks,
                                                )
        test_transport = torch.concat([test_source[:,:self.noncvx_dim], transport], 1)
                
        if (epoch>self.burn_in): 
            test_transport = torch.concat([test_transport, torch.zeros(len(test_target),1)], 1)
            test_target = torch.concat([test_target, torch.ones(len(test_target),1)], 1)
            
            # discriminator evaluation - matching conds
            train_data, test_data = train_test_split(test_target, test_size=0.25,
                                           random_state=42)
            train_trans, test_trans = train_test_split(test_transport, test_size=0.25,
                                           random_state=42)
            train = torch.ones((2*len(train_data), train_data.shape[1]))
            test = torch.ones((2*len(test_data), test_data.shape[1]))
            train[::2] = train_data
            train[1::2] = train_trans
            test[::2] = test_data
            test[1::2] = test_trans

            dataloader_args = {"batch_size":512, "drop_last":False, "shuffle":False}
            train_dataloader_clf = DataLoader(train.detach(), **dataloader_args)
            valid_dataloader_clf = DataLoader(test.detach(), **dataloader_args)

            self.discriminator.init_new_log_dict()
            self.discriminator.run_training(
                            train_dataloader_clf,
                            valid_loader=valid_dataloader_clf,
                            n_epochs=50 if epoch-1 == self.burn_in else 2,
                            verbose=epoch-1 == self.burn_in,
                            )
            self.discriminator.set_optimizer(True, lr=(1e-5 if "prob" in datatype
                                                       else 1e-4)) # changing optimizer to sgd

            # auc = np.max(self.discriminator.loss_data["valid_auc"])
            metric["AUC"] = np.max(self.discriminator.loss_data["valid_auc"])
            metric["train_AUC"] = self.discriminator.evaluate_performance(train_dataloader_clf, log_val_bool=False)[0]
            self.discriminator.plot_log(f"{self.outdir}/plots/discriminator/", ep_nr=epoch) # TODO plot each loss batch instead of epoch

        else:    
            metric["AUC"] = 1.0
            metric["train_AUC"] = 1.0

        # update logging
        self._update_log(metric, epoch)

        torch.cuda.empty_cache()
        if isinstance(self.outdir, str):
            self.save(epoch=epoch)


        return metric

    def _update_log(# pylint: disable=too-many-arguments
        self,
        logging_values,
        epoch_nr:int=-1
    ) -> dict:
        """Saving training information

        Parameters
        ----------
        logging_values : list
            list of list of values that should be logged

        Returns
        -------
        dict
            return dict where the new metrics have been added
        """
        if "epoch" in logging_values:
            logging_values.pop("epoch")

        if epoch_nr == 0:
            for i in logging_values.keys():
                if isinstance(logging_values[i], dict):
                    self.log[i] = {}
                    for j in logging_values[i].keys():
                        self.log[i][j] = {}
                        for k in logging_values[i][j].keys():
                            self.log[i][j][k] = []
                else:
                    self.log[i] = []
        else:
            keys = np.array(list(logging_values.keys()))

            sub_keys = keys[np.in1d(keys, list(self.log.keys()))]
            mask_dict = np.array([isinstance(logging_values[i], dict) for i in sub_keys])
            if (len(sub_keys) > 0) and any(mask_dict):
                for i in sub_keys[mask_dict]:
                    sub_keys_logging = np.array(list(logging_values[i].keys()))
                    sub_keys_logging = sub_keys_logging[~np.in1d(sub_keys_logging, list(self.log[i].keys()))]
                    for j in sub_keys_logging:
                        self.log[i][j] = []

            add_keys = keys[~np.in1d(keys, list(self.log.keys()))]
            for i in add_keys:
                if isinstance(logging_values[i], dict):
                    self.log[i] = {}
                    for j in logging_values[i].keys():
                        if True:#isinstance(logging_values[i][j], dict):
                            self.log[i][j] = {}
                            for k in logging_values[i][j].keys():
                                self.log[i][j][k] = []
                        else:
                            self.log[i][j] = []
                else:
                    self.log[i] = []

        
        for i,j in logging_values.items():
            if isinstance(j, dict):
                for j,k in j.items():
                    if isinstance(k, dict):
                        for k,l in k.items():
                            self.log[i][j][k].append(l if isinstance(l, list) else np.float64(l))
                    else:
                        self.log[i][j].append(k if isinstance(k, list) else np.float64(k))
            else:
                self.log[i].append(j if isinstance(j, list) else np.float64(j))
        
    # @staticmethod
    def dual_formulation(self, model_disc:torch.nn.Module,
                         model_transport:torch.nn.Module,
                         source:list, target:list, run_disc:bool):
        """Kantorovich dual formulation

        Parameters
        ----------
        model_disc : torch.nn.Module
            discriminator model
        model_transport : torch.nn.Module
            Generator
        source : list
            condtions, transport_var & sig_mask
        target : list
            condtions, target_var & sig_mask
        run_disc : bool
            Which formulation to run. Train either f or g

        Returns
        -------
        _type_
            _description_
        """
        conditionals, totransport, signal_bool = source
        targetconds, targettrans, _ = target
        signal_bool = signal_bool.flatten()
        
        # nabla g(x, theta), g(x, theta)
        trans, _ = model_transport.transport(
            conditionals, totransport, signal_bool,
        )
        # nabla f(nabla g(x, theta), theta), f(nabla g(x, theta), theta)
        cvx_disc_trans = model_disc(conditionals, trans)

        if run_disc:
            # nabla f(x, theta'), f(x, theta')
            cvx_disc = model_disc(targetconds,targettrans)
            loss = cvx_disc.mean() - cvx_disc_trans.mean()
        else:
            # nabla f(nabla g(x, theta), theta), f(nabla g(x, theta), theta)
            loss = cvx_disc_trans - torch.sum(  # pylint: disable=E1101
                trans * totransport, keepdim=True, dim=1
            )
            loss = loss.mean()

        return loss, cvx_disc_trans

    def save(self, epoch:int) -> None:
        """save models and info

        Parameters
        ----------
        epoch : int
            The epoch number
        """
        # save log file
        with open(self.outdir + "/log.json", "w", encoding="utf-8") as file_parameters:
            json.dump(self.log, file_parameters)
        
        # save network states
        states_to_save = [self.f_func_optim.state_dict(),
                        self.g_func_optim.state_dict(),
                        self.f_func.state_dict(),
                        self.g_func.state_dict()]

        if self.f_scheduler is not None:
            states_to_save.extend(self.f_scheduler.state_dict())
        if self.g_scheduler is not None:
            states_to_save.extend(self.g_scheduler.state_dict())

        save_state = {}
        for i, name in zip(states_to_save, ["f_optimizer",
                                            "g_optimizer",
                                            "f_func",
                                            "g_func",
                                            "f_scheduler",
                                            "g_scheduler"]):
            save_state[name] = i
        torch.save(save_state, f"{self.outdir}/training_setup/checkpoint_{epoch}.pth")

    def step(self, sourcedset: iter, targetdset: iter, pbar:callable = None) -> dict:
        """run a epoch

        Parameters
        ----------
        sourcedset : iter
            source dataset
        targetdset : iter
            target dataset
        epoch : int
            the epoch number

        Returns
        -------
        dict
            return the information dict
        """
        start_time = time()
    
        loss_dict_all = {}
        
        # load_time={"data":[], "loss":[]}
        loss_dict_iter = {
            "f":{"loss":[], "lr":[], "cycle_error": [], "clip":[]},
            "g":{"loss":[], "lr":[], "cycle_error": [], "clip":[]}
                        }
        optimization_order = [False, True]
        for ep in range(self.epoch_size):
            
            for nr, (iterations, wdist) in enumerate(zip([self.g_per_f, self.f_per_g],
                                                         optimization_order)):
                model_name = "g" if nr==0 else "f"
                for _ in range(iterations):

                    if nr==0: # g network
                        self.g_func_optim.zero_grad(set_to_none=True)
                        self.f_func.eval()
                        self.g_func.train()
                    else: # f network
                        self.f_func.train()
                        self.g_func.eval()
                        self.f_func_optim.zero_grad(set_to_none=True)

                    # iterate over data
                    source = next(sourcedset)
                    target = next(targetdset)

                    loss, disc_cvx_trans = self.dual_formulation(self.f_func, self.g_func, source, target, wdist)

                    
                    loss.backward()

                    # gradient step for networks
                    if nr==0: # g network
                        clip = torch.nn.utils.clip_grad_norm_(self.g_func.parameters(),
                                                              self.grad_clip["g"])
                        self.g_func_optim.step()
                    else: # f network
                        clip = torch.nn.utils.clip_grad_norm_(self.f_func.parameters(),
                                                              self.grad_clip["f"])
                        self.f_func_optim.step()
                    
                    #log values
                    loss_dict_iter[model_name]["loss"].append(
                        loss.detach()
                        )
                    loss_dict_iter[model_name]["clip"].append(
                        clip
                        )

                if nr==0: # always start by training g
                    self.g_scheduler.step()
                else:
                    self.f_scheduler.step()

        # update log
        for model_name in ["f", "g"]:
            loss = torch.stack(loss_dict_iter[model_name]["loss"]).cpu().numpy()
            clip = torch.stack(loss_dict_iter[model_name]["clip"]).cpu().numpy()
            loss_dict_all[f"loss_{model_name}"] = np.mean(loss)
            loss_dict_all[f"loss_{model_name}_abs_log"] = np.log(np.mean(np.abs(loss)))
            loss_dict_all[f"{model_name}_clip"]=np.mean(clip)
            loss_dict_all[f"lr_{model_name}"] = self.g_func_optim.param_groups[0]["lr"]
            loss_dict_all[f"lr_{model_name}"] = self.f_func_optim.param_groups[0]["lr"]
            loss_dict_all[f"steps_{model_name}"] = self.log[f"steps_{model_name}"][-1]+self.g_per_f*self.f_per_g*self.epoch_size

        self._update_log(loss_dict_all)
        
        if self.verbose:
            print("Epoch time: ", time() - start_time)

        if pbar is not None:
            pbar.set_postfix({
                'Log loss F': f"{str(round(loss_dict_all['loss_f_abs_log'],3))}",
                'Loss G': f"{str(round(loss_dict_all['loss_g'],3))}"})
        
        if (np.isnan(loss_dict_all['loss_g']) or np.isnan(loss_dict_all['loss_f'])
            or loss_dict_all['loss_f_abs_log']>30):
            raise ValueError("NaN in loss functions")
                             
        return pbar
