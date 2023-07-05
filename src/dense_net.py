import os
from torch import nn
import torch
import copy
import numpy as np
from tqdm import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt
from torch.nn import BCELoss
from sklearn.model_selection import train_test_split
from src.PICNN.layers import activation_functions
import src.utils as utils

def get_densenet(input_dim, output_dim, n_neurons, n_layers, activation_str,
                 batchnorm, **kwargs):
    activation = activation_functions(activation_str)
    
    network = [nn.Linear(input_dim, n_neurons), activation]
    
    if batchnorm:
        network.append(nn.BatchNorm1d(num_features=n_neurons))
        hidden_layers = [[nn.Linear(n_neurons, n_neurons),
                          nn.BatchNorm1d(num_features=n_neurons),
                          activation] for _ in range(n_layers)]
    else:
        hidden_layers = [[nn.Linear(n_neurons, n_neurons), activation]
                         for _ in range(n_layers)]
    
    # extend with hidden layers
    network.extend(list(np.ravel(hidden_layers)))
    
    # output layer
    network.extend([nn.Linear(n_neurons, output_dim)])

    return network

class DenseNet(nn.Module):
    """A very simple sense network that is initialised with an input and output dimension."""

    def __init__(self, input_dim, N=32, n_layers=4, sigmoid=False,
                 activation_str = "leaky_relu", output_dim=1,
                 device="cpu", network_type="clf", **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.device = device
        self.hidden_features = N
        self.network_type=network_type
        self.train_loader = None
        self.valid_loader = None
        self.eval_str="valid_auc"
        self.dataloader_args = {"pin_memory":True,"batch_size":512,
                                "shuffle":True, "drop_last": True}
        self.input_dim=input_dim
        network = get_densenet(input_dim, output_dim, N,
                               n_layers, activation_str,
                               batchnorm=kwargs.get("batchnorm", True))

        if self.network_type == "clf":
            if sigmoid:
                network.append(nn.Sigmoid())
                self.loss = BCELoss()
            else:
                self.loss = torch.nn.BCEWithLogitsLoss()
        elif self.network_type=="regression":
            self.loss = torch.nn.MSELoss()
        else:
            raise NameError("network_type is unknown. clf or regression!")

        self.network = nn.Sequential(*network).to(device)
        # eval
        self.best_predictions = None
        self.valid_truth_for_best=None
        
        # set optimizer
        self.set_optimizer(kwargs.get("adam", True),
                           kwargs.get("lr", 1e-3))
        # define loss
        self.state_of_best=None
        self.nr_best_epoch=0
        self.init_new_log_dict()
    
    def set_optimizer(self, adam, lr=1e-3):
        if adam:
            self.optimizer = torch.optim.Adam(self.parameters(), lr)
        else:
            self.optimizer = torch.optim.SGD(self.parameters(), lr)

    def init_new_log_dict(self):
        "Creating logging dict"
        self.loss_data = {f"{i}_{j}":[] for i in ["train", "valid"] for j in ["loss", "auc"]}
        self.loss_data["lr"] = []


    def load(self, path, key_name="model"):
        model_state = torch.load(path, map_location =self.device)
        self.load_state_dict(model_state[key_name])
        if "optimizer" in model_state:
            self.optimizer.state_dict = model_state["optimizer"]
        if "loss" in model_state:
            self.loss_data = model_state["loss"]
        self.network.eval()

    def save(self, output_path:str, **kwargs):
        """save state_dict

        Parameters
        ----------
        output_path : str
            path to output the saved model should end with .pt or .pth
        kwargs : 
            Should be state_dict values or loss like values
        """

        state_dict = {i: kwargs[i] for i in kwargs}
        state_dict["model"] = self.state_dict()
        state_dict["optimizer"] = self.optimizer.state_dict()
        state_dict["loss"] = self.loss_data
        torch.save(state_dict, output_path)
    
    def create_loaders(self, train, valid=None, valid_args:dict={}):
        "valid_args has to be args of train_test_split"
        if len(valid_args) and isinstance(valid_args, dict) and (valid is None):
            train, valid = train_test_split(train, **valid_args)
        if valid is not None:
            self.valid_loader = torch.utils.data.DataLoader(valid,
                                                            **self.dataloader_args)
        self.train_loader = torch.utils.data.DataLoader(train, **self.dataloader_args)
    
    def test(self, test, truth_in_data=True):
        test_loader = torch.utils.data.DataLoader(test, batch_size = 512,
                                                  pin_memory = False,
                                                  shuffle=False)
        output_lst = {"truth":[], "output_ratio":[], "predictions":[]}
        for batch in tqdm(test_loader, total=len(test_loader)):
            x = batch[:,:self.input_dim].to(self.device)
            x.requires_grad = True
            output = self.forward(x)
            if truth_in_data:
                y = batch[:,self.input_dim:]
                output_lst["truth"].append(y.numpy())
            output_lst["output_ratio"].append(np.ravel(
                output[0].cpu().detach().numpy())
                )
            if self.network_type=="clf":
                output_lst["predictions"].append(
                    np.ravel(torch.sigmoid(output[1]).cpu().detach().numpy())
                    )
            else:
                output_lst["predictions"].append(
                    np.ravel(output[1].cpu().detach().numpy())
                    )
        output_lst = {i: np.concatenate(output_lst[i],0) for i in output_lst}
        return output_lst
            

    def run_training(self, train_loader=None, valid_loader=None,
              n_epochs=50, load_best=True,
              **kwargs):
        if train_loader is None:
            if self.train_loader is None:
                raise ValueError("To use internal train_loader, run create_loaders()")
            if self.valid_loader is not None:
                valid_loader = self.valid_loader
            train_loader = self.train_loader

        if kwargs.get("standard_lr_scheduler", False):
            self.set_lr_scheduler("singlecosine",
                                  {"T_max": len(train_loader)*n_epochs, "eta_min": 1e-7})

        pbar = tqdm(range(n_epochs), disable= not kwargs.get("verbose", True))
        
        for ep in pbar:
            self.loss_values=[]
            for batch in train_loader:
                self.optimizer.zero_grad()
                x = batch[:,:self.input_dim].to(self.device)
                y = batch[:,self.input_dim:].to(self.device)
                x.requires_grad = True
                y.requires_grad = True
                _, logit = self.forward(x)

                loss = self.loss(logit.view(-1), y.view(-1))
                loss.backward()
                if self.kwargs.get("clip_bool", False):
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 10)
                self.optimizer.step()
                self.loss_values.append(loss.cpu().detach().numpy())
            if valid_loader is not None:
                self.evaluate_performance(valid_loader,ep_nr=ep)
            self.loss_data['train_loss'].append(float(np.mean(self.loss_values)))
            self.loss_data['lr'].append(self.optimizer.param_groups[0]['lr'])
            pbar.set_postfix({"Train loss": f"{str(round(self.loss_data['train_loss'][-1],4))}"})
        
        if load_best:
            print(f" Best epoch: {self.nr_best_epoch} ".center(30, "-"))
            self.load_state_dict(self.state_of_best)
        self.network.eval()

    def forward(self, x) -> tuple:
        p_hat = self.network(x)

        r_hat = (1 - p_hat)/p_hat
        
        return r_hat, p_hat

    def evaluate_performance(self,loader, ep_nr=None, log_val_bool=True):
        loss_lst = []
        output_lst = []
        valid_truth=[]
        self.data_from_loader=[]
        for batch in loader:
            x, y = (batch[:,:self.input_dim].to(self.device),
                   batch[:,self.input_dim:].to(self.device))
            x.requires_grad = True
            y.requires_grad = True
            _, logit = self.forward(x)
            loss = self.loss(logit.view(-1), y.view(-1))
            
            loss_lst.append(loss.cpu().detach().numpy())
            output_lst.append(logit.cpu().detach().numpy())
            self.data_from_loader.append(x.cpu().detach().numpy())
            valid_truth.append(y.cpu().detach().numpy())

        self.data_from_loader = np.concatenate(self.data_from_loader,0)
        self.valid_truth = np.concatenate(valid_truth,0)
        self.output_lst = np.concatenate(output_lst,0)
        loss_lst = np.ravel(loss_lst)

        if self.network_type=="clf":
            valid_auc = calculate_auc(self.output_lst, self.valid_truth) # can use simple logit
        if log_val_bool: # for clf
            if self.eval_str == "valid_loss":
                mask_eval = np.mean(loss_lst) < self.loss_data["valid_loss"]
            elif self.eval_str == "valid_auc":
                mask_eval = valid_auc > self.loss_data["valid_auc"]
                self.loss_data["valid_auc"].append(float(valid_auc))
            else:
                raise ValueError("eval_str unknown")

            if all(mask_eval):
                self.state_of_best = copy.deepcopy(self.state_dict())
                self.best_predictions = self.output_lst.copy()
                self.valid_truth_for_best = self.valid_truth.copy()
                self.nr_best_epoch = ep_nr
                

        if log_val_bool:# for reg
            self.loss_data["valid_loss"].append(float(np.mean(loss_lst)))
        return float(valid_auc), float(np.mean(loss_lst))
    
    def plot_log(self, save_path, ep_nr=""):

        os.makedirs(f"{save_path}/plots_{ep_nr}/", exist_ok=True)

        # plot predictions
        fig = plt.figure()
        style={"alpha": 0.5, "bins":60,
                "range":np.percentile(self.output_lst, [1,99])}
        sig_counts, _, _ = plt.hist(self.output_lst[self.valid_truth==1],
                              label="sig", **style)
        bkg_counts, bins,_= plt.hist(self.output_lst[self.valid_truth==0], label="bkg",
                                  **style)
        bins = (bins[1:]+bins[:-1])/2
        outlier_size = np.sum(self.valid_truth==0)*0.01

        try:
            low_cut = np.round(bins[(sig_counts-bkg_counts)<-outlier_size][-1],3)
            high_cut = np.round(bins[(sig_counts-bkg_counts)>outlier_size][0],3)
            plt.vlines(high_cut, 0, np.max(bkg_counts), colors="blue")
            plt.vlines(low_cut, 0, np.max(bkg_counts), colors="blue")
        except IndexError: # if there is no bins
            low_cut = np.percentile(self.output_lst, 10)
            high_cut = np.percentile(self.output_lst, 90)
            plt.vlines(high_cut, 0, np.max(bkg_counts), colors="red")
            plt.vlines(low_cut, 0, np.max(bkg_counts), colors="red")

        plt.legend()
        plt.xlabel("Predictions")

        if save_path is not None:
            utils.save_fig(fig, f"{save_path}/plots_{ep_nr}/predictions_{ep_nr}.png")
            
        mask_low = np.ravel(low_cut>self.output_lst)
        mask_high = np.ravel(high_cut<self.output_lst)
        
        for i in range(self.data_from_loader.shape[1]):
            fig = plt.figure()
            style={"alpha": 0.5, "bins":50,
                    "range":np.percentile(self.data_from_loader[:,i], [0.1,99.9])}
            plt.hist(self.data_from_loader[mask_high][:,i],
                                label=f"Above {high_cut}", **style)
            plt.hist(self.data_from_loader[mask_low][:,i],
                                         label=f"Below {low_cut}", **style)
            plt.hist(self.data_from_loader[:,i],
                                         label=f"Standard dist", histtype="step",
                                         **style)
            plt.legend()
            if save_path is not None:
                utils.save_fig(fig, f"{save_path}/plots_{ep_nr}/outliers_dist_{i}_{ep_nr}.png")
        

        for key,values in self.loss_data.items():
            if "valid" in key:
                continue
            fig = plt.figure()
            plt.plot(values, label="Train")
            plt.plot(self.loss_data[key.replace("train", "valid")], label="Valid")
            plt.legend()

            if all(np.array(values)>0) & all(np.array(self.loss_data[key.replace("train", "valid")])>0):
                plt.yscale("log")
                
            if save_path is not None:
                utils.save_fig(fig, f"{save_path}/plots_{ep_nr}/{key.replace('train', '')}.png")
    
    def plot_auc(self, path, epoch_nr=""):
        fig = plt.figure()
        label = f"Minimum at {np.argmin(self.loss_data['valid_auc'])}, value at {np.min(self.loss_data['valid_auc'])}"
        plt.plot(self.loss_data["valid_auc"], label = label)
        plt.xlabel("Epoch")
        plt.ylabel(r"valid$_{AUC}$")
        if not ".png" in path:
            path += ".png"
        path = path.replace(".png", f"_{epoch_nr}.png")
        plt.savefig(path)
        plt.close(fig)

def calculate_auc(pred, truth):
    "calculate auc in classification should be changed to torch function"
    fpr, tpr, _ = metrics.roc_curve(truth, pred)
    auc = metrics.auc(fpr, tpr)
    return auc

