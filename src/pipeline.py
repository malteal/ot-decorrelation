# data at https://zenodo.org/record/3606767/#.Y87pt3bMJPa
from glob import glob
import numpy as np
import pandas as pd

import src.utils as utils 

def load_data(cvx_dim, multi_clf="/srv/beegfs/scratch/groups/rodem/anomalous_jets/taggers/supervised/transformer_multiclass/outputs/combined_scores.h5", 
              upper_mass_cut:int=450, training:bool=True):
    # define data
    output, _ = load_multi_cls(multi_clf, upper_mass_cut=upper_mass_cut)
    if cvx_dim==1: # binary decorrelation
            # from logit to probability
            output["encodings"] = np.reshape(2*(1/(1+np.exp(-utils.get_disc_func(2)(output["encodings"]))))-1, (-1,1))

            output = {key: item for key, item in output.items()}
    
    bkg_mask = output["labels"]==0
    output = {i: output[i][np.ravel(bkg_mask)] for i in output}

    # preprocess
    output["mass"] = np.log(output["mass"])

    return output

        
def load_multi_cls(path, key=None, upper_mass_cut=450):
    data = utils.load_hdf(path)
    key = list(data.keys())[0] if key is None else key
    if len(list(data.keys()))>1:
        raise KeyError("multiple keys in data")
    data = data[key][:]
    data = data[(data[:,0]>=20) & (data[:,0]<upper_mass_cut)]
    data[:,2:] = utils.softmax(data[:,2:])
    columns = ["mass", "label", "q_score", "t_score", "w_score"]
    data = pd.DataFrame(data, columns=columns)
    output = {"mass": data["mass"].to_numpy(), "labels": data["label"].to_numpy(),
              "encodings": data[["q_score", "t_score", "w_score"]].to_numpy()}
    return output, data
