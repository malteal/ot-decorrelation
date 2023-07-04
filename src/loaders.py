

import numpy as np
import torch as T

from nflows.distributions.base import Distribution

DATALOADER_args = {"pin_memory": True,
                   "num_workers": 4,
                   "prefetch_factor": 12,
                   "persistent_workers":True
                   }
class Iterator:
    def __init__(self) -> None:
        self.nr_nonconvex_dimensions=None
        self.nr_convex_dimensions=None
        self.dataset=None
        self.device=None

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        if self._iterator is None:
            self._iterator = iter(self.dataset)
        else:
            self._reset()
        return self._iterator

    def _reset(self) -> None:
        self._iterator = iter(self.dataset)

    def __next__(self):
        try:
            batch = next(self._iterator)
        except (StopIteration, AttributeError):
            self._reset()
            batch = next(self._iterator)
        batch = batch.float()

        if batch.shape[1] == self.nr_nonconvex_dimensions+self.nr_convex_dimensions:
            mask = T.ones((len(batch), 1))
        else:
            mask = batch[:,-1:].bool()
        
        batch.requires_grad=True
        batch = batch.to(self.device)
        batch = [
            batch[:, : self.nr_nonconvex_dimensions],
            batch[
                :,
                self.nr_nonconvex_dimensions : self.nr_nonconvex_dimensions
                + self.nr_convex_dimensions,
            ],
            mask.bool().to(self.device),
        ]
        return batch

class Dataset(Iterator):
    "class that initialize the tfrecords pipeline for torch"

    def __init__(
        self,
        data,
        batch_size=32,
        device="cpu",
        nr_convex_dimensions=None,
        nr_nonconvex_dimensions=None,
        **kwargs
    ):
        super().__init__()
        self.batch_size = batch_size
        self.data=data
        if not isinstance(self.data, T.Tensor):
            self.data = T.tensor(self.data).float()

        if kwargs.get("shuffle", False):        
            rand_perm = T.randperm(len(self.data))
            self.data = data[rand_perm]

        self.scaler = kwargs.get("scaler", None)

        if self.scaler is not None:
            self.data = T.tensor(
                self.scaler.transform(self.data)
                ).float()
        
        self.dataset = T.utils.data.DataLoader(self.data.cpu().detach(),
                                               batch_size=self.batch_size,
                                               shuffle=False,drop_last=True,
                                               **DATALOADER_args
                                               )

        self.nr_convex_dimensions = nr_convex_dimensions
        self.nr_nonconvex_dimensions = nr_nonconvex_dimensions
        self.device = device

class BaseDistribution(Iterator):
    # add __next__ to this or utils.Dataset as init
    def __init__(self,distribution, dims:int, batch_size:int=1024, device:str="cuda") -> None:
        if ((not hasattr(distribution, 'log_prob')) 
            and (not hasattr(distribution, 'sample'))):
            raise TypeError("distribution requires log_prob & sample as attributs")
        super().__init__()
        self.distribution = distribution
        self.device = device
        self.dims = dims
        self.sample_loader = None
        self.batch_size = batch_size
        self.mu = 0
        self.sigma = 1
    
    def sample(self, conds_dist, transport_dist=None) -> None:
        self.conds_dist=conds_dist
        self.transport_dist=transport_dist
        if self.transport_dist is None:
            if (isinstance(self.distribution, Dirichlet)
                or isinstance(self.distribution,T.distributions.MultivariateNormal)
                ):
                self.dist = self.distribution.sample((len(conds_dist)))
            elif (isinstance(self.distribution,T.distributions.relaxed_bernoulli.LogitRelaxedBernoulli)
                or isinstance(self.distribution,T.distributions.Normal)):
                self.dist = self.distribution.sample([len(conds_dist)])
            else:
                self.dist = self.distribution.sample((len(conds_dist), self.dims))
        else:
            idx = T.randperm(len(self.transport_dist))
            self.dist = self.transport_dist[idx].detach()

        self.data = T.tensor(np.c_[conds_dist.detach().cpu(), self.dist.cpu()], requires_grad=True).float()
        self.nr_nonconvex_dimensions = conds_dist.shape[1]
        self.nr_convex_dimensions = self.data.shape[1]-self.nr_nonconvex_dimensions
        self.dataset = T.utils.data.DataLoader(self.data.detach(),
                                               batch_size=self.batch_size,
                                               drop_last=True,
                                               pin_memory= True,
                                               num_workers= 4,
                                               )
    def log_prob(self,values):
        return self.distribution.log_prob(values)
        
    def _reset(self) -> None:
        self.sample(self.conds_dist, self.transport_dist)
        self._iterator = iter(self.dataset)

def get_base_distribution(cvx_dim:int, distribution_name, logit:bool=True, device:str="cuda"):
    if cvx_dim==1:
        base_distribution = T.distributions.uniform.Uniform(
            T.tensor(0.0, device = device),
            T.tensor(1.0, device = device)
            )
        if logit:
            transforms = [T.distributions.transforms.SigmoidTransform().inv]
            base_distribution = T.distributions.transformed_distribution.TransformedDistribution(base_distribution, transforms)
    elif (cvx_dim==3) and ("norm" in distribution_name):
        base_distribution = T.distributions.normal.Normal(
            T.tensor([0.0]*cvx_dim, device = device),
            T.tensor([1.0]*cvx_dim, device = device),
            )
    elif cvx_dim==3:
        base_distribution = Dirichlet(
            T.tensor([1.0]*cvx_dim), device = device,
            logit=logit,
            )
    else:
        base_distribution = T.distributions.normal.Normal(
            T.tensor([0.0]*cvx_dim, device = device),
            T.tensor([1.0]*cvx_dim, device = device),
            )
        

class Dirichlet(Distribution):
    def __init__(self, alpha, logit=False, drop_dim=0, device="cuda"):
        super().__init__()
        self.alpha = alpha
        self.logit=logit
        self.drop_dim=drop_dim
        self.base_dist = T.distributions.dirichlet.Dirichlet(alpha.to(device))
        self._shape = alpha.shape
        self.batch_shape=512

    def _log_prob(self, inputs, context):
        # Note: the context is ignored.
        if self.drop_dim >0:
            inputs = T.concat([inputs,1-inputs.sum(1).view(-1,1)],1)
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(
                    self._shape, inputs.shape[1:]
                )
            )

        if self.logit:
            inputs = self.probsfromlogits(inputs)
        self._log_prob_value = self.base_dist.log_prob(inputs)
        return self._log_prob_value.view(len(inputs),-1).to(inputs.device)

    @staticmethod
    def probsfromlogits(logitps: np.ndarray) -> np.ndarray:
        """reverse transformation from logits to probs

        Parameters
        ----------
        logitps : np.ndarray
            arrray of logit

        Returns
        -------
        np.ndarray
            probabilities from logit
        """
        norm=1
        ps_value = 1.0 / (1.0 + T.exp(-logitps))
        # ps_value = T.exp(ps_value)
        # ps_value = ps_value/ps_value.sum(1).view(-1,1)
        if (ps_value.shape[-1]>1) and (len(ps_value.shape)>1):
            norm = T.sum(ps_value, axis=1)
            norm = T.stack([norm] * logitps.shape[1]).T
        return ps_value / norm

    def _sample(self, num_samples, context):
        context_size = 1 if context is None else context.shape[0]
        samples = self.base_dist.sample([context_size*num_samples])
        
        if self.logit:
            samples = T.log(samples/(1-samples))

        #make the distribution unbound
        samples = samples[:,self.drop_dim:]

        if context is None:
            return samples
        else:
            return torchutils.split_leading_dim(samples, [context_size, num_samples])
