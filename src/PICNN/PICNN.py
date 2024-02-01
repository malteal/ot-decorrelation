"Create the PICNN model"  # pylint: disable=invalid-name
import numpy as np
from omegaconf import OmegaConf

import torch

from src.PICNN.layers import (
    weight,
    bias,
    apply_linear_layer,
    get_act_func,
)


class PICNN(
    torch.nn.Module
):  # pylint: disable=invalid-name,too-many-instance-attributes
    "Create the PICNN model"

    def __init__(  # pylint:disable=too-many-arguments
        self,
        convex_activation: str,
        convex_layersizes: list,
        nonconvex_layersizes: list=None,
        nonconvex_activation: str="",
        activation_params: dict = {},
        act_enforce_cvx: str = "softplus",
        act_weight_zz: str = "softplus",
        correction_trainable:bool = False,
        cvx_norm: str = "",
        noncvx_norm: str = "",
        first_act_sym:str = "no",
        device: str = "cpu",
        verbose:bool=True,
        logit:bool=True,
        **kwargs
    ) -> None:
        # torch.set_default_dtype(torch.float64)
        """Initializing the PICNN model

        Parameters
        ----------
        nonconvex_activation : str
            Name of the desired activation function
        convex_activation : str
            Name of the desired activation function
        nonconvex_layersizes : list
            list of number of layers and size
        convex_layersizes : list
            list of number of layers and size
        norm : str, optional
            to activate batch norm or not, by default False
        device : str, optional
            which device to run on, by default "cpu"
        """

        super().__init__()
        if isinstance(nonconvex_layersizes, int) and isinstance(convex_layersizes, int):
            nonconvex_layersizes = [kwargs["noncvx_dim"]]+[nonconvex_layersizes]*kwargs["n_layers"]
            convex_layersizes = [kwargs["cvx_dim"]]+[convex_layersizes]*kwargs["n_layers"]+[1]
        elif ((isinstance(nonconvex_layersizes, list) and isinstance(convex_layersizes, list))
              or (OmegaConf.is_list(nonconvex_layersizes) and OmegaConf.is_list(convex_layersizes))):
            assert len(nonconvex_layersizes) + 1 == len(convex_layersizes)
            assert act_weight_zz is not None, "Positive project should be a always positive function"


        # should be removed 
        self.PICNN_incorrect_btilde_ = kwargs.get("PICNN_incorrect_btilde_", False)
        self.nhidden = len(convex_layersizes) - 1
        self.device = device
        self.noncvx_norm = noncvx_norm
        self.cvx_norm = cvx_norm
        self.correction_trainable = correction_trainable
        self.convex_activation = convex_activation
        self.nonconvex_activation = nonconvex_activation
        self.scaler = kwargs.get("scaler", None)
        self.logit=logit
        #activation function force convexity on the linear transformations
        self.act_enforce_cvx = get_act_func(act_enforce_cvx, activation_params, device=self.device)
        self.act_weight_zz = get_act_func(act_weight_zz, activation_params, device=self.device)

        # activation functions in layers
        nonconvex_activation_func = get_act_func(nonconvex_activation,
                                                            activation_params, device=self.device)
        self.gtilde_act = [nonconvex_activation_func for _ in range(self.nhidden-1)]
        self.gtilde_act.append(get_act_func("softplus", {}, device=self.device))
        # init activation function
        convex_activation_func = get_act_func(convex_activation, activation_params, device=self.device)

        self.g_act = [convex_activation_func for _ in range(self.nhidden)]

        if isinstance(first_act_sym, str) & (first_act_sym != "no"):
            self.g_act[0] = get_act_func(first_act_sym, device=self.device)
            self.first_act_sym=True
        else:
            self.first_act_sym=False

        self.nonconvex_layersizes = ([0]*self.nhidden if nonconvex_layersizes is None
                                     else nonconvex_layersizes)
        self.convex_layersizes = convex_layersizes
        
        #create trainable parameters
        self.weight_zz = torch.nn.ParameterList()
        self.weight_y = torch.nn.ParameterList()
        self.bias = torch.nn.ParameterList()

        self.bias_z = torch.nn.ParameterList()
        self.weight_zu = torch.nn.ParameterList()
        self.bias_y = torch.nn.ParameterList()
        
        self.weight_yu = torch.nn.ParameterList()
        self.weight_u = torch.nn.ParameterList()

        self.weight_uutilde = torch.nn.ParameterList()
        self.bias_tilde = torch.nn.ParameterList()

        self.initialize_parameters()
        self.to(self.device)

        if verbose:
            n_trainable = self.count_trainable_parameters()
            print(f"Number of trainable parameters: {int(n_trainable)}")

    def count_trainable_parameters(self):
        sum_trainable = np.sum([i.numel() for i in self.parameters() if i.requires_grad])
        return sum_trainable

    def initialize_parameters(self):
        """initialize all parameters"""
        # more-or-less following the nomenclature from
        # arXiv:1609.07152

        # shorthand:
        zsize = self.convex_layersizes
        usize = self.nonconvex_layersizes
        ysize = zsize[0]
        # =============================================================================
        # Build network
        # =============================================================================

        # convex layers
        if zsize[-1] != 1:
            raise ValueError("Last layer of the convex part has to be 1 - change convex_layersizes to 1 as last")
        
        for lay in range(self.nhidden):
            self.weight_zz.append(weight(zsize[lay], zsize[lay + 1], device = self.device))
            self.weight_y.append(weight(ysize, zsize[lay + 1], device = self.device))
            self.bias.append(bias(zsize[lay + 1], device = self.device))
            #PICNN
            self.weight_yu.append(weight(usize[lay], ysize, device = self.device,
                                         requires_grad=usize[lay]!=0))
            self.weight_u.append(weight(usize[lay], zsize[lay + 1],
                                        device = self.device,
                                        requires_grad=usize[lay]!=0))
            self.bias_y.append(bias(ysize,
                                    device = self.device,
                                    requires_grad=usize[lay]!=0))
            self.weight_zu.append(weight(usize[lay], zsize[lay],
                                         device = self.device,
                                         requires_grad=usize[lay]!=0))
            self.bias_z.append(bias(zsize[lay],
                                    device = self.device,
                                    requires_grad=usize[lay]!=0))

        # conditional network PICNN
        for lay in range(self.nhidden-1):
            self.weight_uutilde.append(weight(usize[lay], usize[lay+1],
                                              device = self.device,
                                              requires_grad=usize[lay]!=0))
            if self.PICNN_incorrect_btilde_:
                self.bias_tilde.append(bias(usize[lay],
                                            device = self.device,
                                            requires_grad=usize[lay+1]!=0))
            else:
                self.bias_tilde.append(bias(usize[lay+1],
                                            device = self.device,
                                            requires_grad=usize[lay+1]!=0))


        # the first z_i input is zero and it shouldnt be used in training
        self.weight_zu[0].requires_grad=False
        self.bias_z[0].requires_grad=False
        self.weight_zz[0].data.copy_(
            torch.zeros_like(self.weight_zz[0])  # pylint: disable=E1101
        )
        self.weight_zz[0].requires_grad = False

        correction_weights = weight(usize[-1], 2,
                                    requires_grad=self.correction_trainable)
        correction_biases = bias(2, requires_grad=self.correction_trainable)


        self.weight_uutilde.append(correction_weights)
        self.bias_tilde.append(correction_biases)


    def forward(self, xs_input: torch.Tensor, ys_input_or: torch.Tensor) -> torch.Tensor:
        # self.eval()
        """Run the forward data flow of the model

        Parameters
        ----------
        xs_input : torch.Tensor
            conditional distribution
        ys_input : torch.Tensor
            source distribution

        Returns
        -------
        torch.Tensor
            return the output of the network h(theta,x)
        """
        ui_value = xs_input

        ys_input = ys_input_or.clone()
        zi_value = torch.zeros((len(xs_input),1), device=self.device)

        for i in range(self.nhidden):
            yterm = ys_input * (apply_linear_layer(
                ui_value, self.weight_yu[i]
            )+self.bias_y[i]
            )

            yterm = apply_linear_layer(yterm, self.weight_y[i])

            zterm = (zi_value
                * self.act_enforce_cvx(apply_linear_layer(ui_value, self.weight_zu[i])
                +self.bias_z[i])
                    ) 

            zi_value = self.g_act[i](
                    apply_linear_layer(zterm, self.weight_zz[i],
                                    act_weight_zz = self.act_weight_zz)
                    + yterm
                    + apply_linear_layer(ui_value, self.weight_u[i])
                    + self.bias[i]
            )
            # last iteration will be for correction to output

            ui_value = self.gtilde_act[i](
                    apply_linear_layer(
                        ui_value, self.weight_uutilde[i]
                    )+self.bias_tilde[i]
                )

        if self.correction_trainable:
            zi_value = (ui_value[:,:1]/2 * torch.sum(ys_input_or * ys_input_or, axis=1, keepdim=True)+ ui_value[:,1:] * zi_value)
        else:
            zi_value = (1/2 * torch.sum(ys_input_or * ys_input_or, axis=1, keepdim=True) + zi_value)

        return zi_value

    def transport(self, conditionals, totransport, sig_mask=None,
                  create_graph=True):
        "Transport points with sig mask"
        if sig_mask is None:
            sig_mask = torch.ones_like(totransport[:,0])==1
            sig_mask = sig_mask.to(self.device)

        cvx_output = self.forward(conditionals[sig_mask.flatten()],
                                 totransport[sig_mask.flatten()])

        output = torch.autograd.grad(
            outputs=cvx_output,
            inputs = totransport,
            retain_graph=create_graph,
            create_graph=create_graph,
            grad_outputs=torch.ones_like(cvx_output)
        )[0]

        # add in bkg points
        output = output+(~sig_mask.reshape(-1,1)*totransport)

        return output, cvx_output
    
    def chunk_transport(self, conditionals, totransport, sig_mask=None, n_chunks=20,
                        scaler=None):
        "transport large datasamples in chunks - only for eval"
        if sig_mask is None:
            sig_mask = torch.ones_like(totransport[:,0])==1

        transport = []
        for i,j,mask in zip(
                        conditionals.chunk(n_chunks),
                        totransport.chunk(n_chunks),
                        sig_mask.chunk(n_chunks),
                        ):
            transport.append(self.transport(i.to(self.device),
                                        j.to(self.device),
                                        mask.to(self.device),
                                        create_graph=False
                                        )[0].cpu().detach())
        transport = torch.concat(transport,0)
        return transport
