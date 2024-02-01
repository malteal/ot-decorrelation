"""
Copyright (c) 2021 Chris Pollard and Philipp Windischhofer
an implementation of the partially convex neural networks
introduced in https://arxiv.org/abs/1609.07152
"""
import torch
import numpy as np

def Softplus(threshold=20, zeroed=False, symmetric=False, device="cuda"):
    # can be exported by onnx_gradient
    zero = torch.tensor([0.0], device=device)
    def softplus(x):
        # return torch.log(1+torch.exp(x))
        y = x.clamp(max=threshold)
        return torch.where(x<threshold, torch.log(1+torch.exp(y)),x)
    
    if zeroed:
        def zeroed_softplus(x):
            return softplus(x)-softplus(zero)
        return zeroed_softplus

    elif symmetric:
        def symmetric_softplus(x):
            return softplus(x)-0.5*x
        return symmetric_softplus

    else:
        return softplus

def get_act_func(activation_str: str, device="cuda") -> callable:
    """output different activation function

    Parameters
    ----------
    activation_str : str
        name of activation function

    Returns
    -------
    callable
        return the callable activation function

    Raises
    ------
    ValueError
        if the activation function is unknown
    """
    activation_str = activation_str.lower().replace("_", "")
    if activation_str == "softplus":
        act_func = Softplus(device=device)
    elif activation_str == "softpluszeroed":
        act_func = Softplus(zeroed=True, device=device)
    elif activation_str == "relu":
        act_func = torch.nn.ReLU() 
    elif activation_str == "celu":
        act_func =  torch.nn.CELU() 
    elif activation_str == "tanh":
        act_func =  torch.nn.Tanh() 
    elif activation_str == "leakyrelu":
        act_func = torch.nn.LeakyReLU(0.2) 
    elif activation_str == "symsoftplus":
        act_func = Softplus(symmetric=True)
    elif activation_str == "elu":
        act_func = torch.nn.ELU() 
    elif activation_str == "":
        act_func = lambda x: x # dummy functions
    else:
        raise ValueError(f"Did not recognize the activation_str: {activation_str}")
    return act_func


def apply_linear_layer(
    input_value: torch.Tensor, weights: torch.Tensor,
    act_weight_zz:callable = None) -> torch.Tensor: # torch.nn.functional.softplus
    """apply linear mapping

    Parameters
    ----------
    input_value : torch.Tensor
         x input for the network
    weights : torch.Tensor
        weight input of the neurons
    biases : int, optional
        weight input of the neurons, by default 0

    Returns
    -------
    torch.Tensor
         torch.Tensor: Output the linear mapping
    """
    # try:
    if act_weight_zz is not None:
        gain = 1 / input_value.size(1)
        return input_value.matmul(act_weight_zz(weights.t()))*gain
    else:
        return input_value.matmul(weights.t())


def weight(x_value: int, y_value: int, device: str = "cpu") -> torch.Tensor:
    """Initialize the weight parameter of neurons randomly

    Parameters
    ----------
    x_value : int
        Number of colums
    y_value : int
        Number of rows
    device : str, optional
        Which device to init on, by default "cpu"

    Returns
    -------
    torch.Tensor
        output a random weight tensor with size y times x_value.
    """
    weights = torch.empty((y_value, x_value)) 
    if x_value > 0 and y_value > 0:
        torch.nn.init.kaiming_normal_(weights)

    weights = weights/weights.size(1) # divide with input size
    parameters = torch.nn.parameter.Parameter(weights.to(device)
    )
    return parameters


def bias(x_value: int, device: str = "cpu", requires_grad=True) -> torch.Tensor:
    """Initialize the bias parameter of neurons randomly

    Parameters
    ----------
    x_value : int
        Number of biases
    device : str, optional
        Which device to init, by default "cpu"

    Returns
    -------
    torch.Tensor
        Output the bias vector
    """
    bias = torch.zeros(x_value)

    if (x_value > 0) & requires_grad:
        torch.nn.init.uniform_(
            bias, -np.sqrt(1.0 / x_value), np.sqrt(1.0 / x_value)
        )

    parameters = torch.nn.parameter.Parameter(
        bias.to(device), requires_grad=requires_grad 
    )
    return parameters
