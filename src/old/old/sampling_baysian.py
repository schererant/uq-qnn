"""Forward pass through layer.

Args:
    x: input.

Returns:
    outputs of variational layer
"""
if self.batched_samples:
    n_samples = x.shape[0]
    assert n_samples <= self.max_n_samples, (
        "Number of samples needs to be <= max_n_samples"
        f" but found max_n_samples {self.max_n_samples} "
        f" and {n_samples} in input tensor."
    )
    assert x.dim() == 3, (
        "Expect input to be a tensor of shape "
        "[num_samples, batch_size, num_features], "
        f"but found shape {x.shape}"
    )
else:
    n_samples = 1

delta_weight, delta_bias = self.sample_weights(n_samples)

# forward pass with chosen layer type
if self.layer_type == "reparameterization":
    # sample weight via reparameterization trick
    output = x.matmul((self.mu_weight + delta_weight).transpose(-1, -2))
    if self.bias:
        output = output + (self.mu_bias + delta_bias).unsqueeze(1)
else:
    # linear outputs
    out = F.linear(x, self.mu_weight, self.mu_bias)
    # flipout
    if self.is_frozen:
        torch.manual_seed(0)
    sign_input = x.clone().uniform_(-1, 1).sign()
    sign_output = out.clone().uniform_(-1, 1).sign()
    # get outputs+perturbed outputs
    output = out + (
        (
            (x * sign_input).matmul(delta_weight.transpose(-1, -2))
            + delta_bias.unsqueeze(1)
        )
        * sign_output
    )
if not self.batched_samples:
    output = output.squeeze(0)
return output

def sample_weights(self, n_samples: int) -> tuple[Tensor]:
    """ Sample variational weights for batched sampling.

    Args:
        n_samples: number of samples to draw

    Returns:
        delta_weight and delta_bias
    """ 
    if self.is_frozen:
        eps_weight = self.eps_weight
        bias_eps = self.eps_bias
    else:
        eps_weight = self.eps_weight.data.normal_()
        if self.mu_bias is not None:
            bias_eps = self.eps_bias.data.normal_()

    # select from max_samples
    eps_weight = eps_weight[:n_samples]

    # select first sample if not batched to keep consistent shape

    # sample weight with reparameterization trick
    sigma_weight = F.softplus(self.rho_weight)
    delta_weight = eps_weight * sigma_weight

    delta_bias = torch.zeros(1).to(self.rho_weight.device)

    if self.mu_bias is not None:
        bias_eps = bias_eps[:n_samples]

        # sample bias with reparameterization trick
        sigma_bias = F.softplus(self.rho_bias)
        delta_bias = bias_eps * sigma_bias
    return delta_weight, delta_bias