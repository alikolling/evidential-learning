import torch
import torch.nn.functional as F

tol = torch.finfo(torch.float32).eps

# Normal Inverse Gamma Negative Log-Likelihood
# from https://arxiv.org/abs/1910.02600:
# > we denote the loss, L^NLL_i as the negative logarithm of model
# > evidence ..


def modified_mse(gamma, nu, alpha, beta, target, reduction="mean"):
    """
    Lipschitz MSE loss of the "Improving evidential deep learning via multi-task learning."

    Args:
        gamma ([FloatTensor]): the output of the ENet.
        nu ([FloatTensor]): the output of the ENet.
        alpha ([FloatTensor]): the output of the ENet.
        beta ([FloatTensor]): the output of the ENet.
        target ([FloatTensor]): true labels.
        reduction (str, optional): . Defaults to 'mean'.

    Returns:
        [FloatTensor]: The loss value.
    """
    mse = (gamma - target) ** 2
    c = get_mse_coef(gamma, nu, alpha, beta, target).detach()
    modified_mse = mse * c
    if reduction == "mean":
        return modified_mse.mean()
    elif reduction == "sum":
        return modified_mse.sum()
    else:
        return modified_mse


def get_mse_coef(gamma, nu, alpha, beta, y):
    """
    Return the coefficient of the MSE loss for each prediction.
    By assigning the coefficient to each MSE value, it clips the gradient of the MSE
    based on the threshold values U_nu, U_alpha, which are calculated by check_mse_efficiency_* functions.

    Args:
        gamma ([FloatTensor]): the output of the ENet.
        nu ([FloatTensor]): the output of the ENet.
        alpha ([FloatTensor]): the output of the ENet.
        beta ([FloatTensor]): the output of the ENet.
        y ([FloatTensor]): true labels.

    Returns:
        [FloatTensor]: [0.0-1.0], the coefficient of the MSE for each prediction.
    """
    alpha_eff = check_mse_efficiency_alpha(gamma, nu, alpha, beta, y)
    nu_eff = check_mse_efficiency_nu(gamma, nu, alpha, beta, y)
    delta = (gamma - y).abs()
    min_bound = torch.min(nu_eff, alpha_eff).min()
    c = (min_bound.sqrt() / delta).detach()
    return torch.clip(c, min=False, max=1.0)


def check_mse_efficiency_alpha(gamma, nu, alpha, beta, y, reduction="mean"):
    """
    Check the MSE loss (gamma - y)^2 can make negative gradients for alpha, which is
    a pseudo observation of the normal-inverse-gamma. We can use this to check the MSE
    loss can success(increase the pseudo observation, alpha).

    Args:
        gamma, nu, alpha, beta(torch.Tensor) output values of the evidential network
        y(torch.Tensor) the ground truth

    Return:
        partial f / partial alpha(numpy.array)
        where f => the NLL loss (BayesianDTI.loss.MarginalLikelihood)

    """
    delta = (y - gamma) ** 2
    right = (
        (torch.exp((torch.digamma(alpha + 0.5) - torch.digamma(alpha))) - 1)
        * 2
        * beta
        * (1 + nu)
        / nu
    )

    return (right).detach()


def check_mse_efficiency_nu(gamma, nu, alpha, beta, y):
    """
    Check the MSE loss (gamma - y)^2 can make negative gradients for nu, which is
    a pseudo observation of the normal-inverse-gamma. We can use this to check the MSE
    loss can success(increase the pseudo observation, nu).

    Args:
        gamma, nu, alpha, beta(torch.Tensor) output values of the evidential network
        y(torch.Tensor) the ground truth

    Return:
        partial f / partial nu(torch.Tensor)
        where f => the NLL loss (BayesianDTI.loss.MarginalLikelihood)
    """
    gamma, nu, alpha, beta = gamma.detach(), nu.detach(), alpha.detach(), beta.detach()
    nu_1 = (nu + 1) / nu
    return beta * nu_1 / alpha


def nig_nll(gamma, nu, alpha, beta, target):
    """v = torch.clamp(v, min=tol)
    alpha = torch.clamp(alpha, min=tol)
    beta = torch.clamp(beta, min=tol)"""
    pi = torch.pi
    x1 = torch.log(pi / (nu + tol)) * 0.5
    x2 = -alpha * torch.log(2.0 * beta * (1.0 + nu) + tol)
    x3 = (alpha + 0.5) * torch.log(
        nu * (target - gamma) ** 2 + 2.0 * beta * (1.0 + nu) + tol
    )
    x4 = torch.lgamma(alpha + tol) - torch.lgamma(alpha + 0.5 + tol)
    return x1 + x2 + x3 + x4


# Normal Inverse Gamma regularization
# from https://arxiv.org/abs/1910.02600:
# > we formulate a novel evidence regularizer, L^R_i
# > scaled on the error of the i-th prediction


def nig_reg(gamma, v, alpha, _beta, y):
    error = F.l1_loss(y, gamma, reduction="none")
    evi = 2 * v + alpha
    return error * evi


# KL divergence of predicted parameters from uniform Dirichlet distribution
# from https://arxiv.org/abs/1806.01768
# code based on:
# https://bariskurt.com/kullback-leibler-divergence-between-two-dirichlet-and-beta-distributions/
def dirichlet_reg(alpha, y):
    # dirichlet parameters after removal of non-misleading evidence (from the label)
    alpha = y + (1 - y) * alpha

    # uniform dirichlet distribution
    beta = torch.ones_like(alpha)

    sum_alpha = alpha.sum(-1)
    sum_beta = beta.sum(-1)

    t1 = sum_alpha.lgamma() - sum_beta.lgamma()
    t2 = (alpha.lgamma() - beta.lgamma()).sum(-1)
    t3 = alpha - beta
    t4 = alpha.digamma() - sum_alpha.digamma().unsqueeze(-1)

    kl = t1 - t2 + (t3 * t4).sum(-1)
    return kl.mean()


# Eq. (5) from https://arxiv.org/abs/1806.01768:
# Sum of squares loss
def dirichlet_mse(alpha, y):
    sum_alpha = alpha.sum(-1, keepdims=True)
    p = alpha / sum_alpha
    t1 = (y - p).pow(2).sum(-1)
    t2 = ((p * (1 - p)) / (sum_alpha + 1)).sum(-1)
    mse = t1 + t2
    return mse.mean()


def evidential_classification(alpha, y, lamb=1.0):
    num_classes = alpha.shape[-1]
    y = F.one_hot(y, num_classes)
    return dirichlet_mse(alpha, y) + lamb * dirichlet_reg(alpha, y)


def evidential_regression(dist_params, y, lamb=1.0):
    nll = nig_nll(*dist_params, y).mean()
    reg = nig_reg(*dist_params, y).mean()
    lipschitz_mse = modified_mse(*dist_params, y)
    print(nll, lamb * reg, lipschitz_mse)
    return nll + lipschitz_mse + lamb * reg
