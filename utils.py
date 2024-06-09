import torch
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def lanczos(matrix_vector, dim: int, neigs: int):
    def mv(vec: np.ndarray):
        gpu_vec = torch.tensor(vec, dtype=torch.float)#.cuda()
        return matrix_vector(gpu_vec)
    operator = LinearOperator((dim, dim), matvec=mv)
    evals, evecs = eigsh(operator, neigs)
    return torch.from_numpy(np.ascontiguousarray(evals[::-1]).copy()).float(), \
           torch.from_numpy(np.ascontiguousarray(np.flip(evecs, -1)).copy()).float()


def compute_hvp(model, loss, vector):
    # Determine the device used by the model
    device = next(model.parameters()).device

    # Move input tensors to the same device as the model
    vector = vector.to(device)

    model.zero_grad()

    # Compute gradients of the loss w.r.t. model parameters
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    grads_vector = parameters_to_vector(grads)

    # Dot product of gradients and the input vector
    dot_product = torch.dot(grads_vector, vector)

    # Compute Hessian-vector product
    hvp = torch.autograd.grad(dot_product, model.parameters(), retain_graph=True)

    return parameters_to_vector(hvp)


def get_hessian_eigenvalues(model, loss_fn, neigs=5):
    """ Compute the leading Hessian eigenvalues. """
    hvp_delta = lambda delta: compute_hvp(model, loss_fn, delta).detach().cpu()
    nparams = len(parameters_to_vector((model.parameters())))
    evals, evecs = lanczos(hvp_delta, nparams, neigs=neigs)
    return evals


def get_hessian_trace(model, loss_fn, n_iters=100):
    hvp_delta = lambda delta: compute_hvp(model, loss_fn, delta).detach().cpu()
    nparams = len(parameters_to_vector((model.parameters())))
    trace=0.0
    for _ in range(n_iters):
        v = torch.randn(nparams)
        Hv = hvp_delta(v)
        trace += torch.dot(Hv, v).item()
    return trace/n_iters 