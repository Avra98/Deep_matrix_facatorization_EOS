import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
import argparse
import os
import torch.nn.functional as F
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.autograd import grad, Variable
from torch.autograd.functional import hessian
from torch.nn.utils import parameters_to_vector


def compute_full_hessian(model, criterion, target):
    # Determine the device used by the model
    device = next(model.parameters()).device

    # Move input tensors to the same device as the model
    target = target.to(device)

    model.zero_grad()

    # Forward pass
    output = model()

    # Compute the loss
    loss = criterion(output, target)

    # Compute gradients of the loss w.r.t. model parameters
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    grads_vector = []

    # Vectorize gradients layer by layer
    for layer_grads in grads:
        grads_vector.append((layer_grads).view(-1))
    grads_vector = torch.cat(grads_vector)

    hessian_matrix = []

    # Compute Hessian matrix
    for grad_elem in grads_vector:
        row_grads = grad(grad_elem, model.parameters(), create_graph=True)
        row_grads_vector = []
        
        # Vectorize Hessian row layer by layer
        for layer_row_grads in row_grads:
            row_grads_vector.append(layer_row_grads.view(-1))
        row_grads_vector = torch.cat(row_grads_vector)
        
        hessian_matrix.append(row_grads_vector.detach().cpu().numpy())

    hessian_matrix = np.stack(hessian_matrix)

    # Perform column-wise stacking
    hessian_matrix = np.transpose(hessian_matrix)

    return hessian_matrix

def extract_blocks_and_svd(hessian_matrix, d):
    L = hessian_matrix.shape[0] // (d * d)
    blocks = []

    for i in range(L):
        for j in range(L):
            block = hessian_matrix[i*d**2:(i+1)*d**2, j*d**2:(j+1)*d**2]
            blocks.append(block)
    
    svd_results = []
    for block in blocks:
        U, S, V = np.linalg.svd(block)
        svd_results.append((U, S, V))
    
    return svd_results


def lanczos(matrix_vector, dim: int, neigs: int):
    def mv(vec: np.ndarray):
        gpu_vec = torch.tensor(vec, dtype=torch.float).cuda()
        return matrix_vector(gpu_vec)
    operator = LinearOperator((dim, dim), matvec=mv)
    evals, evecs = eigsh(operator, neigs)
    return torch.from_numpy(np.ascontiguousarray(evals[::-1]).copy()).float(), \
           torch.from_numpy(np.ascontiguousarray(np.flip(evecs, -1)).copy()).float()


def compute_hvp(model, criterion, target, vector):
    # Determine the device used by the model
    device = next(model.parameters()).device

    # Move input tensors to the same device as the model
    target = target.to(device)
    vector = vector.to(device)

    model.zero_grad()

    # Forward pass
    output = model()

    # Compute the loss
    loss = criterion(model(), target)

    # Compute gradients of the loss w.r.t. model parameters
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    grads_vector = parameters_to_vector(grads)

    # Dot product of gradients and the input vector
    dot_product = torch.dot(grads_vector, vector)

    # Compute Hessian-vector product
    hvp = torch.autograd.grad(dot_product, model.parameters(), retain_graph=True)

    return parameters_to_vector(hvp)

def get_hessian_eigenvalues(model, criterion, target,neigs=2):
    """ Compute the leading Hessian eigenvalues. """
    hvp_delta = lambda delta: compute_hvp(model, criterion, target,delta).detach().cpu()
    nparams = len(parameters_to_vector((model.parameters())))
    evals, evecs = lanczos(hvp_delta, nparams, neigs=neigs)
    return evals

class OrthogonalDeepMatrixFactorization(nn.Module):
    """
    Setup for deep matrix factorization with one zero initialization and rest orthogonal.
    """

    def __init__(self, d, depth, init_scale=1e-2):
        super(OrthogonalDeepMatrixFactorization, self).__init__()

        # self.factors = nn.ParameterList([nn.Parameter(torch.zeros(d, d))] # one zero initialization
        #                                 + [torch.nn.init.orthogonal_(torch.empty(d, d), gain=init_scale) for _ in range(depth-1)]) # others are orthogonal
        # self.factors = nn.ParameterList([nn.Parameter(torch.zeros(d, d))] # one zero initialization
        #                                 + [torch.nn.init.eye_(torch.empty(d, d))*init_scale for _ in range(depth-1)]) # others are diagonal
        self.factors = nn.ParameterList([torch.nn.init.eye_(torch.empty(d, d))*init_scale] # one zero initialization
                            + [torch.nn.init.eye_(torch.empty(d, d))*init_scale for _ in range(depth-1)]) # others are diagonal
        # self.factors = nn.ParameterList([nn.Parameter(torch.nn.init.orthogonal_(torch.empty(d, d), gain=init_scale))  # First layer orthogonal
        #   ] + [nn.Parameter(torch.nn.init.orthogonal_(torch.empty(d, d), gain=init_scale)) for _ in range(depth-1)])

    # def forward(self):
    #     product = self.factors[0]
    #     for matrix in self.factors:
    #         product = torch.matmul(matrix, product)  # Apply sigmoid activation and then matrix multiplication
    #     return product
    
    def forward(self):
        product = self.factors[0]
        for i in range(1, len(self.factors)):
            product = product @ self.factors[i]
        return product

def generate_data(shape, rank):
    mat = torch.randn(shape)
    #mat =torch.eye(shape[0])
    U, S, V = torch.svd(mat)
    return U[:, :rank] @ torch.diag(S[:rank]) @ V[:, :rank].T, U, S, V

def train_gd(step_size, jump, n_outer_loops, n_inner_loops, tol, model, target, u_star, v_star, r):
    criterion = torch.nn.MSELoss(reduction='sum')
    #optimizer = torch.optim.SGD(model.parameters(), lr=step_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=step_size, momentum=0.9)
    test_losses = []
    w4_w3_errors, w3_w2_errors, w2_w1_errors = [], [], []
    w4_u_errors, w3_u_errors, w1_v_errors = [], [], []

    first_layer_first_sing, first_layer_second_sing, first_layer_third_sing, first_layer_fourth_sing = [], [], [], []
    second_layer_first_sing, second_layer_second_sing, second_layer_third_sing, second_layer_fourth_sing = [], [], [], []
    third_layer_first_sing, third_layer_second_sing, third_layer_third_sing, third_layer_fourth_sing = [], [], [], []
    

    ## create and make a folder
    os.makedirs(f'./eos_deep', exist_ok=True)

    pbar = tqdm(range(n_outer_loops))
    # pbar = tqdm(range(48000, n_outer_loops))
    # model.load_state_dict(torch.load(f'eos_deep_{step_size}.pt'))

    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = jump * step_size
    for itr in pbar:
            ## save the model
            #torch.save(model.state_dict(), f'eos_deep_{step_size}.pt')
            ## load the model
        for _ in range(n_inner_loops):
            # Update
            optimizer.zero_grad()
            train_loss = criterion(model(), target)
            train_loss.backward()
            optimizer.step()
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name} - Gradient Norm: {param.grad.data.norm().item()}")
            

        test_loss = criterion(model(), target)
        test_losses.append(test_loss.detach().numpy())
        if itr % 1000 == 0:
            # hessian_matrix = compute_full_hessian(model, criterion, target)
            # np.set_printoptions(threshold=np.inf, linewidth=np.inf)
            # print(hessian_matrix)
            # # print("Matrix Factors:")
            # # for idx, factor in enumerate(model.factors):
            # #     print(f"Factor {idx}:")
            # #     print(factor.detach().cpu().numpy())
            # svd_results = extract_blocks_and_svd(hessian_matrix, d=2)
            # for idx, (U, S, V) in enumerate(svd_results):
            #     print(f"Block {idx}:")
            #     print("U:", U)
            #     print("S:", S)
            #     print("V:", V)
            #     print()

            ## calculate top 20 eigenvales  
            

            e1,e2,e3,e4,e5,e6,e7,e8,e9,e10 = get_hessian_eigenvalues(model, criterion, target,neigs=10)
            print(f'Eigenvalues: {e1.item()}, {e2.item()}, {e3.item()}, {e4.item()}, {e5.item()}')
            print(f'Eigenvalues: {e6.item()}, {e7.item()}, {e8.item()}, {e9.item()}, {e10.item()}')
            # Get the top 20 eigenvalues
            # e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15, e16, e17, e18, e19, e20 = get_hessian_eigenvalues(model, criterion, target, neigs=20)

            # # Print the top 20 eigenvalues
            # print('Top 20 Eigenvalues:')
            # print(f'Eigenvalues: {e1.item()}, {e2.item()}, {e3.item()}, {e4.item()}, {e5.item()}')
            # print(f'Eigenvalues: {e6.item()}, {e7.item()}, {e8.item()}, {e9.item()}, {e10.item()}')
            # print(f'Eigenvalues: {e11.item()}, {e12.item()}, {e13.item()}, {e14.item()}, {e15.item()}')
            # print(f'Eigenvalues: {e16.item()}, {e17.item()}, {e18.item()}, {e19.item()}, {e20.item()}')
            # eigenvalues = get_hessian_eigenvalues(model, criterion, target, neigs=40)

            # # Print the top 40 eigenvalues
            # print('Top 40 Eigenvalues:')
            # for i in range(0, 40, 5):
            #     print(f'Eigenvalues: ({eigenvalues[i].item()}, {eigenvalues[i+1].item()}, {eigenvalues[i+2].item()}, {eigenvalues[i+3].item()}, {eigenvalues[i+4].item()})')


        with torch.no_grad():
        # Singular values and error tracking
            u4, s_first_layer, v4 = torch.svd(model.factors[0])
            u3, s_second_layer, v3 = torch.svd(model.factors[1])
            u2, s_third_layer, v2 = torch.svd(model.factors[2])

        #     w4_w3_errors.append(torch.linalg.norm(torch.abs(v4[:, :r].T @ u3[:, :r]) - torch.eye(r)).detach().numpy())
        #     w3_w2_errors.append(torch.linalg.norm(torch.abs(v3[:, :r].T @ u2[:, :r]) - torch.eye(r)).detach().numpy())
        #     w4_u_errors.append(torch.linalg.norm(torch.abs(u4[:, :r].T @ u_star[:, :r]) - torch.eye(r)).detach().numpy())
        #     #w1_v_errors.append(torch.linalg.norm(torch.abs(v2[:, :r].T @ v_star[:, :r]) - torch.eye(r)).detach().numpy())

            first_layer_first_sing.append(s_first_layer[0].detach().numpy())
            first_layer_second_sing.append(s_first_layer[1].detach().numpy())
            first_layer_third_sing.append(s_first_layer[2].detach().numpy())
            first_layer_fourth_sing.append(s_first_layer[3].detach().numpy())
            second_layer_first_sing.append(s_second_layer[0].detach().numpy())
            second_layer_second_sing.append(s_second_layer[1].detach().numpy())
            second_layer_third_sing.append(s_second_layer[2].detach().numpy())
            second_layer_fourth_sing.append(s_second_layer[3].detach().numpy())
            third_layer_first_sing.append(s_third_layer[0].detach().numpy())
            third_layer_second_sing.append(s_third_layer[1].detach().numpy())
            third_layer_third_sing.append(s_third_layer[2].detach().numpy())
            third_layer_fourth_sing.append(s_third_layer[3].detach().numpy())

        pbar.set_description(f"Train loss: {train_loss:.2e}, test loss: {test_loss:.2e}")

        if train_loss < tol:
            break

    # # Figure 1: Reconstruction Errors
    # fig1, ax1 = plt.subplots(figsize=(10, 6))  # Adjust figsize for better visibility
    # ax1.semilogy(w4_w3_errors, label="||V(W4)^T U(W3) - I ||")
    # ax1.semilogy(w3_w2_errors, label="||V(W3)^T U(W2) - I ||")
    # ax1.semilogy(w4_u_errors, label="||U(W4)^T U(M) - I ||")
    # #ax1.semilogy(w1_v_errors, label="||V(W1)^T V(M) - I ||")
    # ax1.legend(title="Singular vector allignment")
    # ax1.set_xlabel('Iterations')
    # ax1.set_ylabel('Log Scale Error')
    # ax1.set_title(f'Singular Vector Allignment at step size {jump}')
    # plt.tight_layout()
    # plt.savefig(f'eos_deep/reconstruction_errors_{jump}.png')  # Save the first plot

    # # Figure 2: Test Losses
    # fig2, ax2 = plt.subplots(figsize=(10, 6))
    # ax2.semilogy(test_losses)
    # ax2.set_xlabel('Iterations',fontsize=14)
    # ax2.set_ylabel('Reconstruction Error',fontsize=14)
    # ax2.set_title(f'Reconstruction loss at step size {jump}')
    # plt.tight_layout()
    # plt.savefig(f'eos_deep/test_losses_{jump}.png')  # Save the second plot

    # Figure 3: Singular Values

    # Singular values data for the first layer
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(first_layer_first_sing[0:1000], label='Layer 1, Singular value 1')
    ax1.plot(first_layer_second_sing[0:1000], label='Layer 1, Singular value 2')
    ax1.plot(first_layer_third_sing[0:1000], label='Layer 1, Singular value 3')
    ax1.set_xlabel('Iterations',fontsize=14)
    ax1.set_ylabel('Singular Value',fontsize=14)
    ax1.set_title(f'Singular Values of Layer-1 at step size {jump}')
    ax1.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(f'eos_deep/layer_1_singular_values_{jump}.png')

    # Singular values data for the second layer
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(second_layer_first_sing[0:1000], label='Layer 2, Singular value 1')
    ax2.plot(second_layer_second_sing[0:1000], label='Layer 2, Singular value 2')
    ax2.plot(second_layer_third_sing[0:1000], label='Layer 2, Singular value 3')
    ax2.set_title(f'Singular Values of Layer-2 at step size {jump}')
    ax2.set_xlabel('Iterations',fontsize=14)
    ax2.set_ylabel('Singular Value')
    ax2.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(f'eos_deep/layer_2_singular_values_{jump}.png')

    # Singular values data for the third layer
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(third_layer_first_sing[0:1000], label='Layer 3, Singular value 1')
    ax3.plot(third_layer_second_sing[0:1000], label='Layer 3, Singular value 2')
    ax3.plot(third_layer_third_sing[0:1000], label='Layer 3, Singular value 3')
    ax3.set_title(f'Singular Values of Layer-3 at step size {jump}')
    ax3.set_xlabel('Iterations',fontsize=14)
    ax3.set_ylabel('Singular Value',fontsize=14)
    ax3.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(f'eos_deep/layer_3_singular_values_{jump}.png')


    # # Adjust layout (optional)
    # plt.tight_layout()

    return {'test_loss': test_losses}

def main(args):
    d = 10
    r = 3
    depth = 3
    n_outer_loops = args.iters
    n_inner_loops = 1
    tol = 0
    target,U,S,V = generate_data((d, d), r)
    
    u, s, v = torch.svd(target)
    # print("target svd"  ,u,s,v)
    
    s[0] = 10
    s[1] = 9.5
    s[2] = 2
    target = u @ torch.diag(s) @ v.T
    model = OrthogonalDeepMatrixFactorization(d, depth, init_scale=args.init_scale)
    results = train_gd(args.step_size, args.jump, n_outer_loops, n_inner_loops, tol, model, target,u,v,r)

    plt.figure(figsize=(10, 3))
    plt.semilogy(results['test_loss'])
    plt.ylabel('Reconstruction Error',fontsize=14)
    plt.xlabel('Iterations',fontsize=14)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step-size", type=float, default=20.0, help="Step size for the gradient descent.")
    parser.add_argument("--jump", type=float, default=1.0, help="Factor to jump the learning rate after 48000 iterations.")
    parser.add_argument("--init-scale", type=float, default=0.1, help="Initialization scale for orthogonal matrices.")
    parser.add_argument("--iters", type=int, default=60000, help="number of iterations")
    
    args = parser.parse_args()
    main(args)
