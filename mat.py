import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch import Tensor
from sam import SAM
import matplotlib.pylab as plt 
from torch.nn.utils import parameters_to_vector
from scipy.sparse.linalg import LinearOperator,eigsh



# Define the matrix factorization model
class MatrixFactorization(nn.Module):
    def __init__(self, n, k, alpha=1e-3):
        super(MatrixFactorization, self).__init__()
        self.F = nn.Parameter(torch.randn(n, k) * torch.sqrt(torch.tensor(alpha)))
        #self.F.requires_grad_(True)

    def forward(self):
        return self.F @ self.F.t()

# Function to generate synthetic data
def generate_data(n, k, r,meas_scale, problem_type, sampling_rate=0.3, noise_variance=0.01):
    X_star = np.random.randn(n, n)
    X_star = (X_star + X_star.T) / 2  # Ensure X_star is symmetric
    eigvals, eigvecs = np.linalg.eig(X_star)
    eigvals = np.abs(eigvals)  # Take absolute values for positive semi-definite matrix
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx][:r]
    eigvecs = eigvecs[:, idx][:, :r]
    X_star = eigvecs @ np.diag(eigvals) @ eigvecs.T
    if problem_type == 'factorization':
        A = np.ones((n, n))
        noise = np.random.normal(scale=noise_variance, size=(n, n))
        y = A * X_star + noise
    elif problem_type == 'completion':
        mask = np.random.choice([0, 1], size=(n, n), p=[1 - sampling_rate, sampling_rate])
        A = mask
        noise = np.random.normal(scale=noise_variance, size=(n, n))
        y = A * X_star + noise
    elif problem_type == 'gaussian_sensing':  
        m=int(meas_scale*n*r)
        measurements = np.zeros(m)
        A_matrices = []
        for i in range(m):
            A_i = np.random.randn(n, n)
            measurement = np.trace(np.dot(A_i, X_star))
            noise = np.random.normal(scale=noise_variance)
            measurement += noise
            measurements[i] = measurement
            A_matrices.append(A_i)      
        A = np.stack(A_matrices)
        y = measurements
    else:
        raise ValueError("Invalid problem type. Please choose 'factorization' or 'completion'.")
    
    return torch.tensor(y, dtype=torch.float), torch.tensor(X_star, dtype=torch.float), torch.tensor(A, dtype=torch.float)

# Main function for optimization
def optimize(n, k, r,meas_scale, optima, lr,beta, num_epochs, problem_type, alpha =1e-3, reg=0.05,  sampling_rate=0.5, noise_variance=0.01):
    # Generate synthetic data
    y, X_star, A = generate_data(n, k, r, meas_scale,problem_type, sampling_rate, noise_variance)
    model = MatrixFactorization(n, k ,alpha)
   
    # Define the loss functions
    loss_fn = nn.MSELoss()       
    #print(loss_fn(trace_operator(A, X_star), y))
    singular_values_list = []
    epochs_list = []
    sharpness_list = []
    hessian_eigenvalues_list = []
    trace_list = []
    
    # Define the optimizer
    if optima == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr,momentum=beta)
    elif optima == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optima == "SAM":
        base_opt = torch.optim.SGD
        optSAM = SAM(model.parameters(), base_opt, rho=args.reg, adaptive=False, lr=lr)
    else:
        raise ValueError("Invalid optimizer. Please choose 'SGD' or 'Adam'.")
    
    # Perform optimization
    for epoch in range(num_epochs):
        if optima == "SGD" or optim == "Adam":
            optimizer.zero_grad()
            output = model()
            if problem_type == 'gaussian_sensing':
                traces = trace_operator(A, output)
                loss = loss_fn(traces, y)
            else:
                loss = loss_fn(y, A*output)    
            # with torch.no_grad():
            #     if epoch == 0:
            #         print("Loss at initialization is:",loss_fn(output, X_star))                   
            loss.backward()
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         print(name, param.grad)
            optimizer.step()

        elif optima == "SAM":
            output = model()
            loss = loss_fn(y, A*output)
            loss.backward()
            optSAM.first_step(zero_grad=True)
            loss_fn(y, A*model()).backward()
            optSAM.second_step(zero_grad=True)
      
        
        # Compute additional metrics


        if epoch%200==0:
            reconstruction_loss = loss_fn(output, X_star)
            u, s, v = torch.svd(output) 
            first_20_singular_values = s[:20].cpu().detach().numpy()
            singular_values_list.append(first_20_singular_values)
            epochs_list.append(epoch)

            hessian_eigenvalues = get_hessian_eigenvalues(model, loss_fn, A, y, problem_type, neigs=2)
            hessian_eigenvalues_list.append(list(hessian_eigenvalues))
            trace_hess = get_trace(model, loss_fn, A, y, problem_type, n_iters=50)
            trace_list.append(trace_hess)
            # output_min = output.min().item()
            # output_max = output.max().item()
            # print(f"Output matrix range: [{output_min}, {output_max}]")

            # # For X_star matrix
            # x_star_min = X_star.min().item()
            # x_star_max = X_star.max().item()
            # print(f"X_star matrix range: [{x_star_min}, {x_star_max}]")
            #trace_norm = torch.norm(output, p='nuc')
        
            # Print metrics
            print(f"Iteration {epoch+1}:")
            #print(f"Training Loss: {loss.item()/torch.norm(X_star, p='fro')}")
            print(f"Training Loss: {loss.item()}")
            #print(f"Reconstruction Loss: {reconstruction_loss.item()/torch.norm(X_star, p='fro')}")
            print(f"Reconstruction Loss: {reconstruction_loss.item()}")
            print(f"Hessian Eigenvalues: {hessian_eigenvalues}")
            print(f"Trace of Hessian: {trace_hess}")
            #print(f"Trace Norm: {trace_norm.item()}")
            # print(f"Stable Rank:  {np.linalg.norm(output.detach().numpy(), 'fro')/np.linalg.norm(output.detach().numpy(),2)}")
            #print("=" * 20)
            # eigenvalues = torch.linalg.eigvals(output)
            # real_eigenvalues = eigenvalues.real
            # #print(real_eigenvalues )
            
            # # Plot and save eigenvalue distribution
            # plt.figure()
            # plt.hist(real_eigenvalues.detach().numpy(), bins=20)
            # plt.xlabel('Real Part of Eigenvalues')
            # plt.ylabel('Frequency')
            # plt.title('Eigenvalue Distribution')
            # plt.savefig(f'eigenvalues_epoch_{epoch+1}.png')
            # plt.close()

    filename = f"plot_alpha_{alpha}_lr_{lr}_n_{n}_k_{k}_r_{r}_beta_{beta}_noisevar_{noise_variance}_optima_{optima}.png"
    filenameh = f"plot_hessian_eigenvalues_alpha_{alpha}_lr_{lr}_n_{n}_k_{k}_r_{r}_beta_{beta}_noisevar_{noise_variance}_optima_{optima}.png"
    for i in range(20):
        plt.plot(epochs_list, [vals[i] for vals in singular_values_list], label=f"Singular Value {i+1}")
    plt.xlabel("Epoch")
    plt.ylabel("Singular Value")
    plt.title("Singular Values over Epochs")
    #plt.legend()
    plt.savefig(filename, dpi=300) 
    plt.show()    

    num_eigenvalues = len(hessian_eigenvalues_list[0])  # Assuming that hessian_eigenvalues_list is a list of lists or a 2D array
    for i in range(num_eigenvalues):
        plt.plot(epochs_list, [vals[i] for vals in hessian_eigenvalues_list], label=f"Hessian Eigenvalue {i+1}")
    plt.xlabel("Epoch")
    plt.ylabel("Hessian Eigenvalue")
    plt.title("Hessian Eigenvalues over Epochs")
    #plt.legend()
    plt.savefig(filename, dpi=300)
    plt.show()

 
    output_np = output.detach().numpy()
    X_star_np = X_star.detach().numpy() 
    save_matrix_as_image(output_np, "output_matrix.png")
    save_matrix_as_image(X_star_np, "X_star_matrix.png")

def trace_operator(A, output):
    # Element-wise multiply and sum for each slice in the batch
    traces = (A * output).sum(dim=(1,2))
    return traces

def save_matrix_as_image(matrix, filename):
    plt.imshow(matrix, cmap='viridis')  # You can choose a different colormap if you like
    plt.colorbar()
    plt.savefig(filename)
    plt.close()  


def compute_hvp(model: nn.Module, loss_fn: nn.Module, A: Tensor, y: Tensor, vector: Tensor, problem_type: str):
    p = sum(p.numel() for p in model.parameters())
    hvp = torch.zeros(p, dtype=torch.float)  # hvp tensor on CPU
    # Do not move vector to GPU
    output = model()  # Ensure model is on CPU
    if problem_type == 'gaussian_sensing':
        traces = trace_operator(A, output)
        loss = loss_fn(traces, y) / len(y)
    else:
        loss = loss_fn(y, A * output)
        
    grads = torch.autograd.grad(loss, inputs=model.parameters(), create_graph=True)
    dot = parameters_to_vector(grads).mul(vector).sum()  # Ensure that both operands are on CPU
    grads = [g.contiguous() for g in torch.autograd.grad(dot, model.parameters(), retain_graph=True)]
    hvp = parameters_to_vector(grads)
    
    return hvp

def lanczos(matrix_vector, dim: int, neigs: int):
    def mv(vec: np.ndarray):
        gpu_vec = torch.tensor(vec, dtype=torch.float)  # Do not move to GPU
        return matrix_vector(gpu_vec)

    operator = LinearOperator((dim, dim), matvec=mv)  # Ensure LinearOperator is available in your namespace
    evals, evecs = eigsh(operator, neigs)
    return torch.from_numpy(evals[::-1].copy()).float(), torch.from_numpy(np.flip(evecs, -1).copy()).float()


def get_hessian_eigenvalues(model: nn.Module, loss_fn: nn.Module, A: Tensor, y: Tensor, problem_type: str, neigs=6):
    hvp_delta = lambda delta: compute_hvp(model, loss_fn, A, y, delta, problem_type).detach()
    nparams = len(parameters_to_vector(model.parameters()))
    evals, evecs = lanczos(hvp_delta, nparams, neigs=neigs)
    return evals

def get_trace(network: nn.Module, loss_fn: nn.Module, A: torch.Tensor, y: torch.Tensor, 
              problem_type: str, n_iters=50):
    
    # Updating the lambda function to match the new compute_hvp definition
    hvp_delta = lambda delta: compute_hvp(network, loss_fn, A, y, delta, problem_type).detach().cpu()
    
    # Getting the number of parameters in the model
    nparams = len(parameters_to_vector((network.parameters())))
    
    trace = 0.0
    for _ in range(n_iters):
        v = torch.randn(nparams).to(A.device)  # ensure v is on the same device as A and y
        Hv = hvp_delta(v)
        trace += torch.dot(Hv, v).item()
    
    return trace / n_iters

def calculate_inductive_norm(A: torch.Tensor, output: torch.Tensor) -> float:
    m = A.shape[0]  # Number of matrices in the stack
    
    sum_left = torch.zeros_like(A[0] @ A[0].t())  # Initializing the sum of A_i A_i^T
    sum_right = torch.zeros_like(A[0].t() @ A[0])  # Initializing the sum of A_i^T A_i
    
    for i in range(m):
        Ai = A[i]
        sum_left += Ai @ Ai.t()
        sum_right += Ai.t() @ Ai
    
    # Computing square roots of the summed matrices
    sum_left = sum_left.sqrt()
    sum_right = sum_right.sqrt()
    
    # Multiplying with the output and calculating the trace norm
    result_matrix = (1/m) * (sum_left @ output @ sum_right)
    trace_norm = torch.trace(result_matrix)
    
    return trace_norm.item()    
                
# Argument Parser
# ...

# Argument Parser
parser = argparse.ArgumentParser(description='Matrix Factorization using PyTorch optimizer.')
parser.add_argument('--n', type=int, default=50, help='Dimension n')
parser.add_argument('--k', type=int, default=20, help='Dimension k')
parser.add_argument('--r', type=int, default=10, help='Rank of X_star')
parser.add_argument('--meas_scale', type=float, default=3, help='Scale of number of measurements')
parser.add_argument('--optima', type=str, default='SGD', choices=['SGD', 'Adam','SAM'], help='Optimizer (SGD or Adam or SAM)')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--beta', type=float, default=0.0, help='Momentum')
parser.add_argument('--num_epochs', type=int, default=25000, help='Number of epochs')
parser.add_argument('--problem_type', type=str, default='gaussian_sensing', choices=['factorization', 'completion','gaussian_sensing'], help='Type of recovery problem')
parser.add_argument('--sampling_rate', type=float, default=0.5, help='Sampling rate for matrix completion')
parser.add_argument('--noise_variance', type=float, default=0.0, help='Variance of the noise')
parser.add_argument('--alpha', type=float, default=1e-6, help='Initialization scaling factor')
parser.add_argument('--reg', type=float, default=0.05, help='SAM Radius')

args = parser.parse_args()

# Run the optimization
optimize(args.n, args.k, args.r, args.meas_scale,args.optima, args.lr,args.beta, args.num_epochs, args.problem_type, args.alpha, args.reg, args.sampling_rate, args.noise_variance)



