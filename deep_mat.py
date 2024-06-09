import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sam import SAM
import matplotlib.pylab as plt 
import torch.nn.functional as F
import random
from scipy.sparse.linalg import LinearOperator, eigsh
from torch.nn.utils import parameters_to_vector, vector_to_parameters

class DeepMatrixFactorization_noiseinj(nn.Module):
    def __init__(self, n, fac, alpha=1e-3):
        super(DeepMatrixFactorization_noiseinj, self).__init__()
        self.scaling_factor = alpha ** (1.0 / fac)
        self.factors = nn.ParameterList([nn.Parameter(torch.randn(n, n) * self.scaling_factor) for _ in range(fac)])

    def forward(self, noise_std=0.0):
        product = self.factors[0]
        if noise_std > 0:
            noise = torch.normal(mean=0., std=noise_std * self.scaling_factor, size=product.shape).to(product.device)
            product = product + noise

        for i in range(1, len(self.factors)):
            factor = self.factors[i]
            if noise_std > 0:
                noise = torch.normal(mean=0., std=noise_std * self.scaling_factor, size=factor.shape).to(factor.device)
                factor = factor + noise
            product = product @ factor
        return product

    

# class DeepMatrixFactorization(nn.Module):
#     def __init__(self, n, fac, alpha=1e-3):
#         super(DeepMatrixFactorization, self).__init__()
#         scaling_factor = alpha ** (1.0 / fac)
#         # Each factor is of dimension n x n
#         self.factors = nn.ParameterList([nn.Parameter(torch.randn(n, n) * scaling_factor) for _ in range(fac)])

#     def forward(self):
#         product = self.factors[0]
#         for i in range(1, len(self.factors)):
#             product = product @ self.factors[i]
#         return product


class DeepMatrixFactorization(nn.Module):
    def __init__(self, n, fac, alpha=1e-3, activation='relu',init='normal'):
        super(DeepMatrixFactorization, self).__init__()
        scaling_factor = alpha ** (1.0 / fac)
        if init=='normal':
            self.factors = nn.ParameterList([nn.Parameter(torch.randn(n, n) * scaling_factor) for _ in range(fac)])
        elif init=='ortho': 
            self.factors = nn.ParameterList([nn.Parameter(torch.nn.init.orthogonal_(torch.empty(n, n)) * scaling_factor) for _ in range(fac)]
        )
        #self.factors[0] = nn.Parameter(torch.zeros(n, n))
        self.activation_type = activation
        if self.activation_type == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.01)
        # Initialize other activations here if needed

    def forward(self):
        product = self.factors[0]
        for i in range(1, len(self.factors)):
            if self.activation_type == 'relu':
                product = F.relu(product)
            elif self.activation_type == 'leakyrelu':
                product = self.activation(product)
            elif self.activation_type == 'tanh':
                product = torch.tanh(product)
            elif self.activation_type == 'sigmoid':
                product = torch.sigmoid(product)
            elif self.activation_type == 'hardtanh':
                product = F.hardtanh(product)
            # No activation if 'none'
            product = product @ self.factors[i]
        return product
    


def lanczos(matrix_vector, dim: int, neigs: int):
    def mv(vec: np.ndarray):
        gpu_vec = torch.tensor(vec, dtype=torch.float).cuda()
        return matrix_vector(gpu_vec)
    operator = LinearOperator((dim, dim), matvec=mv)
    evals, evecs = eigsh(operator, neigs)
    return torch.from_numpy(np.ascontiguousarray(evals[::-1]).copy()).float(), \
           torch.from_numpy(np.ascontiguousarray(np.flip(evecs, -1)).copy()).float()


def compute_hvp(model, loss_fn, A, y, vector):
    # Determine the device used by the model
    device = next(model.parameters()).device

    # Move input tensors to the same device as the model
    A = A.to(device)
    y = y.to(device)
    vector = vector.to(device)

    model.zero_grad()

    # Forward pass
    output = model()
    traces = trace_operator(A, output)

    # Compute the loss
    loss = loss_fn(traces, y)

    # Compute gradients of the loss w.r.t. model parameters
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    grads_vector = parameters_to_vector(grads)

    # Dot product of gradients and the input vector
    dot_product = torch.dot(grads_vector, vector)

    # Compute Hessian-vector product
    hvp = torch.autograd.grad(dot_product, model.parameters(), retain_graph=True)

    return parameters_to_vector(hvp)

def get_hessian_eigenvalues(model, loss_fn, A,y,neigs=2):
    """ Compute the leading Hessian eigenvalues. """
    hvp_delta = lambda delta: compute_hvp(model, loss_fn, A,y,delta).detach().cpu()
    nparams = len(parameters_to_vector((model.parameters())))
    evals, evecs = lanczos(hvp_delta, nparams, neigs=neigs)
    return evals

def get_trace(model, loss_fn, A,y, n_iters=50):
    hvp_delta = lambda delta: compute_hvp(model, loss_fn,  A,y,delta).detach().cpu()
    nparams = len(parameters_to_vector((model.parameters())))
    trace=0.0
    for _ in range(n_iters):
        v = torch.randn(nparams)
        Hv = hvp_delta(v)
        trace += torch.dot(Hv, v).item()
    return trace/n_iters 


def compute_alignment(model):
    alignment_list = []
    for i in range(len(model.factors) - 1):
        _, _, v1 = torch.svd(model.factors[i])  # Right singular vectors of factor i
        u2, _, _ = torch.svd(model.factors[i + 1])  # Left singular vectors of factor i+1
        
        # Compute the cosine similarities between the corresponding singular vectors
        alignment = torch.trace(v1.T @ u2)/v1.shape[0]
        
        # Compute and store the mean alignment
        alignment_list.append(alignment.item())
    return alignment_list

def generate_data(n, r, meas_scale, problem_type, sampling_rate=0.3, noise_variance=0.01):
    U, singular_values, V = np.linalg.svd(np.random.randn(n, n))
    singular_values = np.abs(singular_values)  # Ensure positive singular values
    S = np.diag(np.concatenate((singular_values[:r], [0] * (n-r))))  # Low-rank singular values
    X_star = U @ S @ V
    print(np.linalg.matrix_rank(X_star))

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

def optimize(n, fac, r,meas_scale,batch_scale, optima, lr,beta, num_epochs, problem_type, alpha =1e-3, reg=0.05,  sampling_rate=0.5, noise_variance=0.01,show_every=100,activation='none', init='normal'):
    # Generate synthetic data
    y, X_star, A = generate_data(n,r, meas_scale,problem_type, sampling_rate, noise_variance)
  
    model = DeepMatrixFactorization(n,fac,alpha,activation,init)
    
    #print(model.factors)
    #print("AA^T is:",model.factors[2]@ model.factors[2].T)
    # Define the loss functions
    loss_fn = nn.MSELoss()   
    singular_values_list = []
    epochs_list = []    
    alignments_list = []
    loss_list = []
    rec_loss_list = []
    trace_list= []
    eig=[]
    #print(loss_fn(trace_operator(A, X_star), y))
    m = len(y)  # Total number of measurements
    indices = list(range(m))
    batch_size = int(args.batch_scale*m)
    

    # Define the optimizer
    if optima == 'SGD' or 'GD':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optima == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optima == "SAM":
        base_opt = torch.optim.SGD
        optSAM = SAM(model.parameters(), base_opt, rho=args.reg, adaptive=False, lr=lr)
    else:
        raise ValueError("Invalid optimizer. Please choose 'SGD' or 'Adam'.")
    
    if optima == "SGD":
        num_epochs = int(num_epochs * args.batch_scale)
        show_every = int(show_every * args.batch_scale)
    
    # Perform optimization
    for epoch in range(num_epochs):
        if optima == "GD" or optim == "Adam":
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
        elif optima == "SGD":
            random.shuffle(indices)
            
            for i in range(0, m, batch_size):
                batch_indices = indices[i:i + batch_size]
                y_batch = y[batch_indices]
                A_batch = A[batch_indices]
                
                optimizer.zero_grad()
                output = model()
                
                if problem_type == 'gaussian_sensing':
                    traces = get_trace(model ,loss_fn, A,y, n_iters=50)
                    #traces = trace_operator(A_batch, output)
                    loss = loss_fn(traces, y_batch)
                else:
                    raise ValueError("SGD only implemented for 'gaussian_sensing'")
                
                loss.backward()
                optimizer.step()

        #elif optima == "PAC-Bayes":
            


        elif optima == "SAM":
            output = model()
            loss = loss_fn(y, A*output)
            loss.backward()
            optSAM.first_step(zero_grad=True)
            loss_fn(y, A*model()).backward()
            optSAM.second_step(zero_grad=True)
      
           # Compute additional metrics


        if epoch % show_every==0:
            #reconstruction_loss = loss_fn(output, X_star.T)
            reconstruction_loss = loss_fn(output, X_star)
            u, s, v = torch.svd(output) 
            first_20_singular_values = s[:20].cpu().detach().numpy()
            singular_values_list.append(first_20_singular_values)
            epochs_list.append(epoch)
            loss_list.append(torch.log(loss).item())
            rec_loss_list.append(torch.log(reconstruction_loss).item())

            #trace = get_trace(model, loss_fn, A,y)
            e1,e2=get_hessian_eigenvalues(model, loss_fn, A,y)
            
            # print(f"Trace: {trace}")
            # trace_list.append(trace)
            # print(f"Eigenvalues: {e1}")
            # eig.append(e1)

            #alignment_list = compute_alignment(model)

        
            # Print metrics
            print(f"Iteration {epoch+1}:")
            #print(f"Training Loss: {loss.item()/torch.norm(X_star, p='fro')}")
            print(f"Training Loss: {torch.log(loss).item()}")
            #print(f"Reconstruction Loss: {reconstruction_loss.item()/torch.norm(X_star, p='fro')}")
            print(f"Reconstruction Loss: {torch.log(reconstruction_loss).item()}")
            #print(f"Alignment: {alignment_list}")


            #print(f"Trace Norm: {trace_norm.item()}")
            # print(f"Stable Rank:  {np.linalg.norm(output.detach().numpy(), 'fro')/np.linalg.norm(output.detach().numpy(),2)}")
            #print("=" * 20)
            # eigenvalues = torch.linalg.eigvals(output)
            # real_eigenvalues = eigenvalues.real
            #print(real_eigenvalues )
            
            # Plot and save eigenvalue distribution
            # plt.figure()
            # plt.hist(real_eigenvalues.detach().numpy(), bins=20)
            # plt.xlabel('Real Part of Eigenvalues')
            # plt.ylabel('Frequency')
            # plt.title('Eigenvalue Distribution')
            # plt.savefig(f'eigenvalues_epoch_{epoch+1}.png')
            # plt.close()
    filename = f"factor_plot_alpha_{alpha}_lr_{lr}_n_{n}_fac_{fac}_r_{r}_act_{activation}_init_{init}.png"
    plt.figure(figsize=(10,15)) # You can change the figure size according to your needs

    # Subplot for Singular Values
    plt.subplot(3,1,1) # 1 row, 3 columns, 1st subplot
    for i in range(n):
        plt.plot(epochs_list, [vals[i] for vals in singular_values_list], label=f"Singular Value {i+1}")
    plt.xlabel("Epoch")
    plt.ylabel("Singular Value")
    plt.title("Singular Values over Epochs")

    # Subplot for Training Loss
    plt.subplot(3,1,2) # 1 row, 3 columns, 2nd subplot
    plt.plot(epochs_list, loss_list, label="Loss", color='r')
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss over Epochs")

    # Subplot for Reconstruction Loss
    # plt.subplot(3,1,3) # 1 row, 3 columns, 3rd subplot
    # plt.plot(epochs_list, trace_list, label="Reconstruction Loss")
    # plt.xlabel("Epoch")
    # plt.ylabel(" Hessian Trace")
    # plt.title("Hessian Trace")

    plt.subplot(3,1,3)
    plt.plot(epochs_list, rec_loss_list, label="Reconstruction Loss", color='g')
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss")
    plt.title("Reconstruction Loss over Epochs")  

    # plt.subplot(5,1,5)
    # plt.plot(epochs_list, eig, label="sharpness")
    # plt.xlabel("Epoch")
    # plt.ylabel("sharpness")
    # plt.title("sharpness")        

    # Save and show the figure
    plt.tight_layout() # Adjusts subplot parameters to give specified padding
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
# Argument Parser
# ...

# Argument Parser
parser = argparse.ArgumentParser(description='Matrix Factorization using PyTorch optimizer.')
parser.add_argument('--n', type=int, default=20, help='Dimension n')
parser.add_argument('--r', type=int, default=5, help='Rank of X_star')
parser.add_argument('--fac', type=int, default=3, help='Number of matrix factors ')
parser.add_argument('--meas_scale', type=float, default=3, help='Scale of number of measurements')
parser.add_argument('--optima', type=str, default='GD', choices=['SGD','GD', 'Adam','SAM'], help='Optimizer (SGD or Adam or SAM)')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--beta', type=float, default=0.0, help='Momentum')
parser.add_argument('--num_epochs', type=int, default=30000, help='Number of epochs')
parser.add_argument('--problem_type', type=str, default='gaussian_sensing', choices=['factorization', 'completion','gaussian_sensing'], help='Type of recovery problem')
parser.add_argument('--sampling_rate', type=float, default=0.5, help='Sampling rate for matrix completion')
parser.add_argument('--noise_variance', type=float, default=0.0, help='Variance of the noise')
parser.add_argument('--alpha', type=float, default=1e-3, help='Initialization scaling factor')
parser.add_argument('--reg', type=float, default=0.05, help='SAM Radius')
parser.add_argument('--batch_scale', type=float, default=0.2, help='batch size for SGD')
parser.add_argument('--show_every', type=int, default=100, help='store result after how many epochs')
parser.add_argument('--activation', type=str, default="none", help='what type of non-linearity to use')
parser.add_argument('--init', type=str, default="normal",choices=["normal","ortho"], help='what type of non-linearity to use')    



args = parser.parse_args()

# Run the optimization
optimize(args.n, args.fac, args.r, args.meas_scale,args.batch_scale, args.optima, 
        args.lr,args.beta, args.num_epochs, args.problem_type, args.alpha, args.reg,
        args.sampling_rate, args.noise_variance, args.show_every,args.activation, args.init)         