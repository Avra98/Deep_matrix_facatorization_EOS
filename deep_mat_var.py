import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


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

def trace_operator(A, output):
    # Element-wise multiply and sum for each slice in the batch
    traces = (A * output).sum(dim=(1,2))
    return traces

class DeepMatrixFactorization(nn.Module):
    def __init__(self, n, fac, alpha=1e-3):
        super(DeepMatrixFactorization, self).__init__()
        scaling_factor = alpha ** (1.0 / fac)
        self.factors = nn.ParameterList([nn.Parameter(torch.randn(n, n) * scaling_factor) for _ in range(fac)])

class NoiseModel(nn.Module):
    def __init__(self, n, fac):
        super(NoiseModel, self).__init__()
        self.log_noise_vars = nn.ParameterList([nn.Parameter(torch.ones(n, n).log() * 0) for _ in range(fac)])

    def forward(self):
        noise_list = [(log_noise_var.exp()).sqrt() * torch.normal(mean=0., std=1., size=log_noise_var.size()) for log_noise_var in self.log_noise_vars]
        return noise_list


def kl_divergence(log_var_posterior, log_var_prior=torch.tensor(0.0)):
    var_posterior = log_var_posterior.exp()
    var_prior = log_var_prior.exp()
    return 0.5 * torch.sum(torch.log(var_prior / var_posterior) + var_posterior / var_prior - 1)


# Initialize models and optimizers
n = 20
r=5
fac = 3
meas_scale=0.3
problem_type='gaussian_sensing'
model = DeepMatrixFactorization(n, fac)
noise_model = NoiseModel(n, fac)

optimizer_model = optim.Adam(model.parameters(), lr=1e-3)
optimizer_noise = optim.Adam(noise_model.parameters(), lr=1e-3)

loss_fn = nn.MSELoss()

# Generate synthetic data (y and A)
# ... (your data generation code here)
y, X_star, A = generate_data(n,r, meas_scale,problem_type)

# Optimization loop
num_epochs = 1000
num_noise_samples = 20  # Number of noise realizations per epoch

for epoch in range(num_epochs):
    optimizer_model.zero_grad()
    optimizer_noise.zero_grad()

    avg_loss = 0.0  # To store the average loss over multiple noise samples

    for _ in range(num_noise_samples):
        factors = model.factors  # Get the factors from the DeepMatrixFactorization model
        noise_list = noise_model()  # Get the noise from the NoiseModel

        # Initialize combined_output as the first factor plus noise
        combined_output = factors[0] + noise_list[0]

        # Loop through the remaining factors and noise, adding and then multiplying
        for i in range(1, len(factors)):
            combined_output = combined_output @ (factors[i] + noise_list[i])

        traces = trace_operator(A, combined_output)
        loss = loss_fn(y, traces)

        kl_term = 0
        for log_noise_var in noise_model.log_noise_vars:
            kl_term += kl_divergence(log_noise_var)
        loss += 0*kl_term

        avg_loss += loss / num_noise_samples  # Accumulate and average the loss

    avg_loss.backward()  # Backpropagate on the average loss

    optimizer_model.step()
    optimizer_noise.step()

    # Logging and other operations go here
    print(f"Epoch {epoch+1}, Loss: {avg_loss.item()}")


