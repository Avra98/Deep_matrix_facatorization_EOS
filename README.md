## Analyzing deep matrix factorization at Edge of Stability


This repository contains the source code for observing edge of stability oscillations in a small subspace in deep matrix factorization. 


## Table of Contents
- [One-time Setup](#one-time-setup)
  - [Installation](#installation)
  - [Environment Setup](#environment-setup)
- [Updating](#updating)

- [Working with the Code](#working-with-the-code)
  - [EOS-Deep](#EOS-Deep) 



- [Contributing](#contributing)
- [License](#license)


##  Setup
1. Install conda (if not already installed).
2. Create the environment: 
```bash
conda create --name deep_eos 
```
3. Activate environment:
 ```bash
 conda activate deep_eos
 ```
4. Install the required packages:
```bash
pip install -r requirements.txt
```


## Working 

### EOS-Deep: 

```python   
python eos_deep.py --step-size=0.077 --init-scale=0.1 --num-steps=10000
```

The following code shows that whenever $\frac{2}{\text{step-size}} > L s_i^{2-\frac{2}{L}}$, then oscillations start occurring in all singular values greater than $s_i$.

All other singular values and singular vectors remain stationary.

