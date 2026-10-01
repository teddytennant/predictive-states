JAX implementation of predictive-state sequence modeling, arXiv:2609.38524, RFF equation 10 and the two-stage Algorithm 1.

```bash
pip install -e .
pytest
python train.py
```

What does not match: no RNN, no mixture density network, no Figure 1 L2 curve, no claimed rates. The one-step conditional law is a Gaussian with mean linear in the estimated predictive state and diagonal residual variance, a stand-in for the paper's mixture density network. Hyperparameters are fixed rather than chosen by the paper's criterion.

Paper equation (10) draws each phase from Uniform(-pi, pi), not Uniform(0, 2 pi), and each frequency from N(0, I_d), the Gaussian spectral measure of the RBF kernel. The predictive state fit here is equation (13), Z_t = E[phi(X_{t+1:t+m+1}) | F_t]. Equation (11) is Y_{t+1} = phi(X_{t+1:t+m+1}), not that conditional expectation. The one-step law is Theorem 3, L(X_{t+1} | Z_t).

DGP 1 is scaled to r = d = 1 with A_s = 0.7 and A_x = 0.9, so the spectral radius of the linear part is below 1. Draws use a normal truncated to [-2.5, 2.5], as in section 4.
