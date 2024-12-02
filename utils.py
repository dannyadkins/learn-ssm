import jax
import numpy as np

def random_SSM(rng, N):
    a_r, b_r, c_r = jax.random.split(rng, 3)
    A = jax.random.uniform(a_r, (N, N))
    B = jax.random.uniform(b_r, (N, 1))
    C = jax.random.uniform(c_r, (1, N))
    return A, B, C

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    N = 4
    A, B, C = random_SSM(key, N)
    
    # A shape should be (N,N), as it maps a hidden state x(t) to a new hidden state x'(t) 
    print("A shape:", A.shape)
    # B shape should be (N,1) as it encodes a 1D input signal u(t) to a new hidden state x'(t) 
    print("B shape:", B.shape) 
    # C shape should be (1,N) as it decodes a hidden state x(t) to a 1D output signal y(t) 
    print("C shape:", C.shape)

    # Let's simulate the SSM on some mock input data
    T = 10  # number of timesteps
    u = jax.random.uniform(key, (T, 1))  # random input signal
    x = np.zeros((T, N))  # hidden states
    y = np.zeros((T, 1))  # output signal

    # Simulate the system
    for t in range(1, T):
        x[t] = A @ x[t-1] + B @ u[t]  # update hidden state
        y[t] = C @ x[t]  # compute output

    print("\nSimulation results:")
    print("Input signal shape:", u.shape)
    print("Hidden states shape:", x.shape)
    print("Output signal shape:", y.shape)
    print("\nFirst few timesteps:")
    print("t=0:", "u =", u[0][0], "y =", y[0][0])
    print("t=1:", "u =", u[1][0], "y =", y[1][0])
    print("t=2:", "u =", u[2][0], "y =", y[2][0])
