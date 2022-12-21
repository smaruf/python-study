import jax
import jax.numpy as jnp
from jax import random
from jax import grad, jit

key = random.PRNGKey(0)

def selu_np(x, alpha=1.67, lmbda=1.05):
  return lmbda * np.where(x > 0, x, alpha * np.exp(x) - alpha)

def selu_jax(x, alpha=1.67, lmbda=1.05):
  return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

x = random.normal(key, (1000000,))

selu_jax_jit = jit(selu_jax)
%time x_jax = jax.device_put(x) 
%time selu_jax_jit(x_jax).block_until_ready() 
%timeit selu_jax_jit(x_jax).block_until_ready() 
