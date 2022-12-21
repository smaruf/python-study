import jax
import jax.numpy as jnp
from jax import random

key = random.PRNGKey(0)
size = 5000

x = random.normal(key, (size, size)).astype(jnp.float32)
%time x_jax = jax.device_put(x)
%time jnp.dot(x_jax, x_jax.T).block_until_ready()
%timeit jnp.dot(x_jax, x_jax.T).block_until_ready()
