import jax.numpy as jnp

size = 5000
x = jnp.random.normal(size=(size, size)).astype(np.float32)
%timeit jnp.dot(x, x.T).block_until_ready()
