import jax.numpy as jnp
from jaxtyping import Float, Array

# decoder
CONTEXT_LENGTH = 20
BINS = jnp.linspace(-5, 5, 1024)
TEMPERATURE = 0.8

# clipping
MAX_ACC_DELTA = 0.5
VELOCITY_CLIP = 0.5

# phys stuff
ACC_G = 9.81

# i hate you
def encode(value: Float[Array, "d"]):
    """Encode lataccel to token."""
    value = jnp.clip(value, -5, 5)
    return jnp.digitize(value, BINS, right=True)

def decode(token):
    """Decode token to lataccel."""
    return BINS[jnp.clip(token, 0, 1023)]
