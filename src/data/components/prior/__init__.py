from typing import Callable, Dict, Tuple

from src.data.components.prior.cart_coords import CART_COORDS_SAMPLERS
from src.data.components.prior.lattice_params import LATTICE_PARAMS_SAMPLERS

SAMPLER_REGISTRY: Dict[Tuple[str, str], Callable] = {}

for _prop_type, _samplers in [
    ("cart_coords", CART_COORDS_SAMPLERS),
    ("lattice_params", LATTICE_PARAMS_SAMPLERS),
]:
    for _name, _fn in _samplers.items():
        SAMPLER_REGISTRY[(_prop_type, _name)] = _fn


def get_sampler(property_type: str, algorithm_name: str) -> Callable:
    """Look up a prior sampler by property type and algorithm name."""
    key = (property_type, algorithm_name)
    if key not in SAMPLER_REGISTRY:
        available = [
            f"{pt}/{alg}" for pt, alg in sorted(SAMPLER_REGISTRY.keys())
        ]
        raise KeyError(
            f"No sampler registered for '{property_type}/{algorithm_name}'. "
            f"Available: {available}"
        )
    return SAMPLER_REGISTRY[key]
