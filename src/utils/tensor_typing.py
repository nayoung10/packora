from torch import Tensor
from jaxtyping import (
    Bool,
    Float,
    Int,
    Shaped,
)


class TorchTyping:
    def __init__(self, abstract_dtype):
        self.abstract_dtype = abstract_dtype

    def __getitem__(self, shapes: str):
        return self.abstract_dtype[Tensor, shapes]


Shaped = TorchTyping(Shaped)
Float = TorchTyping(Float)
Int = TorchTyping(Int)
Bool = TorchTyping(Bool)
