"""
B-ASIC SVD Operations Module.

Contains custom operations used by the singular value decomposition (SVD)
signal flow graph generators.
"""

from b_asic.graph_component import Name, TypeName
from b_asic.operation import AbstractOperation
from b_asic.port import SignalSourceProvider


class Sign(AbstractOperation):
    r"""
    Sign operation.

    Gives the sign of its input, as a scalar :math:`\pm 1`.

    .. math:: y = 1 \text{ if } x \geq 0 \text{ else } -1

    Parameters
    ----------
    src0 : :class:`~b_asic.port.SignalSourceProvider`, optional
        The signal to compute the sign of.
    name : Name, optional
        Operation name.
    """

    def __init__(
        self,
        src0: SignalSourceProvider | None = None,
        name: Name = Name(""),
    ) -> None:
        """Construct a Sign operation."""
        super().__init__(input_count=1, output_count=1, name=name, input_sources=[src0])

    @classmethod
    def type_name(cls) -> TypeName:
        return TypeName("sign")

    def evaluate(self, a, data_type=None):
        return 1.0 if a >= 0 else -1.0


__all__ = ["Sign"]
