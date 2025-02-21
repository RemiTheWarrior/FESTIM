import numpy as np
import ufl
from dolfinx import fem

from festim.helpers import Value
from festim.species import Species
from festim.subdomain.volume_subdomain import VolumeSubdomain


class SourceBase:
    """
    Source base class

    Args:
        value: the value of the source
        volume: the volume subdomains where the source is applied

    Attributes:
        value: the value of the source
        volume: the volume subdomains where the source is applied
    """

    value: float | int | fem.Constant | np.ndarray | fem.Expression | \
           ufl.core.expr.Expr | fem.Function
    volume: VolumeSubdomain

    def __init__(
            self,
            value: float | int | fem.Constant | np.ndarray | fem.Expression |
                   ufl.core.expr.Expr | fem.Function,
            volume: VolumeSubdomain
            ):

        self.value = value
        self.volume = volume

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = Value(value)

    @property
    def volume(self):
        return self._volume

    @volume.setter
    def volume(self, value):
        # check that volume is festim.VolumeSubdomain
        if not isinstance(value, VolumeSubdomain):
            raise TypeError("volume must be of type festim.VolumeSubdomain")
        self._volume = value


class ParticleSource(SourceBase):
    """
    Particle source class

    Args:
        value: the value of the source
        volume: the volume subdomains where the source is applied
        species: the species to which the source is applied

    Attributes:
        value: the value of the source
        volume: the volume subdomains where the source is applied
        species: the species to which the source is applied

    Examples:

        .. highlight:: python
        .. code-block:: python

            from festim import ParticleSource

            ParticleSource(volume=my_vol, value=1, species="H")
            ParticleSource(volume=my_vol, value=lambda x: 1 + x[0], species="H")
            ParticleSource(volume=my_vol, value=lambda t: 1 + t, species="H")
            ParticleSource(volume=my_vol, value=lambda T: 1 + T, species="H")
            ParticleSource(volume=my_vol, value=lambda x, t: 1 + x[0] + t, species="H")
    """

    species: Species

    def __init__(self, value, volume, species: Species):
        self.species = species
        super().__init__(value, volume)

    @property
    def species(self):
        return self._species

    @species.setter
    def species(self, value):
        # check that species is festim.Species or list of festim.Species
        if not isinstance(value, Species):
            raise TypeError("species must be of type festim.Species")

        self._species = value


class HeatSource(SourceBase):
    """
    Heat source class

    Args:
        value: the value of the source
        volume: the volume subdomains where the source is applied

    Attributes:
        value: the value of the source
        volume: the volume subdomains where the source is applied

    Examples:

        .. highlight:: python
        .. code-block:: python

            from festim import HeatSource

            HeatSource(volume=my_vol, value=1)
            HeatSource(volume=my_vol, value=lambda x: 1 + x[0])
            HeatSource(volume=my_vol, value=lambda t: 1 + t)
            HeatSource(volume=my_vol, value=lambda x, t: 1 + x[0] + t)
    """
    def __init__(self, value, volume):
        super().__init__(value, volume)

        if self.value.temperature_dependent:
            raise ValueError("Heat source cannot be temperature dependent")
