from abc import ABC, abstractmethod


class DiffusionModelBase(ABC):
    """
    Abstract base class for diffusion models.
    """

    @abstractmethod
    def q_sample(self):
        """Abstract method for the forward process (q_sample)."""
        raise NotImplementedError("q_sample method not implemented.")

    @abstractmethod
    def p_sample(self):
        """Abstract method for the reverse process (p_sample)."""
        raise NotImplementedError("p_sample method not implemented.")
