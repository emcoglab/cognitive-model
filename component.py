"""
===========================
Model component shared code.
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2018
---------------------------
"""
from abc import ABCMeta, abstractmethod
from collections.__init__ import namedtuple
from typing import Set, Tuple, Dict


# Activations will very likely stay floats, but we alias that here in case we need to change it at any point
ActivationValue = float
Label = str


class ActivationRecord(namedtuple('ActivationRecord', ['activation',
                                                       'time_activated'])):
    """
    ActivationRecord stores a historical node activation.

    It is immutable, so must be used in conjunction with TSA.node_decay_function in order to determine the
    current activation of a node.

    `activation` stores the total accumulated level of activation at this node when it was activated.
    `time_activated` stores the clock value when the node was last activated, or -1 if it has never been activated.

    Don't thoughtlessly change this class as it probably needs to remain a namedtuple for performance reasons.
    """
    __slots__ = ()


def blank_node_activation_record() -> ActivationRecord:
    """A record for an unactivated node."""
    return ActivationRecord(activation=0, time_activated=-1)


class ItemActivatedEvent(namedtuple('ItemActivatedEvent', ['activation',
                                                           'time_activated',
                                                           'label'])):
    """
    A node activation event.
    Used to pass out of TSA.tick().
    """
    label: Label
    activation: ActivationValue
    time_activated: int

    def __repr__(self) -> str:
        return f"<'{self.label}' ({self.activation}) @ {self.time_activated}>"


class ModelComponent(metaclass=ABCMeta):

    def __init__(self, item_labelling_dictionary: Dict):
        self.idx2label = item_labelling_dictionary
        self.label2idx = {v: k for k, v in item_labelling_dictionary.items()}

        # Zero-indexed tick counter.
        self.clock: int = int(0)

    @abstractmethod
    def tick(self) -> Set[ItemActivatedEvent]:
        pass

    @abstractmethod
    def activate_item_with_idx(self, idx, activation: ActivationValue) -> Tuple[bool, bool]:
        pass

    def activate_item_with_label(self, label: Label, activation: ActivationValue) -> Tuple[bool, bool]:
        return self.activate_item_with_idx(self.label2idx[label], activation)
