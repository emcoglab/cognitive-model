"""
===========================
Common features of components of the cognitive model.
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2019
---------------------------
"""

import json
from collections import namedtuple
from typing import Dict

from model.graph import Node

ActivationValue = float
ItemIdx = Node
ItemLabel = str


class ActivationRecord(namedtuple('ActivationRecord', ['activation',
                                                       'time_activated'])):
    """
    ActivationRecord stores a historical node activation.

    It is immutable, so must be used in conjunction with TSA.node_decay_function in order to determine the
    current activation of a node.

    `activation` stores the total accumulated level of activation at this node when it was activated.
    `time_activated` stores the clock value when the node was last activated, or -1 if it has never been activated.

    Don't thoughtlessly change this class as it probably needs to remain a small namedtuple for performance reasons.
    """
    __slots__ = ()


class ItemActivatedEvent(namedtuple('ItemActivatedEvent', ['activation',
                                                           'time_activated',
                                                           'label'])):
    """
    A node activation event.
    Used to pass out of TSA.tick().
    """
    # TODO: this is basically the same as ActivationRecord, and could probably be removed in favour of it.
    label: ItemLabel
    activation: ActivationValue
    time_activated: int

    def __repr__(self) -> str:
        return f"<'{self.label}' ({self.activation}) @ {self.time_activated}>"


def _load_labels(nodelabel_path: str) -> Dict[ItemIdx, ItemLabel]:
    with open(nodelabel_path, mode="r", encoding="utf-8") as nrd_file:
        node_relabelling_dictionary_json = json.load(nrd_file)
    # TODO: this isn't a great way to do this
    node_labelling_dictionary = dict()
    for k, v in node_relabelling_dictionary_json.items():
        node_labelling_dictionary[ItemIdx(k)] = v
    return node_labelling_dictionary


def blank_node_activation_record() -> ActivationRecord:
    """A record for an unactivated node."""
    return ActivationRecord(activation=0, time_activated=-1)