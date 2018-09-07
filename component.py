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
from collections import defaultdict
from collections.__init__ import namedtuple
from typing import Set, Tuple, Dict, DefaultDict

# Activations will very likely stay floats, but we alias that here in case we need to change it at any point
ActivationValue = float
ItemIdx = int
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
    label: ItemLabel
    activation: ActivationValue
    time_activated: int

    def __repr__(self) -> str:
        return f"<'{self.label}' ({self.activation}) @ {self.time_activated}>"


class ModelComponent(metaclass=ABCMeta):

    def __init__(self, item_labelling_dictionary: Dict):
        self.idx2label = item_labelling_dictionary
        self.label2idx = {v: k for k, v in item_labelling_dictionary.items()}

        # A node-keyed dictionaries of node ActivationRecords.
        # Stores the most recent activation of each node, if any.
        self._activation_records: DefaultDict[ItemIdx, ActivationRecord] = defaultdict(blank_node_activation_record)

        # Impulses are stored in an arrival-time-keyed dict of destination-idx-keyed dicts of cumulative activation
        # scheduled for arrival.
        # This way, when an arrival time is reached, we can .pop() a destination-idx-keyed dict of activations to
        # process.  Nice!
        # ACTUALLY we'll use a defaultdict here, so we can quickly and easily add a scheduled activation in the right
        # place without verbose checks.
        self._scheduled_activations: DefaultDict[int, DefaultDict[ItemIdx, ActivationValue]] = defaultdict(
            # In case the aren't any scheduled activations due to arrive at a particular time, we'll just find an empty
            # defaultdict
            lambda: defaultdict(
                # In case there aren't any scheduled activations due to arrive at a particular node, we'll just find
                # 0 activation, which allows for handy use of +=
                ActivationValue
            ))

        # Zero-indexed tick counter.
        self.clock: int = int(0)

    @abstractmethod
    def tick(self) -> Set[ItemActivatedEvent]:
        pass

    def _apply_activations(self) -> Set:
        """
        Applies scheduled all scheduled activations.
        :return:
            Set of nodes which became consciously active.
        """

        items_which_became_activated = set()
        items_which_crossed_conscious_access_threshold = set()

        if self.clock in self._scheduled_activations:

            # This should be a item-keyed dict of activation ready to arrive
            scheduled_activation: DefaultDict = self._scheduled_activations.pop(self.clock)

            if len(scheduled_activation) > 0:
                for destination_item, activation in scheduled_activation.items():
                    item_did_become_activated, item_did_cross_conscious_access_threshold = self.activate_item_with_idx(destination_item, activation)
                    if item_did_become_activated:
                        items_which_became_activated.add(destination_item)
                    if item_did_cross_conscious_access_threshold:
                        items_which_crossed_conscious_access_threshold.add(destination_item)

        return items_which_crossed_conscious_access_threshold

    def schedule_activation_of_item_with_idx(self, idx: ItemIdx, activation: ActivationValue, arrival_time: int):
        self._scheduled_activations[arrival_time][idx] += activation

    @abstractmethod
    def activation_of_item_with_idx(self, idx) -> ActivationValue:
        pass

    def activation_of_item_with_label(self, label: ItemLabel) -> ActivationValue:
        return self.activation_of_item_with_idx(self.label2idx[label])

    @abstractmethod
    def activate_item_with_idx(self, idx, activation: ActivationValue) -> Tuple[bool, bool]:
        pass

    def activate_item_with_label(self, label: ItemLabel, activation: ActivationValue) -> Tuple[bool, bool]:
        return self.activate_item_with_idx(self.label2idx[label], activation)
