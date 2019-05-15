"""
===========================
Events associated with model function.
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

from model.graph_propagation import ActivationValue, ItemIdx


class ModelEvent:
    """An event associated with model activity."""
    def __init__(self, time: int):
        """
        :param time: The time at which the event occurred.
        """
        self.time: int = time


class ItemEvent(ModelEvent):
    """An event involving an item."""
    def __init__(self, time: int, item: ItemIdx):
        """
        :param item: The item being activated.
        """
        super(ItemEvent, self).__init__(time=time)
        self.item: ItemIdx = item


class ItemActivatedEvent(ItemEvent):
    """An item is activated."""
    def __init__(self, time: int, item: ItemIdx, activation: ActivationValue):
        super(ItemActivatedEvent, self).__init__(time=time, item=item)
        self.activation: ActivationValue = activation


class ItemEnteredBufferEvent(ItemEvent):
    """An item entered the working memory buffer."""
    def __init__(self, time: int, item: ItemIdx):
        super(ItemEnteredBufferEvent, self).__init__(time=time, item=item)
