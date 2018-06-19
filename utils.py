"""
===========================
Useful utility functions.
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

import os
from os import path

from pandas import Series, read_csv, DataFrame
from pandas.errors import EmptyDataError


def touch(fname, times=None):
    """Touches a file on disk."""
    with open(fname, 'a'):
        os.utime(fname, times)


def add_column_to_csv(column, column_name: str, csv_path: str, index_col=None):
    """
    Adds a new column to an existing csv file on disk.
    ¡ OVERWRITES EXISTING FILE !
    Creates new file if none exist.
    :param column: A List or Series of data
    :param column_name: The name of the column to add
    :param csv_path: The path to the existing CSV, which will be overwritten.
    :param index_col: (Optional.) If specified and not None, will be used as the name of the index column
    """
    if not path.isfile(csv_path):
        # TODO: this won't work if index_col isn't None
        touch(csv_path)
    try:
        existing_data = read_csv(csv_path, index_col=index_col, header=0)
    except EmptyDataError:
        existing_data = DataFrame()
    if column_name in existing_data.columns.values:
        raise Exception(f"{column_name} column already exists in csv")
    existing_data[column_name] = Series(data=column, index=None)
    existing_data.to_csv(csv_path, index=(index_col is not None), header=True)


def partition(iterable, predicate):
    """
    Separates the an iterable into two sub-iterables; those which satisfy predicate and those which don't.
    Thanks to https://stackoverflow.com/a/4578605/2883198 and https://stackoverflow.com/questions/949098/python-split-a-list-based-on-a-condition#comment24295861_12135169.
    """
    trues = []
    falses = []
    for item in iterable:
        trues.append(item) if predicate(item) else falses.append(item)
    return trues, falses


def set_partition(iterable, predicate):
    trues = set()
    falses = set()
    for item in iterable:
        trues.add(item) if predicate(item) else falses.add(item)
    return trues, falses
