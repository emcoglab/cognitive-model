"""
===========================
Utility functions for files.
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

from pandas import read_csv, DataFrame, Series
from pandas.errors import EmptyDataError


def touch(file_path: str) -> None:
    """Touches a file on disk."""
    with open(file_path, 'a'):
        os.utime(file_path, times=None)


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


def comment_line_from_str(message: str) -> str:
    """Converts a string into a commented line (with trailing newline)."""
    return f"# {message}\n"
