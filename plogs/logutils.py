# -*- coding: utf8 -*-

# File: logutils.py
# Author: Doug Rudolph
# Created: November 19, 2018
from enum import Enum

import os
import sys

class Levels(Enum):
    INFO = 1
    SUCCESS = 2
    WARNING = 3
    CRITICAL = 4
    ERROR = 5
    STATUS = 6


class Colors:

    _color_map = {
        Levels.INFO: '\033[94m',
        Levels.STATUS: '\033[1m',
        Levels.SUCCESS: '\033[92m',
        Levels.WARNING: '\033[93m',
        Levels.ERROR: '\033[91m',
        Levels.CRITICAL: '\u001b[41;1m',
        'end': '\033[0m',
    }

    @staticmethod
    def color(log_lvl):
        return Colors._color_map[log_lvl]


class LogLevel:

    _level_map = {
        Levels.INFO: 'INFO',
        Levels.STATUS: 'STATUS',
        Levels.SUCCESS: 'SUCCESS',
        Levels.WARNING: 'WARNING',
        Levels.ERROR: 'ERROR',
        Levels.CRITICAL: 'CRITICAL',
    }

    @staticmethod
    def level(log_lvl):
        return LogLevel._level_map[log_lvl]


def check_config(pretty, show_levels, show_time, to_file, file_location, filename):

    # if trying to write to file, the following cases must hold true
    if to_file:
       # file_location and filename must be strings
        try:
            assert isinstance(file_location, str)
            assert isinstance(filename, str)
            assert len(file_location) > 0
            assert len(filename) > 0
        except AssertionError as err:
            print('`filename` or `file_location` must be a non-empty string')
            raise

        # file_location must exist
        if not os.path.exists(file_location):
            try:
                os.mkdir(file_location)
            except Exception as err:
                print('Permission Denied: cannot write to `file_location: ', file_location)
                print('Must run program as root user')
                sys.exit(0)
        else:
            try:
                assert os.path.exists(file_location)
            except AssertionError as err:
                print('`file_location`:', file_location,' - is broken')

    # config variables must be type `bool`
    try:
        assert isinstance(pretty, bool)
        assert isinstance(show_levels, bool)
        assert isinstance(show_time, bool)
        assert isinstance(to_file, bool)
    except AssertionError as err:
        print(
            'One of the following config variables is not of type `bool`:\n ',
            'pretty: ', pretty,
            'show_levels: ', show_levels,
            'show_time: ', show_time,
            'to_file: ', to_file,
        )
        raise

    # return updated vars
    return pretty, show_levels, show_time, to_file, file_location, filename
