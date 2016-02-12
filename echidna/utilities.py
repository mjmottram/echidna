""" Utilities module to include functions that may be useful throughout
echidna.
"""
import numpy

import echidna
import echidna.output as output

import time
import logging
from colorlog import ColoredFormatter
import inspect
import os
import socket


class Timer:
    """ Useful timer class to show time elapsed performing a code block.

    Examples:
      >>> With Timer() as t:
      ...     # block of code to time
      >>> print ('Code executed in %.03f sec.' % t._interval)
    """
    def __enter__(self):
        """
        Attributes:
          _start (float): start time of code block

        Returns:
          :class:`Timer`: class instance
        """
        self._start = time.clock()
        return self

    def __exit__(self, *args):
        """
        Attributes:
          _end (float): end time of code block
          _interval (float): time elapsed during code block
        """
        self._end = time.clock()
        self._interval = self._end - self._start


def get_array_errors(array, lin_err=0.01, frac_err=None,
                     log=False, log10=False):
    shape = array.shape
    array = array.ravel()
    errors = numpy.zeros(array.shape)
    for index, value in enumerate(array):
        if log:
            value = numpy.log(value)
        elif log10:
            value = numpy.log10(value)
        if lin_err:
            error = value + lin_err
        elif frac_err:
            error = value * frac_err
        else:
            raise ValueError("Must provide either lin_err or frac_err")
        if log:
            error = numpy.exp(error)
        elif log10:
            error = numpy.power(10., error)
        errors[index] = error
    errors = errors - array
    errors.reshape(shape)
    return errors


class DispatchingFormatter:
    """Dispatch formatter for logger and it's sub logger.

    Adapted from: https://stackoverflow.com/questions/1741972/
      how-to-use-different-formatters-with-the-same-logging-handler-in-python.

    Args:
      formatters (dict): Dictionary of named formatters
      default_formatter (:class:`logging.Formatter`): Default formatter
        to use when no named formatter matches.
    """
    def __init__(self, formatters, default_formatter):
        self._formatters = formatters
        self._default_formatter = default_formatter

    def format(self, record):
        """ Format each record accordingly

        Args:
          record (:class:`logging.LogRecord`): Log record to format.

        Returns:
          :class:`logging.Formatter`: Appropriate Formatter.
        """
        # Search from record's logger up to it's parents:
        logger = logging.getLogger(record.name)
        while logger:
            # Check if suitable formatter for current logger exists:
            if logger.name in self._formatters:
                formatter = self._formatters[logger.name]
                break
            else:
                logger = logger.parent
        else:
            # If no formatter found, just use default:
            formatter = self._default_formatter
        return formatter.format(record)


logger_set = False  # Create flag for logger set/un-set


def start_logging(short_name=False, script_name=True):
    """ Function to initialise logging output.

    Adapted from:
    https://docs.python.org/2/howto/logging-cookbook.html#
      logging-to-multiple-destinations

    Args:
      short_name (bool, optional): If true the log name is always
        'echidna.log'
      script_name (string, optional): Name of script to add to log
        filename

    Returns:
      :class:`logging.Logger`: Logger to use in script.
    """
    global logger_set  # use module-level variable

    # Get current module
    current_module = inspect.getouterframes(inspect.currentframe())[1][1]
    current_module = current_module[
        current_module.rfind("/")+1:current_module.rfind(".")]

    if logger_set:  # Don't need to create
        logging.info("Starting script: %s.py" % current_module)
        return logging.getLogger(name=current_module)

    # Set up logging to file
    path = output.__default_save_path__ + "/"
    filename = "echidna"
    if script_name:
        filename += "." + current_module
    if not short_name:
        filename += ".%s.%d" % (socket.gethostname(), os.getpid())
    filename += ".log"

    # Format filename.py:XX [function_name()] LEVEL: message
    FORMAT = ("%(filename)s:%(lineno)s [%(funcName)s()] "
              "%(levelname)-8s: %(message)s")
    logging.basicConfig(
        level=logging.DEBUG,  # Include all logging levels in file
        format=FORMAT,
        filename=path+filename,
        filemode='w')

    # Define Handler which writes INFO messages or higher to the sys.stdout
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # Set a format which is simpler for console use - and in COLOUR!
    FORMAT = "%(reset)s%(log_color)s%(name)-20s: %(message)s"
    color_formatter = ColoredFormatter(
        FORMAT,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "white,bg_black",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "white,bg_red"})
    extra_formatter = ColoredFormatter(
        "%(reset)s%(log_color)s%(message)s",
        log_colors={
            "DEBUG": "white,bg_black",
            "INFO": "white,bg_black",
            "WARNING": "white,bg_black",
            "ERROR": "white,bg_black",
            "CRITICAL": "white,bg_black"})

    # tell handler when to use each formatter
    console.setFormatter(DispatchingFormatter(
        formatters={"extra": extra_formatter},
        default_formatter=color_formatter))
    # add the handler to the root logger
    logging.getLogger().addHandler(console)

    # Start logging
    logging.info("echidna-v%s" % echidna.__version__)
    logging.info("Saving logfile to %s" % path + filename)
    logging.info("Starting script: %s.py" % current_module)
    logging.getLogger("extra").info("Use the 'extra' logger at any time "
                                    "to add extra information.")

    # Mark logger as set
    logger_set = True

    return logging.getLogger(name=current_module)
