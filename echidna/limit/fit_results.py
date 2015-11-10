""" Fit results module, containing ``FitResults`` class.
"""
import numpy

import copy


class FitResults(object):
    """ Base class for handling results of the fit.

    Args:
      fit_config (:class:`echidna.limit.fit.FitConfig`): Configuration
        for fit. This should be a direct copy of the
        :class:`echidna.limit.fit.FitConfig` object in
        :class:`echidna.limit.fit.Fit`.
      name (str, optional): Name of this :class:`FitResults` class
        instance. If no name is supplied, name from fit_results will be
        taken and appended with "_results".

    Attributes:
      _fit_config (:class:`echidna.limit.fit.FitConfig`): Configuration
        for fit. This should be a direct copy of the
        :class:`echidna.limit.fit.FitConfig` object in
        :class:`echidna.limit.fit.Fit`.
      _name (string): Name of this :class:`FitResults` class instance.
      _data (:class:`numpy.ndarray`): Array of values of the test
        statistic calculated during the fit.

    Examples:

        >>> fit_results = FitResults(fitter.get_config())
    """
    def __init__(self, fit_config, name=None):
        self._fit_config = fit_config
        if name is None:
            name = fit_config.get_name() + "_results"
        self._name = name
        self._data = numpy.zeros(self.get_shape())

    def get_fit_config(self):
        """
        Returns:
          (:class:`echidna.limit.fit.FitConfig`): Configuration of fit.
        """
        return self._fit_config

    def get_fit_data(self):
        """
        Returns:
          (:class:`numpy.ndarray`): Array store in :attr:`_data`.
            Values of the test statistic calculated during the fit.
        """
        return self._data

    def get_shape(self):
        """ Determine the shape of the grid of parameter values.

        Returns:
          tuple: Shape of parameter grid.
        """
        shape = []
        for par in self._fit_config.get_pars():
            parameter = self._fit_config.get_par(par)
            shape.append(len(parameter.get_values()))
        return tuple(shape)

    def get_summary(self):
        """
        """
        fit_results = {}
        for par in self._fit_config.get_pars():
            parameter = self._fit_config.get_par(par)
            fit_results[par] = parameter.get_best_fit()
        return fit_results