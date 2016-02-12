import numpy
from scipy import interpolate

import collections
import yaml
import copy
import abc
import logging


class Parameter(object):
    """ The base class for creating parameter classes.

    Args:
      type_name (string): The type of the parameter.
      name (str): The name of this parameter
      low (float): The lower limit to float the parameter from
      high (float): The higher limit to float the parameter from
      bins (int): The number of steps between low and high values

    Attributes:
      _type (string): The type of the parameter.
      _name (str): The name of this parameter
      _low (float): The lower limit to float the parameter from
      _high (float): The higher limit to float the parameter from
      _bins (int): The number of steps between low and high values
    """

    def __init__(self, type_name, name, low, high, bins):
        """ Initialise config class
        """
        self._type = type_name
        self._name = name
        self._low = float(low)
        self._high = float(high)
        self._bins = int(bins)

    def get_bins(self):
        """ Get the number of bins.

        Returns:
          int: Number of bins for this parameter.
        """
        return self._bins

    def get_high(self):
        """ Get the high value of the parameter

        Returns:
          float: The high value of the parameter.
        """
        return self._high

    def get_low(self):
        """ Get the low value of the parameter.

        Returns:
          float: The low value the parameter.
        """
        return self._low

    def get_name(self):
        """ Get the name of the parameter.

        Returns:
          float: The name of the parameter.
        """
        return self._name

    def get_type(self):
        """ Get the type of the parameter.

        Returns:
          float: The type of the parameter.
        """
        return self._type

    def get_width(self):
        """Get the width of the binning for the parameter

        Returns:
          float: Bin width.
        """
        return (self._high - self._low) / float(self._bins)

    def to_dict(self, basic=False):
        """ Represent the properties of the parameter in a dictionary.

        .. note:: The attributes :attr:`_name`, :attr:`_type` are never
          included in the dictionary. This is because it is expected
          that the dictionary returned here will usually be used as
          part of a larger dictionary where type and/or parameter_name
          are keys.

        Returns:
          dict: Representation of the parameter in the form of a
            dictionary.
        """
        parameter_dict = {}
        parameter_dict["low"] = self._low
        parameter_dict["high"] = self._high
        parameter_dict["bins"] = self._bins
        return parameter_dict


class FitParameter(Parameter):
    """Simple data container that holds information for a fit parameter
    (i.e. a systematic to float).

    .. warning:: The sigma value can be explicitly set as None. This
    is so that you disable a penalty term for a floating parameter.
    If a parameter is being floated, but sigma is None, then no penalty
    term will be added for the parameter.

    .. note:: The :class:`FitParameter` class offers three different
    scales for constructing the array of values for the parameter::

      * **linear**: A standard linear scale is the default option. This
        creates an array of equally spaced values, starting at
        :arg:`low` and ending at :arg:`high` (*includive*). The array
        will contain :arg:`bins` values.
      * **logscale**: This creates an array of values that are equally
        spaced in log-space, but increase exponentially in linear-space,
        starting at :arg:`low` and ending at :arg:`high` (*includive*).
        The array will contain :arg:`bins` values.
      * **logscale_deviation**: This creates an array of values -
        centred around the prior - whose absolute deviations from the
        prior are equally spaced in log-space, but increase
        exponentially in linear-space. The values start at :arg:`low`
        and end at :arg:`high` (*includive*). The array will contain
        :arg:`bins` values.

    Args:
      name (str): The name of this parameter
      prior (float): The prior of the parameter
      sigma (float): The sigma of the parameter
      low (float): The lower limit to float the parameter from
      high (float): The higher limit to float the parameter from
      bins (int): The number of steps between low and high values
      dimension (string, optional): The spectral dimension to which the
        fit parameter applies.
      values (:class:`numpy.array`, optional): Array of parameter
        values to test in fit.
      best_fit (float, optional): Best-fit value calculated by fit.
      penalty_term (float, optional): Penalty term value at best fit.
      logscale (bool, optional): Flag to create an logscale array of
        values, rather than a linear array.
      base (float, optional): Base to use when creating an logscale
        array. Default is base-e.
      logscale_deviation (bool, optional): Flag to create a logscale deviation
        array of values rather than a linear or logscale array.

    Attributes:
      _prior (float): The prior of the parameter
      _sigma (float): The sigma of the parameter
      _dimension (string): The spectral dimension to which the fit
        parameter applies.
      _values (:class:`numpy.array`): Array of parameter values to
        test in fit.
      _best_fit (float): Best-fit value calculated by fit.
      _penalty_term (float): Penalty term value at best fit.
      _logscale (bool): Flag to create an logscale array of values,
        rather than a linear array.
      _base (float): Base to use when creating an logscale array.
        Default is base-e
      _logscale_deviation (bool): Flag to create a logscale deviation
        array of values rather than a linear or logscale array.
      _bin_boundaries (:class:`numpy.array`): Array of bin boundaries
        corresponding to :attr:`_values`.
    """

    def __init__(self, name, prior, sigma, low, high, bins, dimension=None,
                 values=None, current_value=None, penalty_term=None,
                 best_fit=None, logscale=None, base=numpy.e,
                 logscale_deviation=None):
        """Initialise FitParameter class
        """
        super(FitParameter, self).__init__("fit", name, low, high, bins)
        self._logger = logging.getLogger("fit_parameter")
        self._prior = float(prior)
        if sigma is None:
            self._logger.warning(
                "Setting sigma explicitly as None for %s - "
                "No penalty term will be added for this parameter!" % name)
        self._sigma = sigma
        self._dimension = dimension
        self._values = values
        self._current_value = current_value
        self._best_fit = best_fit
        self._penalty_term = penalty_term
        self._logscale = None
        self._base = None
        self._logscale_deviation = None
        self._bin_boundaries = None
        if logscale:
            self._logger.info("Setting logscale %s for parameter %s" %
                              (logscale, name))
            logging.getLogger("extra").info(" --> with base: %.4g" % base)
            if logscale_deviation is not None:
                self._logger.warning("Recieved logscale_deviation flag that "
                                     "will not have any effect")
            self._logscale = logscale
            self._base = base
        elif logscale_deviation:
            self._logger.info("Setting logscale_deviation %s for parameter %s"
                              % (logscale_deviation, name))
            self._logscale_deviation = logscale_deviation

    @abc.abstractmethod
    def apply_to(self, spectrum):
        """ Applies current value of fit parameter to spectrum.

        Args:
          spectrum (:class:`Spectra`): Spectrum to which current value
            of parameter should be applied.

        Returns:
          (:class:`Spectra`): Modified spectrum.

        Raises:
          ValueError: If :attr:`_current_value` is not set.
        """
        pass

    def check_values(self):
        """ For symmetric arrays, check that the prior is in the values.

        Raises:
          ValueError: If prior is not in the values array.
        """
        values = self.get_values()
        if not numpy.any(numpy.around(values / self._prior, 12) ==
                         numpy.around(1., 12)):
            log_text = ""
            log_text += "Values: %s\n" % str(values)
            log_text += "Prior: %.4g\n" % self._prior
            logging.getLogger("extra").warning("\n%s\n" % log_text)
            raise ValueError("Prior not in values array. "
                             "This can be achieved with an odd number "
                             "of bins and symmetric low and high values "
                             "about the prior.")

    def get_best_fit(self):
        """
        Returns:
          float: Best fit value of parameter - stored in
            :attr:`_best_fit`.

        Raises:
          ValueError: If the value of :attr:`_best_fit` has not yet
            been set.
        """
        if self._best_fit is None:
            raise ValueError("Best fit value for parameter" +
                             self._name + " has not been set")
        return self._best_fit

    def get_bin_boundaries(self):
        """ Returns an array of bin boundaries, based on the :attr:`_low`,
        :attr:`_high` and :attr:`_bins` parameters, and any flags
        (:attr:`_logscale` or :attr:`_logscale_deviation`) that have
        been applied.

        Returns:
          (:class:`numpy.array`): Array of bin_baoundaries for the
            parameter values stored in :attr:`_values`.
        """
        if self._bin_boundaries is None:  # Generate array of values
            if self._logscale:
                if self._low <= 0.:  # set low = -log(high)
                    low = -numpy.log(self._high)
                    logging.warning("Correcting fit parameter value <= 0.0")
                    logging.debug(" --> changed to %.4g (previously %.4g)" %
                                  (numpy.exp(low), self._low))
                else:
                    low = numpy.log(self._low)
                high = numpy.log(self._high)
                width = (numpy.log(high) - numpy.log(low)) / int(self._bins)
                self._bin_boundaries = numpy.logspace(
                    low - 0.5*width, high + 0.5*width,
                    num=self._bins+1, base=numpy.e)
            elif self._logscale_deviation:
                delta = self._high - self._prior
                width = numpy.log(delta + 1.) / int(self._bins / 2)
                deltas = numpy.linspace(
                    0.5 * width, numpy.log(delta + 1.) + 0.5*width,
                    num=int((self._bins + 1) / 2))
                pos = self._prior + numpy.exp(deltas) - 1.
                neg = self._prior - numpy.exp(deltas[::-1]) + 1.
                self._bin_boundaries = numpy.append(neg, pos)
            else:
                width = self.get_width()
                self._bin_boundaries = numpy.linspace(self._low + 0.5*width,
                                                      self._high + 0.5*width,
                                                      self._bins + 1)
        return self._bin_boundaries

    def get_current_value(self):
        """
        Returns:
          float: Current value of fit parameter - stored in
            :attr:`_current_value`
        """
        if self._current_value is None:
            raise ValueError("Current value not yet set " +
                             "for parameter " + self._name)
        return self._current_value

    def get_dimension(self):
        """
        Returns:
          string: Dimension to which fit parameter is applied.
        """
        return self._dimension

    def get_penalty_term(self):
        """ Gets the value of the penalty term at the best fit.

        Returns:
          float: Penalty term value of parameter at best fit - stored in
            :attr:`_penalty_term`.

        Raises:
          ValueError: If the value of :attr:`_penalty_term` has not yet
            been set.
        """
        if self._penalty_term is None:
            raise ValueError("Penalty term value for parameter" +
                             self._name + " has not been set")
        return self._penalty_term

    def get_pre_convolved(self, directory, filename):
        """ Appends the name and current value of a the :class:`FitParameter`

        .. note:: Before any calls to this function, the base directory
          should be of the form::

              ../hyphen-separated-dimensions/spectrum_name/

          and a base filename of the form ``spectrum_name``.

        .. note:: Each call to this method, then appends the name of
          the :class:`FitParamter` to the ``directory`` and its current
          value to the ``filename``. So for three :class:`FitParameters``,
          after three calls to this method, the directory should be e.g.::

              ../energy_mc-radial3_mc/Te130_0n2b/syst1/syst2/syst3/

          and the filename might be::

              Te130_0n2b_250.0_0.012_1.07

        .. note:: To construct the full path to pass to
          :funct:`echidna.output.store.load`, the ``directory`` and
          ``filename`` returned by the last call to this method,
          should be added together, and appended with ``".hdf5"``.

              path = director + filename + ".hdf5"

        Args:
          directory (string): Current or base directory containing
            pre-convolved :class:`Spectra` object
          name (string): Current or base name of :class:`Spectra`
            object

        Returns:
          string: Directory containing pre-convolved :class:`Spectra`,
            appended with name of this :class:`FitParameter`
          string: Name of pre-convolved :class:`Spectra`, appended with
            current value of this :class:`FitParameter`

        Raises:
          ValueError: If :attr:`_current_value` is not set.
        """
        if self._current_value is None:
            raise ValueError("Current value of fit parameter %s "
                             "has not been set" % self._name)
        directory += "_%s/" % self._name
        value_string = "%f" % self._current_value
        # Strip leading/trailling zeros in filename
        filename += ("_%s" % value_string.strip("0"))
        return directory, filename

    def get_prior(self):
        """
        Returns:
          float: Prior value of fit parameter - stored in
            :attr:`_prior`
        """
        return self._prior

    def get_sigma(self):
        """
        Returns:
          float: Sigma of fit parameter - stored in :attr:`_sigma`
        """
        return self._sigma

    def get_values(self):
        """ Returns an array of values, based on the :attr:`_low`,
        :attr:`_high` and :attr:`_bins` parameters, and any flags
        (:attr:`_logscale` or :attr:`_logscale_deviation`) that have
        been applied.

        .. warning:: Calling this method with the :attr:`logscale_deviation`
        flag enabled, may alter the value of :attr:`_low`, as this
        scale must be symmetric about the prior.

        Returns:
          (:class:`numpy.array`): Array of parameter values to test in
            fit. Stored in :attr:`_values`.
        """
        if self._values is None:  # Generate array of values
            if self._logscale:
                # Create an array that is equally spaced in log-space
                self._logger.info("Creating logscale array of values "
                                  "for parameter %s" % self._name)
                if self._low <= 0.:  # set low = -log(high)
                    low = -numpy.log(self._high)
                    logging.warning("Correcting fit parameter value <= 0.0")
                    logging.debug(" --> changed to %.4g (previously %.4g)" %
                                  (numpy.exp(low), self._low))
                else:
                    low = numpy.log(self._low)
                high = numpy.log(self._high)
                self._values = numpy.logspace(low, high, num=self._bins,
                                              base=numpy.e)
            elif self._logscale_deviation:
                # Create an array with the prior as the central value, and
                # then absolute deviations from the prior that are
                # linearly spaced in log-space.
                self._logger.info("Creating logscale_deviation array of "
                                  "values for parameter %s" % self._name)
                delta = self._high - self._prior
                deltas = numpy.linspace(0., numpy.log(delta + 1.),
                                        num=(self._bins + 1.) / 2.)
                pos = self._prior + numpy.exp(deltas[1:]) - 1.
                neg = self._prior - numpy.exp(deltas[::-1]) + 1.
                self._values = numpy.append(neg, pos)
                if not numpy.allclose(self._values[0], self._low):
                    low = self._values[0]
                    self._logger.warning(
                        "Changing value of attr _low, from %.4g to %.4g, "
                        "for parameter %s" % (self._low, low, self._name))
            else:  # Create a normal linear array
                self._logger.info("Creating linear array of values "
                                  "for parameter %s" % self._name)
                self._values = numpy.linspace(self._low,
                                              self._high, self._bins)
        return self._values

    def get_value_at(self, index):
        """ Access the parameter value at a given index in the array.

        Args:
          index (int): Index of parameter value requested.

        Returns:
          float: Parameter value at the given index.
        """
        return self.get_values()[index]

    def get_value_index(self, value):
        """ Get the index corresponding to a given parameter value.

        Args:
          value (float): Parameter value for which to get corresponding
            index.

        Returns:
          int: Index of corresponding to the given parameter value.

        .. warning:: If there are multiple occurences of ``value`` in
          the array of parameter values, only the index of the first
          occurence will be returned.
        """
        indices = numpy.where(self.get_values() == value)[0]
        if len(indices) == 0:
            raise ValueError("No value %.2g found in parameter values " +
                             "for parameter %s." % (value, self._name))
        return int(indices[0])

    def set_best_fit(self, best_fit):
        """ Set value for :attr:`_best_fit`.

        Args:
          best_fit (float): Best fit value for parameter
        """
        self._best_fit = best_fit

    def set_current_value(self, value):
        """ Set value for :attr:`_current_value`.

        Args:
          value (float): Current value of fit parameter
        """
        self._current_value = value

    def set_par(self, **kwargs):
        """Set a fitting parameter's values after initialisation.

        Args:
          kwargs (dict): keyword arguments

        .. note::

          Keyword arguments include:

            * prior (float): Value to set the prior to of the parameter
            * sigma (float): Value to set the sigma to of the parameter
            * low (float): Value to set the lower limit to of the parameter
            * high (float): Value to set the higher limit to of the parameter
            * bins (float): Value to set the size of the bins between low and
              high of the parameter
            * logscale (bool): Flag to create an logscale array of
              values, rather than a linear array.
            * base (float): Base to use when creating an logscale array.

        Raises:
          TypeError: Unknown variable type passed as a kwarg.
        """
        for kw in kwargs:
            if kw == "prior":
                self._prior = float(kwargs[kw])
            elif kw == "sigma":
                if kwargs[kw] is None:
                    self._logger.warning("Setting sigma explicitly as None - "
                                         "No penalty term will be applied")
                self._sigma = kwargs[kw]
            elif kw == "low":
                self._low = float(kwargs[kw])
            elif kw == "high":
                self._high = float(kwargs[kw])
            elif kw == "bins":
                self._bins = float(kwargs[kw])
            elif kw == "logscale":
                self._logscale = bool(kwargs[kw])
            elif kw == "base":
                self._base = float(kwargs[kw])
            elif kw == "logscale_deviation":
                self._logscale_deviation = bool(kwargs[kw])
            elif kw == "dimension":
                self._dimension = str(kwargs[kw])
            else:
                raise TypeError("Unhandled parameter name / type %s" % kw)
        self._values = None

    def set_penalty_term(self, penalty_term):
        """ Set value for :attr:`_penalty_term`.

        Args:
          penalty_term (float): Value for penalty term of parameter at
            best fit.
        """
        self._penalty_term = penalty_term

    def to_dict(self, basic=False):
        """ Represent the properties of the parameter in a dictionary.

        Args:
          basic (bool, optional): If True, only the basic properties:
            prior, sigma, low, high and bins are included.

        .. note:: The attributes :attr:`_name`, :attr:`_dimension`,
          :attr:`_values` and :attr:`_logger` are never included in
          the dictionary. For the first two this is because it is
          expected that the dictionary returned here will usually be
          used as part of a larger dictionary where dimension and
          parameter_name are keys. The :attr:`values` attribute is not
          included because this is a lrge numpy array. The logger is
          not included as this is for internal use only.

        Returns:
          dict: Representation of the parameter in the form of a
            dictionary.
        """
        parameter_dict = {}
        # Add basic attributes
        parameter_dict["prior"] = self._prior
        parameter_dict["sigma"] = self._sigma
        parameter_dict["low"] = self._low
        parameter_dict["high"] = self._high
        parameter_dict["bins"] = self._bins
        parameter_dict["logscale"] = self._logscale
        parameter_dict["base"] = self._base
        parameter_dict["logscale_deviation"] = self._logscale_deviation
        if basic:
            return parameter_dict
        # Add non-basic attributes
        parameter_dict["current_value"] = self._current_value
        parameter_dict["best_fit"] = self._best_fit
        parameter_dict["penalty_term"] = self._best_fit
        return parameter_dict


class RateParameter(FitParameter):
    """ Data container that holds information for a rate parameter that
    is included in the fit.

    Args:
      name (str): The name of this parameter
      prior (float): The prior of the parameter
      sigma (float): The sigma of the parameter
      low (float): The lower limit to float the parameter from
      high (float): The higher limit to float the parameter from
      bins (int): The number of steps between low and high values
      logscale (bool, optional): Flag to create an logscale array of
        values, rather than a linear array.
      base (float, optional): Base to use when creating an logscale array.
      kwargs (dict): Other keyword arguments to pass to
        :class:`FitParameter`

    Attributes:
      _logscale (bool): Flag to create an logscale array of values,
        rather than a linear array.
      _base (float): Base to use when creating an logscale array.
    """
    def __init__(self, name, prior, sigma, low, high,
                 bins, logscale=None, base=numpy.e,
                 logscale_deviation=None, **kwargs):
        super(RateParameter, self).__init__(
            name, prior, sigma, low, high, bins, logscale=logscale,
            base=base, logscale_deviation=logscale_deviation, **kwargs)

    def apply_to(self, spectrum):
        """ Scales spectrum to current value of rate parameter.

        Args:
          spectrum (:class:`Spectra`): Spectrum which should be scaled
            to current rate value.

        Returns:
          (:class:`Spectra`): Scaled spectrum.

        Raises:
          ValueError: If :attr:`_current_value` is not set.
        """
        if self._current_value is None:
            raise ValueError("Current value of rate parameter %s "
                             "has not been set" % self._name)
        spectrum.scale(self._current_value)
        return spectrum


class ResolutionParameter(FitParameter):
    """ Data container that holds information for a resulution parameter
    that is included in the fit.

    Args:
      name (str): The name of this parameter
      prior (float): The prior of the parameter
      sigma (float): The sigma of the parameter
      low (float): The lower limit to float the parameter from
      high (float): The higher limit to float the parameter from
      bins (int): The number of steps between low and high values
      dimension (string): The spectral dimension to which the
        resolution parameter applies.
      kwargs (dict): Other keyword arguments to pass to
        :class:`FitParameter`
    """

    def __init__(self, name, prior, sigma, low,
                 high, bins, dimension, **kwargs):
        super(ResolutionParameter, self).__init__(
            name, prior, sigma, low, high, bins, dimension, **kwargs)

    def apply_to(self, spectrum):
        """ Smears spectrum to current value of resolution.

        Args:
          spectrum (:class:`Spectra`): Spectrum which should be smeared.

        Returns:
          (:class:`Spectra`): Smeared spectrum.

        Raises:
          ValueError: If :attr:`_current_value` is not set.
        """
        if self._current_value is None:
            raise ValueError("Current value of rate parameter %s "
                             "has not been set" % self._name)
        NotImplementedError("ResolutionParameter.apply_to not yet implemented")


class ScaleParameter(FitParameter):
    """ Data container that holds information for a scale parameter
    that is included in the fit.

    Args:
      name (str): The name of this parameter
      prior (float): The prior of the parameter
      sigma (float): The sigma of the parameter
      low (float): The lower limit to float the parameter from
      high (float): The higher limit to float the parameter from
      bins (int): The number of steps between low and high values
      dimension (string): The spectral dimension to which the scale
        parameter applies.
      kwargs (dict): Other keyword arguments to pass to
        :class:`FitParameter`
    """

    def __init__(self, name, prior, sigma, low,
                 high, bins, dimension, **kwargs):
        super(ScaleParameter, self).__init__(
            name, prior, sigma, low, high, bins, dimension, **kwargs)

    def apply_to(self, spectrum):
        """ Convolves spectrum with current value of scale parameter.

        Args:
          spectrum (:class:`Spectra`): Spectrum to be convolved.

        Returns:
          (:class:`Spectra`): Convolved spectrum.

        Raises:
          ValueError: If :attr:`_current_value` is not set.
        """
        if self._current_value is None:
            raise ValueError("Current value of scale parameter %s "
                             "has not been set" % self._name)
        NotImplementedError("ScaleParameter.apply_to not yet implemented")


class ShiftParameter(FitParameter):
    """ Data container that holds information for a shift parameter
    that is included in the fit.

    Args:
      name (str): The name of this parameter
      prior (float): The prior of the parameter
      sigma (float): The sigma of the parameter
      low (float): The lower limit to float the parameter from
      high (float): The higher limit to float the parameter from
      bins (int): The number of steps between low and high values
      dimension (string): The spectral dimension to which the shift
        parameter applies.
      kwargs (dict): Other keyword arguments to pass to
        :class:`FitParameter`
   """

    def __init__(self, name, prior, sigma, low,
                 high, bins, dimension, **kwargs):
        super(ShiftParameter, self).__init__(
            name, prior, sigma, low, high, bins, dimension, **kwargs)

    def apply_to(self, spectrum):
        """ Convolves spectrum with current value of shift parameter.

        Args:
          spectrum (:class:`Spectra`): Spectrum to be convolved.

        Returns:
          (:class:`Spectra`): Convolved spectrum.

        Raises:
          ValueError: If :attr:`_current_value` is not set.
        """
        if self._current_value is None:
            raise ValueError("Current value of shift parameter %s "
                             "has not been set" % self._name)
        NotImplementedError("ShiftParameter.apply_to not yet implemented")


class SpectraParameter(Parameter):
    """Simple data container that holds information for a Spectra parameter
    (i.e. axis of the spectrum).

    Args:
      name (str): The name of this parameter
      low (float): The lower limit of this parameter
      high (float): The upper limit of this parameter
      bins (int): The number of bins for this parameter
    """

    def __init__(self, name, low, high, bins):
        """Initialise SpectraParameter class
        """
        super(SpectraParameter, self).__init__("spectra", name, low, high,
                                               bins)

    def get_bin(self, x):
        """ Gets the bin index which contains value x.

        Args:
          x (float): Value you wish to find the bin index for.

        Raises:
          ValueError: If x is less than parameter lower bounds
          ValueError: If x is more than parameter upper bounds

        Returns:
          int: Bin index
        """
        if x < self._low:
            raise ValueError("%s is below parameter lower bound %s"
                             % (x, self._low))
        if x > self._high:
            raise ValueError("%s is above parameter upper bound %s"
                             % (x, self._high))
        return int((x - self._low) / self.get_width())

    def get_bin_boundaries(self):
        """ Returns the bin boundaries for the parameter

        Returns:
          :class:`numpy.ndarray`: Bin boundaries for the parameter.
        """
        return numpy.linspace(self._low, self._high, self._bins+1)

    def get_bin_centre(self, bin):
        """ Calculates the central value of a given bin

        Args:
          bin (int): Bin number.

        Raises:
          TypeError: If bin is not int
          ValueError: If bin is less than zero
          ValueError: If bin is greater than the number of bins - 1

        Returns:
          float: value of bin centre
        """
        if type(bin) != int and type(bin) != numpy.int64:
            raise TypeError("Must pass an integer value")
        if bin < 0:
            raise ValueError("Bin number (%s) must be zero or positive" % bin)
        if bin > self._bins - 1:
            raise ValueError("Bin number (%s) is out of range. Max = %s"
                             % (bin, self._bins))
        return self._low + (bin + 0.5)*self.get_width()

    def get_bin_centres(self):
        """ Returns the bin centres of the parameter

        Returns:
          :class:`numpy.ndarray`: Bin centres of parameter.
        """
        return numpy.arange(self._low+self.get_width()*0.5,
                            self._high,
                            self.get_width())

    def get_unit(self):
        """Get the default unit for a given parameter

        Raises:
          Exception: Unknown parameter.

        Returns:
          string: Unit of the parameter
        """
        if self._name.split('_')[0] == "energy":
            return "MeV"
        if self._name.split('_')[0] == "radial":
            return "mm"

    def round(self, x):
        """ Round the value to nearest bin edge

        Args:
          x (float): Value to round.

        Returns:
          float: The value of the closest bin edge to x
        """
        return round(x/self.get_width())*self.get_width()

    def set_par(self, **kwargs):
        """Set a limit / binning parameter after initialisation.

        Args:
          kwargs (dict): keyword arguments

        .. note::

          Keyword arguments include:

            * low (float): Value to set the lower limit to of the parameter
            * high (float): Value to set the higher limit to of the parameter
            * bins (int): Value to set the number of bins of the parameter

        Raises:
          TypeError: Unknown variable type passed as a kwarg.
        """
        for kw in kwargs:
            if kw == "low":
                self._low = float(kwargs[kw])
            elif kw == "high":
                self._high = float(kwargs[kw])
            elif kw == "bins":
                self._bins = int(kwargs[kw])
            else:
                raise TypeError("Unhandled parameter name / type %s" % kw)


class Config(object):
    """ The base class for creating config classes.

    Args:
      name (string): The name of the config.

    Attributes:
      _name (string): The name of the config.
      _type (string): The type of the config, this affects it's
        parameter types
      _parameters (:class:`collections.OrderedDict`): Dictionary of
        parameters.
    """

    def __init__(self, name, parameters):
        """ Initialise config class
        """
        self._name = name
        self._type = "general"
        self._parameters = parameters

    def add_par(self, par):
        """ Add parameter to the config.

        Args:
          par (:class:`echidna.core.spectra.Parameter`): The parameter you want
            to add.
        """
        self._parameters[par._name] = par

    @abc.abstractmethod
    def dump(self):
        """ Abstract base class method to override.

        Dumps the config to a config dictionary, containing all
        parameters. The dictionary has the form specified in the
        :meth:`Config.load` method.

        Returns:
          dict: Dictionary containing all the information on the
            parameters.
        """
        raise NotImplementedError("The dump method can only be used "
                                  "when overriden in a derived class.")

    @abc.abstractmethod
    def dump_to_file(self, path="", filename=None):
        """ Abstract base class method to override.

        Write config to YAML file.

        Args:
          path (string, optional): Location to save yaml file to,
            default is the current directory.
          filename (string, optional): Filename for yaml file. If no
            filename is supplied, the default is "spectra_config.yml".
            If a blank filename "" is given the config's name is used.
        """
        raise NotImplementedError("The dump_to_file method can only be used "
                                  "when overriden in a derived class.")

    def get_index(self, parameter):
        """Return the index of a parameter within the existing set

        Args:
          parameter (string): Name of the parameter.

        Raises:
          IndexError: parameter is not in the config.

        Returns:
          int: Index of the parameter
        """
        for i, p in enumerate(self.get_pars()):
            if p == parameter:
                return i
        raise IndexError("Unknown parameter %s" % parameter)

    def get_name(self):
        """
        Returns:
          string: Name of :class:`Config` class instance - stored in
            :attr:`_name`.
        """
        return self._name

    def get_par(self, name):
        """Get a named FitParameter.

        Args:
          name (string): Name of the parameter.

        Returns:
          :class:`echidna.core.spectra.Parameter`: Named parameter.
        """
        return self._parameters[name]

    def get_par_by_index(self, index):
        """ Get parameter corresponding to given index

        Args:
          index (int): Index of parameter.

        Returns:
          :class:`echidna.core.spectra.Parameter`: Corresponding
            parameter.
        """
        name = self.get_pars()[index]
        return self.get_par(name)

    def get_pars(self):
        """Get list of all parameter names in the config.

        Returns:
          list: List of parameter names
        """
        return self._parameters.keys()

    def get_shape(self):
        """ Get the shape of the parameter space.

        Returns:
          tuple: A tuple constructed of the number of bins for each
            parameter in the config - this can be thought of as the
            full shape of the parameter space, whether it is the shape
            of the parameter space for the fit, or the shape of the
            spectral dimensions.
        """
        return tuple([self.get_par(par).get_bins() for par in self.get_pars()])

    def get_type(self):
        """
        Returns:
          string: Type of :class:`Config` class instance - stored in
            :attr:`_name`.
        """
        return self._name

    @classmethod
    @abc.abstractmethod
    def load(cls, config, name="config"):
        """ Abstract base class method to override.

        Initialise Config class from a config dictionary (classmethod).

        Args:
          config (dict): Dictionary to create config out of.
          name (string, optional): Name to assign to the
            :class:`Config`. If no name is supplied the default
            'spectra_config' will be used.

        Returns:
          (:class:`Config`): A config object containing the parameters
            from the config dictionary.

        Raises:
          KeyError: If the :arg:`config` dictionary has the wrong format.

        .. warning:: Valid format is::

            {"parameters":
                "<parameter>":
                    "low": <low>
                    "high": <high>
                    "bins": <bins>}
        """
        raise NotImplementedError("The load method can only be used "
                                  "when overriden in a derived class.")

    @classmethod
    @abc.abstractmethod
    def load_from_file(cls, filename, name=None):
        """ Abstract base class method to override.

        Initialise Config class from a config file (classmethod).

        Args:
          filename (str): path to config file
          name (string, optional): Assign a name to the :class:`Config`
            created. If no name is supplied, the default is 'config'.
            If a blank string is supplied, the name of the file will
            be used.

        Returns:
          (:class:`Config`): A config object containing the parameters
            in the file.
        """
        raise NotImplementedError("The load_from_file method can only be used "
                                  "when overriden in a derived class.")


class GlobalFitConfig(Config):
    """Configuration container for floating systematics and fitting Spectra
      objects.  Able to load directly with a set list of FitParameters or
      from yaml configuration files.

    Args:
      config_name (string): Name of config
      parameters (:class:`collections.OrderedDict`): List of
        FitParameter objects
    """

    def __init__(self, config_name, parameters):
        """Initialise GlobalFitConfig class
        """
        super(GlobalFitConfig, self).__init__(config_name, parameters)
        self._type = "global_fit"

    def add_config(self, config):
        """ Add pars from a :class:`echidna.core.spectra.Config` to this
          :class:`echidna.core.spectra.GlobalFitConfig`

        Args:
          config (:class:`echidna.core.spectra.Config`): Config to be added.
        """
        if config._type == "spectra_fit":
            spectra_name = config._spectra_name
            for par_name in config.get_pars():
                name = spectra_name + "_" + par_name
                par = config.get_par(par_name)
                par._name = name
                self.add_par(par, "spectra")
        elif config._type == "global_fit":
            for par_name in config.get_pars():
                self.add_par(config.get_par(par_name), "global")
        else:
            raise ValueError("Cannot add %s-type config to a config "
                             "of type %s" % (config._type, self._type))

    def add_par(self, par, par_type):
        """ Add parameter to the global fit config.

        Args:
          par (:class:`echidna.core.spectra.FitParameter`): Parameter you want
            to add.
          par_type (string): The type of parameter (global or spectra).
        """
        if par_type != 'global' and par_type != 'spectra':
            raise IndexError("%s is an invalid par_type. Must be 'global' or "
                             "'spectra'." % par_type)
        self._parameters[par._name] = {'par': par, 'type': par_type}

    def dump(self, basic=False):
        """ Dumps the config to a global fit config dictionary,
        containing all the 'global' parameters, and a spectral fit
        comfig dictionary (if required), containing any 'spectral'
        parameters that have been added. The dictionaries have,
        respectively, the forms specified in the
        :meth:`GlobalFitConfig.load` and
        :meth:`echidna.core.spectra.SpectralFitConfig.load` methods.

        Returns:
          dict: Dictionary containing all the information on the
            'global' parameters.
          dict: Dictionary containing all the information on the
            'spectral' parameters.
        """
        # Global fit parameters
        main_key = "global_fit_parameters"
        global_fit_config = {main_key: {}}

        for par in self.get_global_pars():
            dimension = par.get_dimension()

            # Make entry for dimensions - as required
            if dimension not in global_fit_config[main_key].keys():
                global_fit_config[main_key][dimension] = {}

            name = par.get_name()
            # Remove dimension from name, if required
            if dimension in name:
                name = name.replace(dimension+"_", "")

            # Get parameter dict from par
            global_fit_config[main_key][dimension][name] = par.to_dict(basic)

        # Spectral fit parameters
        main_key = "spectral_fit_parameters"
        spectral_fit_config = {main_key: {}}

        for par in self.get_spectra_pars():
            # No dimesnions required here
            name = par.get_name()

            # Get parameter dict from par
            spectral_fit_config[main_key][name] = par.to_dict(basic)

        return global_fit_config, spectral_fit_config

    def dump_to_file(self, path="", global_fname=None,
                     spectral_fname=None, basic=False):
        """ Write config(s) to YAML file. Separate files are created
        for global and spectral parameters.

        Args:
          path (string, optional): Location to save yaml file(s) to,
            default is the current directory.
          global_fname (string, optional): Filename for global
            parameters yaml file. If no filename is supplied, the
            default is "global_fit_config.yml". If a blank filename ""
            is given the config's name is used (+ "_global").
          spectral_fname (string, optional): Filename for spectral
            parameters yaml file. If no filename is supplied, the
            default is "spectral_fit_config.yml". If a blank filename ""
            is given the config's name is used (+ "_spectral").
          basic (bool, optional): If True, only the basic properties:
            prior, sigma, low, high and bins are included.
        """
        global_fit_config, spectral_fit_config = self.dump(basic)
        if global_fname is None:
            global_fname = "global_fit_config"
        elif global_fname == "":
            global_fname = self.get_name()
        if ".yml" not in global_fname:
            global_fname += ".yml"
        with open(path+global_fname, "w") as stream:
            yaml.dump(global_fit_config, stream=stream, indent=8)

        if spectral_fname is None:
            spectral_fname = "spectral_fit_config"
        elif spectral_fname == "":
            spectral_fname = self.get_name()
        if ".yml" not in spectral_fname:
            spectral_fname += ".yml"
        with open(path+spectral_fname, "w") as stream:
            yaml.dump(spectral_fit_config, stream=stream, indent=8)

    def get_par(self, name):
        """ Get requested parameter:

        Args:
          name (string): Name of the parameter

        Returns:
          :class:`echidna.core.spectra.FitParameter`: The requested parameter.
        """
        return self._parameters[name]['par']

    def get_global_pars(self):
        """ Gets the parameters which are applied to all spectra
          simultaneously.

        Returns:
          list: Of :class:`echidna.core.spectra.FitParameter` objects.
        """
        pars = []
        for name in self._parameters:
            if self._parameters[name]['type'] == 'global':
                pars.append(self._parameters[name]['par'])
        return pars

    def get_spectra_pars(self):
        """ Gets the parameters that are applied to individual spectra.

        Returns:
          list: Of :class:`echidna.core.spectra.FitParameter` objects.
        """
        pars = []
        for name in self._parameters:
            if self._parameters[name]['type'] == 'spectra':
                pars.append(self._parameters[name]['par'])
        return pars

    @classmethod
    def load(cls, global_config, spectral_config=None,
             name="global_fit_config"):
        """Initialise GlobalFitConfig class from a config dictionary
        (classmethod).

        Args:
          config (dict): Dictionary to create config out of.
          spectral_config (dict): Dictionary of spectral fit parameters
            to create config out of.
          name (string, optional): Name to assign to the
            :class:`GlobalFitConfig`. If no name is supplied the
            default 'global_fit_config' will be used.

        Returns:
          (:class:`echidna.core.spectra.GlobalFitConfig`): A config object
            containing the parameters in the file called filename.

        Raises:
          KeyError: If the :arg:`global_config` dictionary does not
            start with the key 'global_fit_parameters' as this suggests
            the dictionary has the wrong format.
          IndexError: If an invalid global fit parameter name is
            encountered.
          KeyError: If the :arg:`spectral_config` dictionary does not
            start with the key 'spectral_fit_parameters' as this
            suggests the dictionary has the wrong format.
          IndexError: If an invalid spectral fit parameter name is
            encountered.

        .. warning:: Valid format (for global_config) is::

            {"gloabal_fit_parameters":
                "<spectral_dimension>":
                    "<parameter_name>":
                        "prior": <prior>,
                        "sigma": <sigma>
                        "low": <low>
                        "high": <high>
                        "bins": <bins>}

        For spectral config see :meth:`SpectralFitConfig.load`.
        """
        main_key = "global_fit_parameters"
        parameters = collections.OrderedDict()
        if main_key not in global_config.keys():
            raise KeyError("Cannot read global_config dictionary. "
                           "Please check it has the correct form")
        for dim in global_config[main_key]:
            for syst in global_config[main_key][dim]:
                name = dim + "_" + syst
                if syst == 'resolution' or syst == 'resolution_ly':
                    parameters[name] = {
                        'par': ResolutionParameter(
                            name, dimension=dim,
                            **global_config[main_key][dim][syst]),
                        'type': 'global'}
                elif syst == 'shift':
                    parameters[name] = {
                        'par': ShiftParameter(
                            name, dimension=dim,
                            **global_config[main_key][dim][syst]),
                        'type': 'global'}
                elif syst == 'scale':
                    parameters[name] = {
                        'par': ScaleParameter(
                            name, dimension=dim,
                            **global_config[main_key][dim][syst]),
                        'type': 'global'}
                else:
                    raise IndexError("%s is not a valid global fit parameter."
                                     % syst)
        if spectral_config is None:
            return cls(name, parameters)

        # Add spectral fit parameters:
        main_key = "spectral_fit_parameters"
        if not spectral_config.get(main_key):
            raise KeyError("Cannot read config dictionary. "
                           "Please check it has the correct form")
        for syst in spectral_config[main_key]:
            if "rate" in syst:
                parameters[syst] = {
                    'par': RateParameter(
                        syst, **spectral_config[main_key][syst]),
                    'type': 'spectral'}
            else:
                raise IndexError("Unknown systematic in config: %s" % syst)

        return cls(name, parameters)

    @classmethod
    def load_from_file(cls, filename, sf_filename=None, name=None):
        """Initialise GlobalFitConfig class from a config file (classmethod).

        Args:
          filename (string): path to config file
          sf_filename (string, optional): path to a separate spectral
            fit config file, to include.
          name (string, optional): Assign a name to the
            :class:`GlobalFitConfig` created. If no name is supplied,
            the default is 'global_fit_config'. If a blank string is
            supplied, the name of the file will be used.

        Returns:
          (:class:`echidna.core.spectra.GlobalFitConfig`): A config object
            containing the parameters in the file called filename.
        """
        config = yaml.load(open(filename, 'r'))
        if sf_filename:
            spectral_fit_config = yaml.load(open(sf_filename, "r"))
        else:
            spectral_fit_config = None
        if not name:
            return cls.load(config, spectral_config=spectral_fit_config)
        if name == "":
            name = filename[filename.rfind("/")+1:filename.rfind(".")]
        return cls.load(config, spectral_config=spectral_fit_config, name=name)


class SpectraFitConfig(Config):
    """Configuration container for floating systematics and fitting Spectra
      objects.  Able to load directly with a set list of FitParameters or
      from yaml configuration files.

    Args:
      config_name (string): Name of config
      parameters (:class:`collections.OrderedDict`): List of
        FitParameter objects
      spectra_name (string): Name of the spectra associated with the
         :class:`echidna.core.spectra.SpectraFitConfig`

    Attributes:
      _spectra_name (string): Name of the spectra associated with the
        :class:`echidna.core.spectra.SpectraFitConfig`
    """

    def __init__(self, config_name, parameters, spectra_name):
        """Initialise SpectraFitConfig class
        """
        super(SpectraFitConfig, self).__init__(config_name, parameters)
        self._type = "spectra_fit"
        self._spectra_name = spectra_name

    def dump(self, basic=False):
        """ Dumps the config to a spectral fit comfig dictionary,
        containing all 'spectral' fit parameters. The dictionary has
        the form specified in the :meth:`SpectralFitConfig.load`
        method.

        Returns:
          dict: Dictionary containing all the information on the
            'spectral' parameters.
        """
        # Spectral fit parameters
        main_key = "spectral_fit_parameters"
        spectral_fit_config = {main_key: {}}

        for parameter in self.get_pars():
            par = self.get_par(parameter)

            # Get parameter dict from par
            spectral_fit_config[main_key][parameter] = par.to_dict(basic)

        return spectral_fit_config

    def dump_to_file(self, path="", spectral_fname=None, basic=False):
        """ Write config(s) to YAML file. Separate files are created
        for global and spectral parameters.

        Args:
          path (string, optional): Location to save yaml file(s) to,
            default is the current directory.
          spectral_fname (string, optional): Filename for spectral
            parameters yaml file. If no filename is supplied, the
            default is "spectral_fit_config.yml". If a blank filename ""
            is given the config's name is used (+ "_spectral").
          basic (bool, optional): If True, only the basic properties:
            prior, sigma, low, high and bins are included.
        """
        spectral_fit_config = self.dump(basic)
        if spectral_fname is None:
            spectral_fname = "spectral_fit_config"
        elif spectral_fname == "":
            spectral_fname = self.get_name()
        if ".yml" not in spectral_fname:
            spectral_fname += ".yml"
        with open(path+spectral_fname, "w") as stream:
            yaml.dump(spectral_fit_config, stream=stream, indent=8)

    @classmethod
    def load(cls, config, spectra_name, name="spectral_fit_config"):
        """Initialise SpectraFitConfig class from a config dictionary
        (classmethod).

        Args:
          config (dict): Dictionary to create config out of.
          name (string, optional): Name to assign to the
            :class:`SpectraFitConfig`. If no name is supplied the
            default 'spectral_fit_config' will be used.

        Returns:
          (:class:`SpectraFitConfig`): A config object containing the
            parameters from the config dictionary.

        Raises:
          KeyError: If the :arg:`config` dictionary does not start with
            the key 'spectral_fit_parameters' as this suggests the
            dictionary has the wrong format.
          IndexError: If an invalid spectral fit parameter name is
            encountered.

        .. warning:: Valid format is::

            {"spectral_fit_parameters":
                "<parameter_name>":
                    "prior": <prior>,
                    "sigma": <sigma>
                    "low": <low>
                    "high": <high>
                    "bins": <bins>}
        """
        main_key = "spectral_fit_parameters"
        if not config.get(main_key):
            raise KeyError("Cannot read config dictionary. "
                           "Please check it has the correct form")
        parameters = collections.OrderedDict()
        for syst in config[main_key]:
            if "rate" in syst:
                parameters[syst] = RateParameter(syst,
                                                 **config[main_key][syst])
            else:
                raise IndexError("Unknown systematic in config: %s" % syst)
        return cls(name, parameters, spectra_name)

    @classmethod
    def load_from_file(cls, filename, spectra_name, name=None):
        """Initialise SpectraFitConfig class from a config file (classmethod).

        Args:
          filename (str): path to config file
          spectra_name (string): Name of the spectra associated with the
            :class:`echidna.core.spectra.SpectraFitConfig`
          name (string, optional): Assign a name to the
            :class:`SpectraFitConfig` created. If no name is supplied,
            the default is 'spectral_fit_config'. If a blank string is
            supplied, the name of the file will be used.

        Returns:
          (:class:`SpectraFitConfig`): A config object containing the
            parameters in the file.
        """
        config = yaml.load(open(filename, 'r'))
        if not name:
            return cls.load(config, spectra_name)
        if name == "":
            name = filename[filename.rfind("/")+1:filename.rfind(".")]
        return cls.load(config, spectra_name, name=name)


class SpectraConfig(Config):
    """Configuration container for Spectra objects.  Able to load
    directly with a set list of SpectraParameters or from yaml
    configuration files.

    Args:
      parameters (:class:`collections.OrderedDict`): List of
        SpectraParameter objects
    """

    def __init__(self, config_name, parameters):
        """Initialise SpectraConfig class
        """
        super(SpectraConfig, self).__init__(config_name, parameters)
        self._type = "spectra"

    def dump(self):
        """ Dumps the spectra config to a config dictionary, containing
        all spectra parameters. The dictionary has the form specified
        in the :meth:`SpectraConfig.load` method.

        Returns:
          dict: Dictionary containing all the information on the
            spectra parameters.
        """
        # Spectral parameters
        main_key = "parameters"
        config = {main_key: {}}

        for parameter in self.get_pars():
            par = self.get_par(parameter)

            # Get parameter dict from par
            config[main_key][parameter] = par.to_dict()

        return config

    def dump_to_file(self, path="", filename=None):
        """ Write spectra config to YAML file.

        Args:
          path (string, optional): Location to save yaml file to,
            default is the current directory.
          filename (string, optional): Filename for yaml file. If no
            filename is supplied, the default is "spectra_config.yml".
            If a blank filename "" is given the config's name is used.
        """
        config = self.dump()
        if filename is None:
            filename = "spectra_config"
        elif filename == "":
            filename = self.get_name()
        if ".yml" not in filename:
            filename += ".yml"
        with open(path+filename, "w") as stream:
            yaml.dump(config, stream=stream, indent=8)

    @classmethod
    def load(cls, config, name="config"):
        """Initialise SpectraConfig class from a config dictionary
        (classmethod).

        Args:
          config (dict): Dictionary to create spectra config out of.
          name (string, optional): Name to assign to the
            :class:`SpectraConfig`. If no name is supplied the default
            'spectra_config' will be used.

        Returns:
          (:class:`SpectraConfig`): A config object containing the
            spectra parameters from the config dictionary.

        Raises:
          KeyError: If the :arg:`config` dictionary does not start with
            the key 'parameters' as this suggests the dictionary has
            the wrong format.

        .. warning:: Valid format is::

            {"parameters":
                "<spectral_parameter>":
                    "low": <low>
                    "high": <high>
                    "bins": <bins>}
        """
        main_key = "parameters"
        if not config.get(main_key):
            raise KeyError("Cannot read config dictionary. "
                           "Please check it has the correct form")
        parameters = collections.OrderedDict()
        for parameter in config[main_key]:
            parameters[parameter] = SpectraParameter(
                parameter, **config[main_key][parameter])
        return cls(name, parameters)

    @classmethod
    def load_from_file(cls, filename, name=None):
        """Initialise SpectraConfig class from a config file
        (classmethod).

        Args:
          filename (str): path to config file
          name (string, optional): Assign a name to the
            :class:`SpectraConfig` created. If no name is supplied, the
            default is 'spectra_config'. If a blank string is supplied,
            the name of the file will be used.

        Returns:
          (:class:`SpectraConfig`): A config object containing the
            parameters in the file.
        """
        with open(filename, 'r') as stream:
            config = yaml.load(stream)
        if not name:
            return cls.load(config)
        if name == "":
            name = filename[filename.rfind("/")+1:filename.rfind(".")]
        return cls.load(config, name)

    def get_dims(self):
        """Get list of dimension names.
        The _mc, _reco and _truth suffixes are removed.

        Returns:
          list: List of the dimensions names of the config.
        """
        dims = []
        for par in sorted(self._parameters.keys()):
            par = par.split('_')[:-1]
            dim = ""
            for entry in par:
                dim += entry+"_"
            dims.append(dim[:-1])
        return dims

    def get_dim(self, par):
        """Get the dimension of par.
        The _mc, _reco and _truth suffixes are removed.

        Args:
          par (string): Name of the parameter

        Returns:
          The dimension of par
        """
        dim = ""
        for entry in par.split('_')[:-1]:
            dim += entry+"_"
        return dim[:-1]

    def get_dim_type(self, dim):
        """Returns the type of the dimension i.e. mc, reco or truth.

        Args:
          dim (string): The name of the dimension

        Raises:
          IndexError: dim is not in the spectra.

        Returns:
          string: The type of the dimension (mc, reco or truth)
        """
        for par in sorted(self._parameters.keys()):
            par_split = par.split('_')[:-1]
            cur_dim = ""
            for entry in par_split:
                cur_dim += entry+"_"
            if cur_dim[:-1] == dim:
                return str(par.split('_')[-1])
        raise IndexError("No %s dimension in spectra" % dim)


class Spectra(object):
    """ This class contains a spectra as a function of energy, radius and time.

    The spectra is stored as histogram binned in energy, x, radius, y, and
    time, z. This histogram can be flattened to 2d (energy, radius) or 1d
    (energy).

    Args:
      name (str): The name of this spectra
      num_decays (float): The number of decays this spectra is created to
        represent.
      spectra_config (:class:`SpectraConfig`): The configuration object

    Attributes:
      _data (:class:`numpy.ndarray`): The histogram of data
      _name (str): The name of this spectra
      _config (:class:`SpectraConfig`): The configuration object
      _num_decays (float): The number of decays this spectra currently
        represents.
      _raw_events (int): The number of raw events used to generate the
        spectra. Increments by one with each fill independent of
        weight.
      _bipo (int): Flag to indicate whether the bipo cut was applied to the
        spectra. 0 is No Cut. 1 is Cut.
        Int type as HDF5 does not support bool.
      _style (string): Pyplot-style plotting style e.g. "b-" or
        {"color": "blue"}.
      _rois (dict): Dictionary containing the details of any ROI, along
        any axis, which has been defined.
    """
    def __init__(self, name, num_decays, spectra_config, fit_config=None):
        """ Initialise the spectra data container.
        """
        self._config = spectra_config
        self._raw_events = 0
        bins = []
        for v in self._config.get_pars():
            bins.append(self._config.get_par(v)._bins)
        self._data = numpy.zeros(shape=tuple(bins),
                                 dtype=float)
        self._fit_config = fit_config
        # Flag for indicating bipo cut. HDF5 does not support bool so
        # 0 = no cut and 1 = cut
        self._bipo = 0
        self._style = {"color": "blue"}  # default style for plotting
        self._rois = {}
        self._name = name
        self._num_decays = float(num_decays)

    def add(self, spectrum):
        """ Adds a spectrum to current spectra object.

        Args:
          spectrum (:class:`Spectra`): Spectrum to add.

        Raises:
          ValueError: spectrum has different dimenstions to the current
            spectra.
          IndexError: spectrum does not contain a dimension(s) that is in the
            current spectra config.
          IndexError: The current spectra does not contain a dimension(s) that
            is in the spectrum config.
          ValueError: The upper bounds of a parameter in the current spectra
            and spectra are not equal.
          ValueError: The lower bounds of a parameter in the current spectra
            and spectra are not equal.
          ValueError: The number of bins of a parameter in the current spectra
            and spectra are not equal.
        """
        if self._data.shape != spectrum._data.shape:
            raise ValueError("The spectra have different dimensions.\n"
                             "Dimension of self: %s. Dimension of spectrum %s"
                             % (self._data.shape, spectrum._data.shape))
        for v in self._config.get_dims():
            if v not in spectrum.get_config().get_dims():
                raise IndexError("%s not present in new spectrum" % v)
        for v in spectrum.get_config().get_dims():
            if v not in self._config.get_dims():
                raise IndexError("%s not present in this spectrum" % v)
        # Dictionary containing dimensions which have different types in the
        # two spectra. The type of the dimension of spectrum is the value
        types = {}
        for v in spectrum.get_config().get_pars():
            if v not in self._config.get_pars():
                dim = spectrum.get_config().get_dim(v)
                dim_type = spectrum._config.get_dim_type(dim)
                types[dim] = dim_type
        for v in self._config.get_pars():
            dim = self._config.get_dim(v)
            if dim in types:
                v_spec = dim+'_'+types[dim]
            else:
                v_spec = v
            if not numpy.allclose(self.get_config().get_par(v)._high,
                                  spectrum.get_config().get_par(v_spec)._high):
                raise ValueError("Upper %s bounds in spectra are not equal."
                                 "\n%s upper bound: %s\n%s upper bound: %s"
                                 % (v, self._name,
                                    self.get_config().get_par(v)._high,
                                    spectrum._name,
                                    spectrum.get_config().get_par(v_spec)
                                    ._high))
            if not numpy.allclose(self.get_config().get_par(v)._low,
                                  spectrum.get_config().get_par(v_spec)._low):
                raise ValueError("Lower %s bounds in spectra are not equal."
                                 "\n%s lower bound: %s\n%s lower bound: %s"
                                 % (v, self._name,
                                    self.get_config().get_par(v)._low,
                                    spectrum._name,
                                    spectrum.get_config().get_par(v_spec)
                                    ._low))
            if self.get_config().get_par(v)._bins != \
                    spectrum.get_config().get_par(v_spec)._bins:
                raise ValueError("Number of %s bins in spectra are not equal."
                                 "\n%s bins: %s\n%s lower bins: %s"
                                 % (v, self._name,
                                    self.get_config().get_par(v)._bins,
                                    spectrum._name,
                                    spectrum.get_config().get_par(v_spec)
                                    ._bins))
        self._data += spectrum._data
        self._raw_events += spectrum._raw_events
        self._num_decays += spectrum._num_decays

    def cut(self, **kwargs):
        """ Similar to :meth:`shrink`, but updates scaling information.

        If a spectrum is cut using :meth:`shrink`, subsequent calls to
        :meth:`scale` the spectrum must still scale the *full* spectrum
        i.e. before any cuts. The user supplies the number of decays
        the full spectrum should now represent.

        However, sometimes it is more useful to be able specify the
        number of events the revised spectrum should represent. This
        method updates the scaling information, so that it becomes the
        new *full* spectrum.

        Args:
          kwargs (float): Named parameters to slice on; note that these
            must be of the form [name]_low or [name]_high where [name]
            is a dimension present in the SpectraConfig.
        """
        initial_count = self.sum()  # Store initial count
        self.shrink(**kwargs)
        new_count = self.sum()
        reduction_factor = float(new_count) / float(initial_count)
        # This reduction factor tells us how much the number of detected events
        # has been reduced by shrinking the spectrum. We want the number of
        # decays that the spectrum should now represent to be reduced by the
        # same factor
        self._num_decays *= reduction_factor

    def fill(self, weight=1.0, **kwargs):
        """ Fill the bin with weight.  Note that values for all named
        parameters in the spectra's config (e.g. energy, radial) must be
        passed.

        Args:
          weight (float, optional): Defaults to 1.0, weight to fill the bin
            with.
          kwargs (float): Named values (e.g. for energy_mc, radial_mc)

        Raises:
          Exception: Parameter in kwargs is not in config.
          Exception: Parameter in config is not in kwargs.
          ValueError: If the energy, radius or time is beyond the bin limits.
        """
        # Check all keys in kwargs are in the config parameters and visa versa
        for par in kwargs:
            if par not in self._config.get_pars():
                raise Exception('Unknown parameter %s' % par)
        for par in self._config.get_pars():
            if par not in kwargs:
                raise Exception('Missing parameter %s' % par)
        for v in self._config.get_pars():
            if not self._config.get_par(v)._low <= kwargs[v] < \
                    self._config.get_par(v)._high:
                raise ValueError("%s out of range: %s" % (v, kwargs[v]))
        bins = []
        for v in self._config.get_pars():
            bins.append(int((kwargs[v] - self._config.get_par(v)._low) /
                            (self._config.get_par(v)._high -
                             self._config.get_par(v)._low) *
                            self._config.get_par(v)._bins))
        self._data[tuple(bins)] += weight

    def get_bipo(self):
        """ Get the BiPo flag value of the spectra (no BiPo cut = 0,
        BiPo cut = 1)

        Returns:
          int: The BiPo flag value of the spectra.
        """
        return self._bipo

    def get_config(self):
        """ Get the config of the spectra.

        Returns:
          :class:`echidna.core.spectra.SpectraConfig`: The config of
            the spectra.
        """
        return self._config

    def get_data(self):
        """
        Returns:
          (:class:`numpy.ndarray`): The spectral data.
        """
        return self._data

    def get_fit_config(self):
        """ Get the config of the spectra.

        Returns:
          :class:`echidna.core.spectra.SpectraConfig`: The config of
            the spectra.
        """
        return self._fit_config

    def get_name(self):
        """
        Returns:
          string: The name of the spectra.
        """
        return self._name

    def get_num_decays(self):
        """
        Returns:
          float: The number of decays the spectrum represents (copy).
        """
        return copy.copy(self._num_decays)  # Don't want to edit accidentally!

    def get_roi(self, dimension):
        """ Access information about a predefined ROI for a given dimension

        Returns:
          dict: Dictionary containing parameters defining the ROI, on
            the given dimension.
        """
        return self._rois[dimension]

    def get_style(self):
        """
        Returns:
          string/dict: :attr:`_style` - pyplot-style plotting style.
        """
        return self._style

    def interpolate1d(self, dimension, kind='cubic'):
        """ Interpolates a given dimension of a spectra.

        Args:
          dimension (string): Dimension you want to interpolate.
          kind (string): Method of interpolation.
            See :class:`scipy.interpolate.interp1d` for available methods.

        Returns:
          :class:`scipy.interpolate.interp1d`: Interpolation function.
        """
        x = self._config.get_par(dimension).get_bin_centres()
        y = self.project(dimension)
        return interpolate.interp1d(x, y, kind=kind, bounds_error=False)

    def nd_project(self, dimensions):
        """ Project the histogram along an arbitary number of axes.

        Args:
          dimensions (str): List of axes to project onto

        Returns:
          :class:`numpy.ndarray`: The nd projection of the histogram.
        """
        axes = []
        for dim in dimensions:
            axes.append(self._config.get_index(dim))
        if len(axes) == len(self._config.get_pars()):
            return copy.copy(self._data)
        projection = copy.copy(self._data)
        for i_axis in range(len(self._config.get_pars())):
            if i_axis not in axes:
                projection = projection.sum(i_axis)
        return projection

    def project(self, dimension):
        """ Project the histogram along an axis for a given dimension.
        Note that the dimension must be one of the named parameters in
        the SpectraConfig.

        Args:
          dimension (str): parameter to project onto

        Returns:
          :class:`numpy.ndarray`: The projection of the histogram onto the
            given axis
        """
        axis = self._config.get_index(dimension)
        projection = copy.copy(self._data)
        for i_axis in range(len(self._config.get_pars()) - 1):
            if axis < i_axis+1:
                projection = projection.sum(1)
            else:
                projection = projection.sum(0)
        return projection

    def rebin(self, new_bins):
        """ Rebin spectra data into a smaller spectra of the same rank whose
        dimensions are factors of the original dimensions.

        Args:
          new_bins (tuple): new binning, this must match both the
            number and ordering of dimensions in the spectra config.
            For example if the old data shape is made of bins (1000, 10)
            and you would like to increase the bin width of both by 2 then
            you must pass the tuple (500, 5)

        Raises:
          ValueError: Shape mismatch. Number of dimenesions are different.
          ValueError: Old bins/ New bins must be integer
        """
        # Check all keys in kwargs are in the config parameters and visa versa
        if len(new_bins) != len(self._config.get_pars()):
            raise ValueError('Incorrect number of dimensions; need %s'
                             % len(self._config.get_pars()))
        # Now do the rebinning
        for i, v in enumerate(self._config.get_pars()):
            if self._config.get_par(v)._bins % new_bins[i] != 0:
                raise ValueError("Old bins/New bins must be integer old: %s"
                                 " new: %s for parameter %s"
                                 % (self._config.get_par(v)._bins,
                                    new_bins[i], v))
            self._config.get_par(v)._bins = new_bins[i]

        compression_pairs = [(d, c//d) for d, c in zip(new_bins,
                                                       self._data.shape)]
        flattened = [l for p in compression_pairs for l in p]
        self._data = self._data.reshape(flattened)
        for i in range(len(new_bins)):
            self._data = self._data.sum(-1*(i+1))

    def scale(self, num_decays):
        """ Scale THIS spectra to represent *num_decays* worth of decays over
        the entire unshrunken spectra.

        This rescales each bin by the ratio of *num_decays* to
        *self._num_decays*, i.e. it changes the spectra from representing
        *self._num_decays* to *num_decays*. *self._num_decays* is updated
        to equal *num_decays* after.

        Args:
          num_decays (float): Number of decays this spectra should represent.
        """
        self._data = numpy.multiply(self._data, num_decays / self._num_decays)
        # Make sure self._num_decays stays as a float
        self._num_decays = float(num_decays)

    def set_fit_config(self, config):
        """ Get the config of the spectra.

        Args:
          config (:class:`echidna.core.spectra.SpectraFitConfig`): The fit
            config to assign to the spectra.
        """
        if isinstance(config, SpectraFitConfig):
            self._fit_config = config
        else:
            raise TypeError("Invalid config type: %s" % type(config))

    def set_style(self, style):
        """ Sets plotting style.

        Styles should be valid pyplot style strings e.g. "b-", for a
        blue line, or dictionaries of strings e.g. {"color": "red"}.

        Args:
          style (string, dict): Pyplot-style plotting style.
        """
        self._style = style

    def shrink(self, **kwargs):
        """ Shrink the data such that it only contains values between low and
        high for a given dimension by slicing. This updates the internal bin
        information as well as the data.

        Args:
          kwargs (float): Named parameters to slice on; note that these
            must be of the form [name]_low or [name]_high where [name]
            is a dimension present in the SpectraConfig.

        .. note:

          The logic in this method is the same for each dimension, first
          check the new values are within the existing ones
          (can only compress). Then calculate the low bin number and high bin
          number (relative to the existing binning low).
          Finally update all the bookeeping and slice.

        Raises:
          IndexError: Parameter which is being shrank does not exist in the
            config file.
          ValueError: [parameter]_low value is lower than the parameters lower
            bound.
          ValueError: [parameter]_high value is lower than the parameters
            higher bound.
          IndexError: Suffix to [parameter] is not _high or _low.
        """
        # First check dimensions and bounds in kwargs are valid
        for arg in kwargs:
            high_low = arg.split("_")[-1]
            par = arg[:-1*(len(high_low)+1)]
            if par not in self._config.get_pars():
                raise IndexError("%s is not a parameter in the config" % par)
            if high_low == "low":
                if numpy.allclose(kwargs[arg], self._config.get_par(par)._low):
                    continue  # To avoid floating point errors
                if kwargs[arg] < self._config.get_par(par)._low:
                    raise ValueError("%s is below existing bound for %s (%s)"
                                     % (kwargs[arg], par,
                                        self._config.get_par(par)._low))
            elif high_low == "high":
                if numpy.allclose(kwargs[arg],
                                  self._config.get_par(par)._high):
                    continue  # To avoid floating point errors
                if kwargs[arg] > self._config.get_par(par)._high:
                    raise ValueError("%s is above existing bound for %s (%s)"
                                     % (kwargs[arg], par,
                                        self._config.get_par(par)._high))
            else:
                raise IndexError("%s index invalid. Index must be of the form"
                                 "[dimension name]_high or"
                                 "[dimension name]_low" % arg)
        slices_low = []
        slices_high = []
        for par_name in self._config.get_pars():
            par = self._config.get_par(par_name)
            kw_low = "%s_low" % par_name
            kw_high = "%s_high" % par_name
            if "%s_low" % par_name not in kwargs:
                kwargs[kw_low] = par._low
            if "%s_high" % par_name not in kwargs:
                kwargs[kw_high] = par._high
            # Round down the low bin
            low_bin = int((kwargs[kw_low] - par._low) / par.get_width())
            # Round up the high bin
            high_bin = numpy.ceil((kwargs[kw_high] - par._low) /
                                  par.get_width())
            # new_low is the new lower first bin edge
            new_low = par.round(par._low + low_bin * par.get_width())
            # new_high is the new upper last bin edge
            new_high = par.round(par._low + high_bin * par.get_width())
            # Correct floating point errors: If the difference between
            # input high/low and calculated new high/low is approximately
            # equal (<1%) to a bin width then assume user requested the bin
            # above/below to be cut.
            if numpy.fabs(new_high - kwargs[kw_high]) > (0.99 *
                                                         par.get_width()):
                # print ("WARNING: Correcting possible floating point error in"
                #       "spectra.Spectra.shrink\n%s was the input. %s is the "
                #        "calculated value for %s" % (kwargs[kw_low],
                #                                     new_low, kw_low))
                if (new_high - kwargs[kw_high]) > 0.0:
                    high_bin -= 1
                    new_high = par.round(par._low + high_bin * par.get_width())
                else:
                    high_bin += 1
                    new_high = par.round(par._low + high_bin * par.get_width())
                # print "Corrected %s to %s" % (kw_low, new_low)
            if numpy.fabs(new_low - kwargs[kw_low]) > (0.99 * par.get_width()):
                # print ("WARNING: Correcting possible floating point error in"
                #       "spectra.Spectra.shrink\n%s was the input. %s is the "
                #       "calculated value for %s" % (kwargs[kw_low],
                #                                    new_low, kw_low))
                if (new_low - kwargs[kw_low]) > 0.0:
                    low_bin -= 1
                    new_low = par._low + low_bin * par.get_width()
                else:
                    low_bin += 1
                    new_low = par._low + low_bin * par.get_width()
                print "Corrected %s to %s" % (kw_low, new_low)
            slices_high.append(high_bin)
            slices_low.append(low_bin)
            new_bins = high_bin - low_bin
            par.set_par(low=new_low, high=new_high, bins=new_bins)
        # First set up the command then evaluate. Hacky but hey ho.
        cmd = "self._data["
        for i in range(len(slices_low)):
            low = str(slices_low[i])
            high = str(slices_high[i])
            cmd += low+":"+high+","
        cmd = cmd[:-1]+"]"
        self._data = eval(cmd)

    def shrink_to_roi(self, lower_limit, upper_limit, dimension):
        """ Shrink spectrum to a defined Region of Interest (ROI)

        Shrinks spectrum to given ROI and saves ROI parameters.

        Args:
          lower_limit (float): Lower bound of ROI, along given axis.
          upper_limit (float): Upper bound of ROI, along given axis.
          dimension (str): Name of the dimension to shrink.
        """
        integral_full = self.sum()  # Save integral of full spectrum

        # Shrink to ROI
        kw_low = dimension+"_low"
        kw_high = dimension+"_high"
        kw_args = {kw_low: lower_limit,
                   kw_high: upper_limit}
        self.shrink(**kw_args)

        # Calculate efficiency
        integral_roi = self.sum()  # Integral of spectrum over ROI
        efficiency = float(integral_roi) / float(integral_full)
        par = self.get_config().get_par(dimension)
        self._rois[dimension] = {"low": par._low,
                                 "high": par._high,
                                 "efficiency": efficiency}

    def sum(self):
        """ Calculate and return the sum of the `_data` values.

        Returns:
          float: The sum of the values in the `_data` histogram.
        """
        return self._data.sum()

    def surface(self, dimension1, dimension2):
        """ Project the histogram along two axes for the given dimensions.
        Note that the dimensions must be one of the named parameters in
        the SpectraConfig.

        Args:
          dimension1 (str): first parameter to project onto
          dimension1 (str): second parameter to project onto

        Raises:
          IndexError: Axis of dimension1 is out of range
          IndexError: Axis of dimension2 is out of range

        Returns:
          :class:`numpy.ndarray`: The 2d surface of the histogram.
        """
        axis1 = self._config.get_index(dimension1)
        axis2 = self._config.get_index(dimension2)
        if axis1 < 0 or axis1 > len(self._config.get_pars()):
            raise IndexError("Axis index %s out of range" % axis1)
        if axis2 < 0 or axis2 > len(self._config.get_pars()):
            raise IndexError("Axis index %s out of range" % axis2)
        projection = copy.copy(self._data)
        for i_axis in range(len(self._config.get_pars())):
            if i_axis != axis1 and i_axis != axis2:
                projection = projection.sum(i_axis)
        return projection
