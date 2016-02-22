""" Module containing classes that act as minimisers in a fit.
"""
import numpy

from echidna.fit.fit_results import FitResults, BinnedFitResults

import copy
import abc


class Minimiser(object):
    """ Base class for minimiser objects.

    Args:
      name (string): Name of minimiser.

    Attributes:
      _name (string): Name of minimiser.
      _type (string): Type of minimiser, e.g. GridSearch
    """
    __metaclass__ = abc.ABCMeta  # Only required for python 2

    def __init__(self, name):
        self._name = name
        self._type = None  # No type for base class
        self._binned = False # Default

    def get_name(self):
        return self._name

    def is_binned(self):
        return self._binned

    @abc.abstractmethod
    def minimise(self, fit_config, spectra_config, funct, test_statistic):
        """ Abstract base class method to override.

        Args:
          funct (callable): Callable function to calculate the value
            of the test statistic you wish to minimise, for each
            combination of parameter values tested. The function must
            only accept, as arguments, a variable-sized array of
            parameter values. E.g. ``def funct(*args)``. Within the
            echidna framework, the :meth:`echidna.limit.fit.Fit.funct`
            method is the recommened callable to use here.
          test_statistic (:class:`echidna.limit.test_statistic`): The
            test_statistic object used to calcualte the test statistics.

        Returns:
          float: Minimum value found during minimisation.
        """
        raise NotImplementedError("The minimise method can only be used "
                                  "when overridden in a derived class")


class BinnedMinimiser(Minimiser):
    """ Base class for binned minimisers

    Args:
      name (string): Name of minimiser.
      per_bin (bool, optional): Flag if the minimiser should expect a
        test statistic value per bin.

    Attributes:
      _name (string): Name of minimiser.
      _type (string): Type of minimiser, e.g. GridSearch
      _per_bin (bool, optional): Flag if minimiser should expect a
        test statistic value per-bin.
    """

    def __init__(self, name, per_bin = False):
        super(BinnedMinimiser, self).__init__(name)
        self._per_bin = per_bin
        self._binned = True


class GridSearch(BinnedMinimiser):
    """ A grid-search minimisation algorithm.

    Args:
      name (str, optional): Name of this :class:`FitResults` class
        instance. If no name is supplied, name from fit_results will be
        taken and appended with "_results".
      per_bin (bool, optional): Flag if minimiser should expect a
        test statistic value per-bin.
      use_numpy (bool, optional): Flag to indicate whether to use the
        built-in numpy functions for minimisation and locating the
        minimum, or use the :meth:`find_minimum` method. Default is to
        use numpy.

    Attributes:
      _name (string): Name of this :class:`GridSearch` class instance.
      _stats (:class:`numpy.ndarray`): Array of values of the test
        statistic calculated during the fit.
      _penalties (:class:`numpy.ndarray`): Array of values of the
        penalty terms calculated during the fit.
      _minimum_value (float): Minimum value of the array returned by
        :meth:`get_fit_data`.
      _minimum_position (tuple): Position of the test statistic minimum
        value. The tuple contains the indices along each fit parameter
        (dimension), acting as coordinates of the position of the
        minimum.
      _resets (int): Number of times the grid has been reset.
      _type (string): Type of minimiser, e.g. GridSearch
      _per_bin (bool, optional): Flag if minimiser should expect a
        test statistic value per-bin.
      _use_numpy (bool, optional): Flag to indicate whether to use the
        built-in numpy functions for minimisation and locating the
        minimum, or use the :meth:`find_minimum` method. Default is to
        use numpy.
    """
    def __init__(self, name=None, per_bin=False, use_numpy=True):
        super(GridSearch, self).__init__(name)
        self._type = GridSearch
        self._per_bin = per_bin
        self._use_numpy = use_numpy

    def minimise(self, fit_config, spectra_config, funct, test_statistic):
        """ Method to perform the minimisation.

        Args:
          funct (callable): Callable function to calculate the value
            of the test statistic you wish to minimise, for each
            combination of parameter values tested. The function must
            only accept, as arguments, a variable-sized array of
            parameter values. E.g. ``def funct(*args)``. Within the
            echidna framework, the :meth:`echidna.limit.fit.Fit.funct`
            method is the recommened callable to use here.
          test_statistic (:class:`echidna.limit.test_statistic`): The
            test_statistic object used to calcualte the test statistics.

        Attributes:
          _minimum_value (float): Minimum value of test statistic found.
          _minimum_position (tuple): Position of minimum.

        Returns:
          float: Minimum value found during minimisation.
        """
        # Start by setting the internal fit_config
        self._fit_config = fit_config
        # Generate the array that seems to be required later
        # Automatically resets the grid at the start of minimisation
        self._stats = numpy.zeros(fit_config.get_shape() + spectra_config.get_shape())
        self._penalty_terms = numpy.zeros(fit_config.get_shape())
        # Loop over all possible combinations of fit parameter values
        for values, indices in self._get_fit_par_values():
            # Call funct and pass array to it
            result, penalty = funct(*values)

            # Check result is of correct form
            if self._per_bin:  # expecting numpy.ndarray
                if not isinstance(result, numpy.ndarray):
                    raise TypeError("Expecting result of type numpy.ndarray "
                                    "(not %s), for per_bin enabled" %
                                    type(result))
            self.set_stat(result, tuple(indices))
            self.set_penalty_term(penalty, tuple(indices))

        # Now grid is filled minimise
        minimum = copy.copy(self._stats)

        if self._use_numpy:
            # Set best_fit values
            # This is probably not the most efficient way of doing this
            position = numpy.argmin(minimum)
            position = numpy.unravel_index(position, minimum.shape)
            minimum = numpy.nanmin(minimum)
        else:  # Use find_minimum method
            minimum, position = self.find_minimum(minimum)

        for index, par in zip(position, self._fit_config.get_pars()):
            parameter = self._fit_config.get_par(par)
            best_fit = parameter.get_values()[index]
            sigma = parameter.get_sigma()
            prior = parameter.get_prior()
            parameter.set_best_fit(parameter.get_values()[index])
            if sigma is not None:
                parameter.set_penalty_term(
                    test_statistic.get_penalty_term(best_fit, prior, sigma))
            else:  # penalty term = 0
                parameter.set_penalty_term(0.)

        self._minimum_value = minimum
        self._minimum_position = position
        #self.set_minimum_value(minimum)
        #self.set_minimum_position(position)  # save position of minimum
        # Return minimum to fitting
        #return minimum
        results = BinnedFitResults(fit_config, minimum, position, self._stats,
                                   self._penalty_terms)
        return results

    def _update_coords(self, coords, new_coords):
        """ Internal method called by :meth:`find_minimum` to update the
        stored co-ordinates of minima.

        This method takes the new co-ordinates in ``new_coords`` and
        works out the indices to select the correct (previously-
        calculated) co-ordinates for the positions of minima, in the
        inner dimensions.

        Args:
          coords (:class:`numpy.ndarray`): 2D array containing the
            previously calculated co-ordinates for the inner dimensions.
          new_coords (:class:`numpy.ndarray`): Array containing the
            co-ordinates of each minima calculated for the current
            dimension.

        Returns:
          (:class:`numpy.ndarray`): 2D array containing the updated
            arrays of co-ordinates, for all dimensions processed so far.
        """
        new_coords = new_coords.ravel()
        multiplier = 0  # to calculate indices
        product = 1  # product of dimensions
        for dim in new_coords.shape:
            product *= dim
            multiplier += len(coords) / product

        # Calculate indices
        indices = [i * multiplier + j for i, j in enumerate(new_coords)]

        # Index current co-ordinates
        coords = coords[indices]

        # Append new co-ordinates
        coords = numpy.concatenate((new_coords.reshape(len(new_coords), 1),
                                    coords), axis=1)
        return coords

    def find_minimum(self, array):
        """ Alternative method for finding the minimum.

        Starting from the innermost dimension, locates the minima
        along the axis - effectively minimising over lots of 1D arrays.
        Once the minima are located, this axis is collapsed down to
        next innermost, storing just the values at the minima. The
        position of each minima is also stored in the ``coords`` array.

        This process is then repeated for the next innermost array.
        However now when we locate the position of each minima, we
        also wish to get the corresponding position calculated at the
        previous dimension - the :meth:`_update_coords` does this,
        concatenating the current locations with the previous ones
        so we start to build up an array of co-ordinates.

        As the code works outwards through the dimensions, the
        number of co-ordinates are added, but the choice of co-ordinates
        is reduced. Until the outermost dimension is processed and then
        *ideally* only one minimum value and one set of co-ordinates
        remains.

        Args:
          array (:class:`numpy.ndarray`): Array to minimise.

        Returns:
          float: Minimum value in array.
          tuple: Location (co-ordinates) of minimum.

        .. warning:: If the exact minimum value is repeated more than
          once in the array, the location with the lowest coordinate
          values (starting from the outermost dimension) will be
          returned. E.g. if two equal minima are at (4, 10, 15) and
          (41, 2, 12), the location (4, 10, 15) would be returned
        """
        dims = len(array.shape)
        minimum = copy.copy(array)
        coords = None

        # Loop over dimensions, working outwards
        for i, dim in enumerate(reversed(list(range(dims)))):
            # Work out coordinates
            if coords is None:  # Create array
                coords = numpy.argmin(minimum, axis=dim).ravel()
                coords = coords.reshape(len(coords), 1)
            elif dim > 0:  # Update existing array
                new_coords = numpy.argmin(minimum, axis=dim)
                coords = self._update_coords(coords, new_coords)
            else:  # Last dimension - new_coords is a single float
                new_coords = numpy.argmin(minimum, axis=dim)
                coords = coords[new_coords]
                if coords is not numpy.array:
                    coords = numpy.array([coords])
                coords = numpy.insert(coords, 0, new_coords)

            # Collapse minima into outer dimension
            minimum = numpy.nanmin(minimum, axis=dim)

        coords = coords.ravel()
        coords = tuple(coords)
        return minimum, coords

    def _get_fit_par_values(self):
        """ Internal method. Gets next array of parameter values to
        test in fit.

        Yields:
          (:class:`numpy.array`): Next array of parameter values to
            test in fit.
          (:class:`numpy.array`): Indices of these parameter values.
        """
        index = 0
        values = numpy.zeros((len(self._fit_config.get_pars())))
        indices = numpy.zeros((len(self._fit_config.get_pars())))
        for values, indices in self._get_values(index, values, indices):
            yield values, indices

    def _get_values(self, index, values, indices):
        """ Internal method. Called recursively to populate the array
        of current parameter values.

        Yields:
          (:class:`numpy.array`): Next array of parameter values to
            test in fit.
          (:class:`numpy.array`): Indices of these parameter values.
        """
        if index < len(values):
            parameter = self._fit_config.get_par_by_index(index)
            for value in parameter.get_values():
                values[index] = value
                indices[index] = parameter.get_value_index(value)
                for values, indices in self._get_values(index+1,
                                                        values, indices):
                    yield values, indices
        else:
            yield values, indices

    def set_penalty_terms(self, penalty_terms):
        """ Sets the array containing penalty term values.

        Args:
          penalty_terms (:class:`numpy.ndarray`): The array of penalty
            term values

        Raises:
          TypeError: If penalty_terms is not an :class:`numpy.ndarray`
          ValueError: If the penalty_terms array does not have the required
            shape.
        """
        if not isinstance(penalty_terms, numpy.ndarray):
            raise TypeError("penalty_terms must be a numpy array")
        if penalty_terms.shape != self._fit_config.get_shape():
            raise ValueError("penalty_terms array has incorrect shape (%s), "
                             "expected shape is %s" %
                             (str(penalty_terms.shape),
                              str(self._fit_config.get_shape())))
        self._penalty_terms = penalty_terms

    def set_penalty_term(self, penalty_term, indices):
        """ Sets the total penalty term value at the point in the array
        specified by indices.

        Args:
          penalty_term (float): Best fit value of a fit parameter.
          indices (tuple): The index along each fit parameter dimension
            specifying the coordinates from which to set the total
            penalty term value.

        Raises:
          TypeError: If penalty_term is not a float.
          TypeError: If the indices supplied are not at tuple
          IndexError: If the number of indices supplied does not match
            the dimensions of the fit
          IndexError: If the indices supplied are out of bounds for
            the fit dimensions
        """
        if not isinstance(penalty_term, float):
            raise TypeError("penalty_term must be a float")
        if not isinstance(indices, tuple):
            raise TypeError("indices supplied must be a tuple of integers")
        if len(indices) != len(self._fit_config.get_shape()):
            raise IndexError("dimension mismatch, indices supplied contian "
                             "%d dimensions but fit contains %d dimensions "
                             "(parameters)" %
                             (len(indices), len(self._fit_config.get_shape())))
        if indices > self._fit_config.get_shape():
            raise IndexError(
                "indices %s out of bounds for fit with dimensions %s" %
                (str(indices), str(self._fit_config.get_shape())))
        self._penalty_terms[indices] = penalty_term

    def set_stat(self, stat, indices):
        """ Sets the test statistic values in array at the point
        specified by indices

        Args:
          stat (:class:`numpy.ndarray`): Values of the test statistic.
          indices (tuple): Position in the array.

        Raises:
          TypeError: If the indices supplied are not at tuple
          IndexError: If the number of indices supplied does not match
            the dimensions of the fit
          IndexError: If the indices supplied are out of bounds for
            the fit dimensions
          TypeError: If stat is not a :class:`numpy.ndarray`.
          ValueError: If the stats array has incorrect shape.
        """
        if not isinstance(indices, tuple):
            raise TypeError("indices supplied must be a tuple of integers")
        if len(indices) != len(self._fit_config.get_shape()):
            raise IndexError("dimension mismatch, indices supplied contian "
                             "%d dimensions but fit contains %d dimensions "
                             "(parameters)" %
                             (len(indices), len(self._fit_config.get_shape())))
        if indices > self._fit_config.get_shape():
            raise IndexError(
                "indices %s out of bounds for fit with dimensions %s" %
                (str(indices), str(self._fit_config.get_shape())))
        if not isinstance(stat, numpy.ndarray):
            raise TypeError("stat must be a numpy array")
        if stat.shape != self._stats[indices].shape:
            raise ValueError("stat array has incorrect shape (%s), "
                             "expected shape is %s" %
                             (str(stat.shape),
                              str(self._stats[indices].shape)))
        self._stats[indices] = stat

    def set_stats(self, stats):
        """ Sets the total test statistics array.

        Args:
          stats (:class:`numpy.ndarray`): The total test statistics array.

        Raises:
          TypeError: If stats is not a :class:`numpy.ndarray`.
          ValueError: If the stats array has incorrect shape.
        """
        if not isinstance(stats, numpy.ndarray):
            raise TypeError("stats must be a numpy array")
        if stats.shape != self._stats.shape:
            raise ValueError("stats array has incorrect shape (%s), "
                             "expected shape is %s" %
                             (str(stats.shape), str(self._stats.shape)))
        self._stats = stats


class MetropolisHastings(Minimiser):
    """MetropolisHastings MCMC minimisation
    """
 
    def __init__(self, name = None, burn_in = 5000, niter = 10000, thin_factor = 10):
        super(MetropolisHastings, self).__init__(name)
        self._type = MetropolisHastings
        self._burn_in = burn_in
        self._niter = niter
        self._thin_factor = thin_factor
        self._per_bin = False

    def minimise(self, fit_config, spectra_config, funct, test_statistic):
        """Minimise the docstrings!
        """
        # Choose a starting point
        current_par = [fit_config.get_par(p)._current_value \
                       for p in fit_config.get_pars()]
        step_size = [fit_config.get_par(p).step_size \
                     for p in fit_config.get_pars()]
        current_stat, current_penalty = funct(*current_par)
        
        acceptance = 0
        sample = [] # could reserve points
        best_stat = None
        best_point = None

        for i in range(self._niter):

            # Random jump
            test_par = numpy.random.normal(current_par, step_size)
            test_stat, test_penalty = funct(*test_par)
            
            # Test statistic should always be minimised
            # LL methods should return -ve stat
            if test_stat < current_stat or \
               (numpy.exp(current_stat - test_stat) > numpy.random.uniform()):
                current_par = test_par
                current_stat = test_stat
                acceptance += 1
                accepted = True
            else:
                accepted = False

            # Add to sample according to thinning and burn in
            if i > self._burn_in and not (i % self._thin_factor):
                sample.append(current_par)
                if current_stat < best_stat or best_stat is None:
                    best_stat = current_stat
                    best_point = current_par

        # Return best fit point
        # FIXME: this could either be the mean of the sample
        #        or the point in the sample with the best test_statistic
        self._best_point = best_point
        self._best_ll = best_stat
        self._error = numpy.std(sample, axis=0)
        # TODO: should acceptance be logged only after burn in?
        self._acceptance = float(acceptance) / self._niter
        return numpy.mean(sample, axis=0)
