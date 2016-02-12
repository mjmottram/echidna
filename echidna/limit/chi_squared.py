import numpy


class ChiSquared(object):
    """ This class calculates the chi squared comparing "data" to
    "montecarlo".

    The calculation is based on a spectrum containing observed events
    and one containing expected events. The paradigm assumed is that the
    observed events form the "data" spectrum and the expected events
    form the "montecarlo" spectrum. A couple of different methods for
    calculating chi squared are included, as well as the option to add
    constraints via a penalty term.

    Args:
      form (str, optional): specify form of chi squared calculation to
        use
      penalty_terms (dict, optional): specify (for each penalty term)
        values for:

          * "parameter_value" (optional)
          * "sigma"

    .. note::

      Forms of chi squared include

        * "pearson"
        * "neyman"
        * "poisson_likelihood" (*default*)

    Attributes:
      _form (str): form of chi squared calculation to use
      _penalty_terms (dict): information about each penalty term
      _penalty_terms_set (bool): True if one or more penalty terms have
        been set
      _current_values (dict): Stores the current value of each named
        penalty term
    """
    def __init__(self, form="poisson_likelihood", penalty_terms=None):
        self._form = form
        if penalty_terms:
            self._penalty_terms = penalty_terms
            self._penalty_terms_set = True
        else:
            self._penalty_terms = {}
            self._penalty_terms_set = False
        self._current_values = {}

    def set_penalty_term(self, name, penalty_term):
        """ Set the value of a named penlty term

        Args:
          name (string): Name of penalty term to set
          penalty_term (dict): Specify "parameter_value" and "sigma"
            values in dict.
        """
        self._penalty_terms[name] = penalty_term
        self._penalty_terms_set = True

    def get_chi_squared_per_bin(self):
        """

        Returns:
          :class:`numpy.array`: _chi_squared_per_bin
        """
        return self._chi_squared_per_bin

    def get_chi_squared(self, observed, expected, penalty_terms=None):
        """ Calculate the chi squared comparing observed to expected.

        Args:
          observed (:class:`numpy.array`): energy spectrum of observed
            events
          expected (:class:`numpy.array`): energy spectrum of expected
            events
          penalty_terms (dict, optional): specify (for each penalty term)
            values for:

              * "parameter_value" (optional)
              * "sigma"

        .. warning:: A named penalty term defined here will overwrite
          one with the same name defined in the constructor.

        Returns:
          float: Value of chi squared calculated
        """
        # Set up penalty term
        if penalty_terms:
            if self._penalty_terms_set:
                for name, penalty_term in penalty_terms.iteritems():
                    if (self._penalty_terms.get(name) is not None):
                        _penalty_term = self._penalty_terms.get(name)
                        # overwrite existing entries
                        if (penalty_term.get("parameter_value") is not None):
                            _penalty_term["parameter_value"] = \
                                penalty_term.get("parameter_value")
                        if (penalty_term.get("sigma") is not None):
                            _penalty_term["sigma"] = penalty_term.get("sigma")
                    else:  # create new entry
                        self._penalty_terms[name] = penalty_term
            else:  # no penalty term information currently set
                self._penalty_terms = penalty_terms
                self._penalty_terms_set = True

        # Calculate chi squared
        if (self._form == "pearson"):
            chi_squared = pearson_chi_squared(observed, expected)
        elif (self._form == "neyman"):
            chi_squared = neyman_chi_squared(observed, expected)
        else:  # (self._form == "poisson_likelihood")
            ll, ll_per_bin = log_likelihood(observed, expected, per_bin=True)
            self._chi_squared_per_bin = 2.0 * ll_per_bin
            chi_squared = 2.0 * ll

        # Add penalty term(s)
        if self._penalty_terms_set:
            for name, penalty_term in self._penalty_terms.iteritems():
                value = numpy.power(penalty_term.get("parameter_value") /
                                    penalty_term.get("sigma"), 2.0)
                self._current_values[name] = value
                chi_squared += value
        return chi_squared


def pearson_chi_squared(observed, expected):
    """ Calculates Pearson's chi squared.

    .. note::

      Following the definition in `Baker and Cousins, 1984
      <http://www.sciencedirect.com/science/article/pii/0167508784900164>`_

    Args:
      observed (:class:`numpy.array`, *float*): Number of observed
        events
      expected (:class:`numpy.array`, *float*): Number of expected
        events

    Raises:
      ValueError: If arrays are different lengths.

    Returns:
      float: Calculated Pearson's chi squared
    """
    if len(observed) != len(expected):
        raise ValueError("Arrays are different lengths")
    # Chosen due to backgrounds with low rates in ROI
    epsilon = 1e-34  # Limit of zero
    total = 0
    for i in range(len(observed)):
        if expected[i] < epsilon:
            expected[i] = epsilon
        if observed[i] < epsilon:
            total += expected[i]
        else:
            total += (observed[i]-expected[i])**2/expected[i]
    return total


def neyman_chi_squared(observed, expected):
    """ Calculates Neyman's chi squared.

    .. note::

      Following the definition in `Baker and Cousins, 1984
      <http://www.sciencedirect.com/science/article/pii/0167508784900164>`_

    Args:
      observed (:class:`numpy.array`, *float*): Number of observed
        events
      expected (:class:`numpy.array`, *float*): Number of expected
        events

    Raises:
      ValueError: If arrays are different lengths

    Returns:
      float: Calculated Neyman's chi squared
    """
    if len(observed) != len(expected):
        raise ValueError("Arrays are different lengths")
    # Chosen due to backgrounds with low rates in ROI
    epsilon = 1e-34  # In the limit of zero
    total = 0
    for i in range(len(observed)):
        if observed[i] < epsilon:
            expected[i] = epsilon
        if expected[i] < epsilon:
            total += observed[i]
        else:
            total += (expected[i]-observed[i])**2/observed[i]
    return total


def log_likelihood(observed, expected, per_bin=False):
    """ Calculates the (Baker-Cousins) log likelihood.

    .. note::

      For calculation of Poisson likelihood chi squared.

    .. note::

      Following the definition in `Baker and Cousins, 1984
      <http://www.sciencedirect.com/science/article/pii/0167508784900164>`_

    Args:
      observed (:class:`numpy.array`, *float*): Number of observed
        events
      expected (:class:`numpy.array`, *float*): Number of expected
        events
      per_bin (bool, optional): If True returns
        (:obj:`total`, :obj:`ll_per_bin`), otherwise just returns
        :obj:`total`.

    Raises:
      ValueError: If arrays are different lengths.

    Returns:
      float: Calculated Neyman's chi squared
    """
    # Create chi-squared per bin array
    ll_per_bin = numpy.zeros((0))
    if len(observed) != len(expected):
        raise ValueError("Arrays are different lengths")
    # Chosen due to backgrounds with low rates in ROI
    epsilon = 1e-34  # In the limit of zero
    total = 0
    for i in range(len(observed)):
        if expected[i] < epsilon:
            expected[i] = epsilon
        if observed[i] < epsilon:
            ll = expected[i]
        else:
            ll = expected[i] - observed[i] + observed[i] *\
                numpy.log(observed[i] / expected[i])
        total += ll
        ll_per_bin = numpy.append(ll_per_bin, [ll], axis=0)
    if per_bin:
        return total, ll_per_bin
    else:
        return total


def binned_max_ll(observed, expected, per_bin=False):
    '''Calculate binned maximal log likelihood, does not apply any penalty terms.

    Args:
      observed (:class:`numpy.array`, *float*): Number of observed
        events
      expected (:class:`numpy.array`, *float*): Number of expected
        events
      per_bin (bool, optional): If True returns
        (:obj:`total`, :obj:`ll_per_bin`), otherwise just returns
        :obj:`total`.

    Raises:
      ValueError: If arrays are different lengths.

    Returns:
      float: Calculated binned max LL value
    '''
    epsilon = 1e-34
    expected[expected<epsilon] = epsilon
    binned_ll = observed * numpy.log(expected)
    binned_ll -= expected
    sum_ll = sum(binned_ll)
    if not per_bin:
        return sum_ll
    else:
        return sum_ll, binned_ll
