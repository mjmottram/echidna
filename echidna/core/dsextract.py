import numpy
import rat
import echidna.calc.constants as const

# Setup a list of cut functions that are commonly used
# The simpler ones here might be better encoded as lambda functions
# Love closures, not docstrings.
def ntuple_cut_ev_index(index):
    return lambda entry: entry.evIndex == index
def ntuple_cut_mc():
    return lambda entry: entry.mc == 1
def ntuple_cut_valid_fit():
    return lambda entry: entry.scintFit == 1 and entry.fitValid == 1
def ntuple_cut_reco_energy(energy):
    return lambda entry: entry.scintFit == 1 and entry.fitValid == 1 and entry.energy > energy

def root_cut_ev_index(index_):
    return lambda ds, index: ds.GetEVCount() >= index and \
        index == index_
def root_cut_mc():
    return lambda ds, index: ds.GetMC().GetMCParticleCount() > 0
def root_cut_valid_fit():
    return lambda ds, index: ds.GetEV(index).DefaultFitVertexExists() and \
        ds.GetEV(index).defaultFitVertex.GetValid()
def root_cut_reco_energy(energy):
    return lambda ds, index: ds.GetEV(index).DefaultFitVertexExists() and \
        ds.GetEV(index).GetDefaultFitVertex().ContainsEnergy() and \
        ds.GetEV(index).GetDefaultFitVertex().ValidEnergy()


# Leave some as nested functions
def ntuple_cut_mc_radius(radius):
    def call(entry):
        return (entry.mcPosx**2 + entry.mcPosy**2 + entry.mcPosz**2) < (radius**2)
    return call

def ntuple_cut_reco_radius(radius):
    def call(entry):
        return entry.scintFit==1 and \
            (entry.posx**2 + entry.posy**2 + entry.posz**2) < (radius**2)
    return call

def root_cut_mc_radius(radius):
    def call(ds, index=0):
        return ds.GetMC().GetMCParticle(0).GetPosition().Mag() < radius
    return call

def root_cut_reco_radius(radius):
    def call(ds, index=0):
        return ds.GetEV(index).DefaultFitVertexExists() and \
            ev.GetEV(index).GetDefaultFitVertex().GetPosition().Mag() < radius
    return call


def function_factory(dimension, **kwargs):
    '''Factory function that returns a dsextract class
    corresponding to the dimension (i.e. DS parameter)
    that is being extracted.

    Args:
      dimension (str): to extract from a RAT DS/ntuple file.
      kwargs (dict): to be passed to and checked by the extractor.

    Raises:
      IndexError: dimension is an unknown parameter

    Retuns:
      :class:`echidna.core.dsextract.Extractor`: Extractor object.
    '''
    kwdict = {}
    if "cuts" in kwargs:
        kwdict["cuts"] = kwargs["cuts"]
    if dimension == "energy_mc":
        return EnergyExtractMC(**kwdict)
    elif dimension == "energy_reco":
        return EnergyExtractReco(**kwdict)
    elif dimension == "energy_truth":
        return EnergyExtractTruth(**kwdict)
    elif dimension == "radial_mc":
        return RadialExtractMC(**kwdict)
    elif dimension == "radial_reco":
        return RadialExtractReco(**kwdict)
    elif dimension == "radial3_mc":
        if "outer_radius" in kwargs:
            kwdict["outer_radius"] = kwargs["outer_radius"]
        return Radial3ExtractMC(**kwdict)
    elif dimension == "radial3_reco":
        if "outer_radius" in kwargs:
            kwdict["outer_radius"] = kwargs["outer_radius"]
        return Radial3ExtractReco(**kwdict)
    else:
        raise IndexError("Unknown parameter: %s" % dimension)


class Extractor(object):
    '''Base class for extractor classes.

    Args:
      name (str): of the dimension
      cuts (list): list of additional cuts to apply

    Attributes:
      _name (str): of the dimension
      _user_cuts (list): list of additional cuts to apply
    '''

    def __init__(self, name, cuts):
        '''Initialise the class
        '''
        self._name = name
        self._user_cuts = cuts # A list of checks which can be appended
        self._ntuple_cuts = [] # A list of default cuts for Ntuple extractors
        self._root_cuts = [] # A list of default cuts for ROOT extractors

    def add_cut(self, function):
        self._user_cuts.append(function)

    def get_valid_ntuple(self, entry):
        for cut in self._ntuple_cuts + self._user_cuts:
            if not cut(entry):
                return False
        return True

    def get_valid_root(self, *args):
        for cut in self._root_cuts + self._user_cuts:
            if not cut(*args):
                return False
        return True

class EnergyExtractMC(Extractor):
    '''Quenched energy extraction methods.

    Args:
      cuts (list, options): List of additional cuts to apply.

    Attributes:
      _ntuple_cuts (list): default cuts to apply when extracting from ntuples
      _root_cuts (list): default of cuts to apply when extracting from root
    '''

    def __init__(self, cuts=[]):
        '''Initialise the class
        '''
        super(EnergyExtractMC, self).__init__("energy_mc", cuts)
        self._ntuple_cuts = [ntuple_cut_mc()]
        self._root_cuts = [root_cut_mc()]

    def get_value_root(self, ds, index=0):
        '''Get energy value from a DS::MC

        Args:
          mc (:class:`RAT.DS.MC`): entry

        Returns:
          float: True quenched energy
        '''
        return ds.GetMC().GetScintQuenchedEnergyDeposit()

    def get_value_ntuple(self, entry):
        '''Get energy value from an ntuple MC

        Args:
          entry (:class:`ROOT.TChain`): chain entry

        Returns:
          float: True quenched energy
        '''
        return entry.mcEdepQuenched


class EnergyExtractReco(Extractor):
    '''Reconstructed energy extraction methods.

    Args:
      cuts (list, options): List of additional cuts to apply.

    Attributes:
      _ntuple_cuts (list): default cuts to apply when extracting from ntuples
      _root_cuts (list): default of cuts to apply when extracting from root
    '''

    def __init__(self, cuts=[]):
        '''Initialise the class
        '''
        super(EnergyExtractReco, self).__init__("energy_reco", cuts)
        self._ntuple_cuts = [ntuple_cut_reco_energy(0)]
        self._root_cuts = [root_cut_reco_energy(0)]

    def get_value_root(self, ds, index=0):
        '''Get energy value from a DS::EV

        Args:
          ev (:class:`RAT.DS.EV`): event

        Returns:
          float: Reconstructed energy
        '''
        return ds.GetEV(index).GetDefaultFitVertex().GetEnergy()

    def get_value_ntuple(self, entry):
        '''Get energy value from an ntuple EV

        Args:
          entry (:class:`ROOT.TChain`): chain entry

        Returns:
          float: Reconstructed energy
        '''
        return entry.energy


class EnergyExtractTruth(Extractor):
    '''True MC energy extraction methods.

    Args:
      cuts (list, options): List of additional cuts to apply.

    Attributes:
      _ntuple_cuts (list): default cuts to apply when extracting from ntuples
      _root_cuts (list): default of cuts to apply when extracting from root
    '''

    def __init__(self, cuts=[]):
        '''Initialise the class
        '''
        super(EnergyExtractTruth, self).__init__("energy_truth", cuts)
        self._ntuple_cuts = [ntuple_cut_mc()]
        self._root_cuts = [root_cut_mc()]

    def get_value_root(self, mc):
        '''Get energy value from a DS::MC

        Args:
          mc (:class:`RAT.DS.MC`): entry

        Returns:
          float: True energy
        '''
        return mc.GetScintEnergyDeposit()

    def get_value_ntuple(self, entry):
        '''Get energy value from an ntuple MC

        Args:
          entry (:class:`ROOT.TChain`): chain entry

        Returns:
          float: True energy
        '''
        return entry.mcEdep


class RadialExtractMC(Extractor):
    '''True radial extraction methods.

    Args:
      cuts (list, options): List of additional cuts to apply.

    Attributes:
      _ntuple_cuts (list): default cuts to apply when extracting from ntuples
      _root_cuts (list): default of cuts to apply when extracting from root
    '''

    def __init__(self, cuts=[]):
        '''Initialise the class
        '''
        super(RadialExtractMC, self).__init__("radial_mc", cuts)
        self._ntuple_cuts = [ntuple_cut_mc()]
        self._root_cuts = [root_cut_mc()]

    def get_value_root(self, mc):
        '''Get radius value from a DS::MC

        Args:
          mc (:class:`RAT.DS.MC`): event

        Returns:
          float: True radius
        '''
        return mc.GetMCParticle(0).GetPosition().Mag()

    def get_value_ntuple(self, entry):
        '''Get radius value from an ntuple MC

        Args:
          entry (:class:`ROOT.TChain`): chain entry

        Returns:
          float: True radius
        '''
        return numpy.fabs(numpy.sqrt((entry.mcPosx)**2 +
                                     (entry.mcPosy)**2 +
                                     (entry.mcPosz)**2))


class RadialExtractReco(Extractor):
    '''Reconstructed radial extraction methods.

    Args:
      cuts (list, options): List of additional cuts to apply.

    Attributes:
      _ntuple_cuts (list): default cuts to apply when extracting from ntuples
      _root_cuts (list): default of cuts to apply when extracting from root
    '''

    def __init__(self, cuts=[]):
        '''Initialise the class
        '''
        super(RadialExtractReco, self).__init__("radial_reco", cuts)
        self._ntuple_cuts = [ntuple_cut_valid_fit()]
        self._root_cuts = [root_cut_valid_fit()]

    def get_value_root(self, ev):
        '''Get radius value from a DS::EV

        Args:
          ev (:class:`RAT.DS.EV`): event

        Returns:
          float: Reconstructed radius
        '''
        return ev.GetDefaultFitVertex().GetPosition().Mag()

    def get_value_ntuple(self, entry):
        '''Get radius value from an ntuple EV

        Args:
          entry (:class:`ROOT.TChain`): chain entry

        Returns:
          float: Reconstructed radius
        '''
        return numpy.fabs(numpy.sqrt((entry.posx)**2 +
                                     (entry.posy)**2 +
                                     (entry.posz)**2))


class Radial3ExtractMC(Extractor):
    ''' True :math:`(radius/outer\_radius)^3` radial extraction methods.

    Args:
      cuts (list, options): List of additional cuts to apply.
      outer_radius (float, optional): The fixed radius used in calculating
        :math:`(radius/outer\_radius)^3`. If None then the av_radius in
        :class:`echidna.calc.constants` is used in the calculation.

    Attributes:
      _ntuple_cuts (list): default cuts to apply when extracting from ntuples
      _root_cuts (list): default of cuts to apply when extracting from root
      _outer_radius (float): The fixed radius used in calculating
        :math:`(radius/outer\_radius)^3`.
    '''

    def __init__(self, cuts=[], outer_radius=None):
        '''Initialise the class
        '''
        super(Radial3ExtractMC, self).__init__("radial3_mc", cuts)
        if outer_radius:
            self._outer_radius = outer_radius
        else:
            self._outer_radius = const._av_radius
        self._ntuple_cuts = [ntuple_cut_mc()]
        self._root_cuts = [root_cut_mc()]

    def get_value_root(self, mc):
        '''Get radius value from a DS::MC

        Args:
          mc (:class:`RAT.DS.MC`): event

        Returns:
          float: True :math:`(radius/outer\_radius)^3`
        '''
        return (mc.GetMCParticle(0).GetPosition().Mag() /
                self._outer_radius) ** 3

    def get_value_ntuple(self, entry):
        '''Get radius value from an ntuple MC

        Args:
          entry (:class:`ROOT.TChain`): chain entry

        Returns:
          float: True :math:`(radius/outer\_radius)^3`
        '''
        return (numpy.fabs(numpy.sqrt((entry.mcPosx)**2 +
                                      (entry.mcPosy)**2 +
                                      (entry.mcPosz)**2)) /
                self._outer_radius) ** 3


class Radial3ExtractReco(Extractor):
    ''' Reconstructed :math:`(radius/outer_radius)^3` radial extraction
      methods.

    Args:
      cuts (list, options): List of additional cuts to apply.
      outer_radius (float, optional): The fixed radius used in calculating
        :math:`(radius/outer\_radius)^3`. If None then the av_radius in
        :class:`echidna.calc.constants` is used in the calculation.

    Attributes:
      _ntuple_cuts (list): default cuts to apply when extracting from ntuples
      _root_cuts (list): default of cuts to apply when extracting from root
      _outer_radius (float): The fixed radius used in calculating
        :math:`(radius/outer\_radius)^3`.
    '''

    def __init__(self, cuts = [], outer_radius=None):
        '''Initialise the class
        '''
        super(Radial3ExtractReco, self).__init__("radial3_reco", cuts)
        if outer_radius:
            self._outer_radius = outer_radius
        else:
            self._outer_radius = const._av_radius
        self._ntuple_cuts = [ntuple_cut_valid_fit()]
        self._root_cuts = [root_cut_valid_fit()]

    def get_value_root(self, ev):
        '''Get radius value from a DS::EV

        Args:
          ev (:class:`RAT.DS.EV`): event

        Returns:
          float: Reconstructed :math:`(radius/outer\_radius)^3`
        '''
        return (ev.GetDefaultFitVertex().GetPosition().Mag() /
                self._outer_radius) ** 3

    def get_value_ntuple(self, entry):
        '''Get radius value from an ntuple EV

        Args:
          entry (:class:`ROOT.TChain`): chain entry

        Returns:
          float: Reconstructed :math:`(radius/outer\_radius)^3`
        '''
        return (numpy.fabs(numpy.sqrt((entry.posx)**2 +
                                      (entry.posy)**2 +
                                      (entry.posz)**2)) /
                self._outer_radius) ** 3
