"""Contains all user-defined event types.

All user-defined events are defined here, which
must be derived from the parent class EventBase.  

See also
---------
Module: base.events for documentation about the methods possible(), get_rate(), and do_event().

"""

import os, os.path
import json

import numpy as np
from MonteCoffee_changed.NeighborKMC.base.events import EventBase
from user_entropy import get_Zvib, get_Z_CO, get_Z_O2

from user_constants import mCO, mO2, Asite, modes_COads, \
    modes_Oads, modes_TS_COOx, modes_COgas, modes_O2gas, kB, eV2J, s0CO, s0O, h

from user_energy import EadsCO, EadsO, get_Ea, \
    get_repulsion, EdiffCO, EdiffO


PD_EV_CONSTANTS = {
    'Asite': Asite,
    's0CO': s0CO,
    's0O': s0O,
    'EadsCO': EadsCO,
    'EadsO': EadsO,
    'EdiffCO': EdiffCO,
    'EdiffO': EdiffO,
    'Ea_const': 0.168 + 0.47238,
}
# TODO ugly structured code
with open(f'{os.path.expanduser("~")}/RL_22_07_MicroFluidDroplets/data/PdDynamicAdvParams_diff(0.00).txt', 'r') as fread:
    PD_EV_CONSTANTS.update(json.load(fread))


class COAdsEvent(EventBase):
    """CO adsorption event class.
    The event is CO(g) + * -> CO*.
    The event is possible if the site is empty.  
    The rate comes from collision theory.  
    Performing the event adds a CO to the site.

    """

    def __init__(self, params):
        EventBase.__init__(self, params)

    def possible(self, system, site, other_site):
        if system.sites[site].covered == 0:
            return True
        else:
            return False

    def get_rate(self, system, site, other_site):
        R = (PD_EV_CONSTANTS['s0CO'] * self.params['pCO']) / (PD_EV_CONSTANTS['Asite'] * np.sqrt(2. * np.pi * mCO * kB * eV2J * self.params['T']))
        return self.alpha * R  # alpha important for temporal acceleration.

    def do_event(self, system, site, other_site):
        system.sites[site].covered = 1

    def get_involve_other(self):
        return False 


class CODesEvent(EventBase):
    """CO desorption event class.
    The event is CO\* -> CO(g) + \*.
    The event is possible if the site is CO-covered.  
    The rate comes from the forward rate and the
    equilibrium constant.  
    Performing the event removes a CO from the site.

    """

    def __init__(self, params):
        EventBase.__init__(self, params)
        self.dZ = 0.
        self.recompute()

    def recompute(self):
        Zads = get_Zvib(self.params['T'], modes_COads)
        Zgas = get_Z_CO(self.params['T'], self.params['pCO'])
        self.dZ = Zads / Zgas

    def possible(self, system, site, other_site):
        # If site is covered with CO (species no. 1).
        if system.sites[site].covered == 1:
            return True
        else:
            return False

    def get_rate(self, system, site, other_site):
        Ncovs = system.get_ncovs(site)
        ECO = max(PD_EV_CONSTANTS['EadsCO'] - get_repulsion(1, Ncovs, 0), 0)
        K = self.dZ * np.exp(ECO/(kB * self.params['T']))
        RF = (self.params['pCO'] * PD_EV_CONSTANTS['s0CO']) / (PD_EV_CONSTANTS['Asite'] * np.sqrt(2. * np.pi * mCO * kB * eV2J * self.params['T']))
        R = self.alpha * RF / K
        return R 

    def do_event(self, system, site, other_site):
        system.sites[site].covered = 0

    def get_involve_other(self):
        return False 


class OAdsEvent(EventBase):
    """Oxygen adsorption event class.
    The event is O2(g) + 2* -> 2O*.
    The event is possible if two neighbor sites are empty.  
    The rate comes from collision theory and time 0.5 because of two atoms produced.  
    Performing the event adds O to the two empty neighbor sites.  
    
    """

    def __init__(self, params):
        EventBase.__init__(self, params)
        self.active = True  # TODO crutch to turn OAds when thetaO > threshold

    def possible(self, system, site, other_site):
        if system.sites[site].covered == 0 and system.sites[other_site].covered == 0:
            return self.active
        else:
            return False

    def get_rate(self, system, site, other_site):
        R = (PD_EV_CONSTANTS['s0O'] * self.params['pO2']) / (PD_EV_CONSTANTS['Asite'] * np.sqrt(2. * np.pi * mO2 * kB * eV2J * self.params['T']))
        return self.alpha * R

    def do_event(self, system, site, other_site):
        # Cover it with O, which is species number 2.
        system.sites[site].covered = 2
        system.sites[other_site].covered = 2

    def get_involve_other(self):
        return True


class ODesEvent(EventBase):
    """Oxygen adsorption event class.
    The event is 2O* -> O2(g) + 2*.
    The event is possible if two neighbor 
    sites are covered with species 2 (O).
    The rate comes from the forward rate and the
    equilibrium constant.   
    Performing the event empties the two sites by setting
    covered to 0.

    """

    def __init__(self, params):
        EventBase.__init__(self, params)
        self.dZ = 0.
        self.recompute()

    def recompute(self):
        Zads = get_Zvib(self.params['T'], modes_Oads)  # fac 2?
        Zgas = get_Z_O2(self.params['T'], self.params['pO2'])
        self.dZ = Zads / Zgas

    def possible(self, system, site, other_site):
        if system.sites[site].covered == 2 and system.sites[other_site].covered == 2:
            return True
        else:
            return False

    def get_rate(self, system, site, other_site):
        Ncovs = system.get_ncovs(site)
        Ncovsother = system.get_ncovs(other_site)
        E2O = max(2. * PD_EV_CONSTANTS['EadsO'] - get_repulsion(2, Ncovs, 0) - get_repulsion(2, Ncovsother, 0), 0.)
        Rf = (PD_EV_CONSTANTS['s0O'] * self.params['pO2']) / (PD_EV_CONSTANTS['Asite'] * np.sqrt(2. * np.pi * mO2 * kB * eV2J * self.params['T']))
        K = self.dZ * np.exp(E2O / (kB * self.params['T']))
        return self.alpha * Rf / K

    def do_event(self, system, site, other_site):
        system.sites[site].covered = 0
        system.sites[other_site].covered = 0

    def get_involve_other(self):
        return True


class CODiffEvent(EventBase):
    """CO diffusion event class.
    The event is CO* + * -> * + CO*.
    The event is possible if the site is CO-covered,
    and the neighbor site is empty.  
    The rate comes from transition state theory.  
    Performing the event removes a CO from the site,
    and adds it to the neighbor site.  

    """

    def __init__(self, params):
        EventBase.__init__(self, params)
        self.diffev = True
        self.dZ = 0.
        self.recompute()

    def recompute(self):
        Zini = get_Zvib(self.params['T'], modes_COads)
        Zts = np.sqrt(Zini)
        self.dZ = Zts/Zini

    def possible(self, system, site, other_site):
        if (system.sites[site].covered == 1 and system.sites[other_site].covered == 0) or \
                (system.sites[site].covered == 0 and system.sites[other_site].covered == 1):
            return True
        else:
            return False

    def get_rate(self, system, site, other_site):
        EadsCO_ = PD_EV_CONSTANTS['EadsCO']
        if system.sites[site].covered == 1:
            Ncovs = [system.sites[n].covered for n in system.neighbors[site]]
            E = max(0., EadsCO_ - get_repulsion(1, Ncovs, 0))
            system.sites[site].covered = 0
            system.sites[other_site].covered = 1
            Nothercovs = [system.sites[n].covered for n in system.neighbors[other_site]]
            Eother = max(0, EadsCO_ - get_repulsion(1, Nothercovs, 0))
            system.sites[site].covered = 1 
            system.sites[other_site].covered = 0
            dE = max(E - Eother, 0.)
        else:
            Ncovs = [system.sites[n].covered for n in system.neighbors[other_site]]
            E = max(0., EadsCO_ - get_repulsion(1, Ncovs, 0))
            system.sites[site].covered = 1
            system.sites[other_site].covered = 0
            Nothercovs = [system.sites[n].covered for n in system.neighbors[site]]
            Eother = max(0, EadsCO_ - get_repulsion(1, Nothercovs, 0))
            system.sites[site].covered = 0 
            system.sites[other_site].covered = 1
            dE = max(E - Eother, 0.)

        Eact = dE + PD_EV_CONSTANTS['EdiffCO']
        R = self.alpha * self.dZ * np.exp(-Eact / (kB * self.params['T'])) * kB * self.params['T'] / (h)
        return R

    def do_event(self, system, site, other_site):
        old_site = system.sites[site].covered
        old_othersite = system.sites[other_site].covered
        system.sites[site].covered = old_othersite
        system.sites[other_site].covered = old_site

    def get_involve_other(self):
        return True


class ODiffEvent(EventBase):
    """O diffusion event class.
    The event is O* + * -> * + O*.
    The event is possible if the site is O-covered,
    and the neighbor site is empty.  
    The rate comes from transition state theory.  
    Performing the event removes a O from the site,
    and adds it to the other site.

    """

    def __init__(self, params):
        EventBase.__init__(self, params)
        self.diffev = True
        self.dZ = 0.
        self.recompute()

    def recompute(self):
        Zini = get_Zvib(self.params['T'], modes_Oads)
        Zts = np.sqrt(Zini)
        self.dZ = Zts / Zini

    def possible(self, system, site, other_site):
        if (system.sites[site].covered == 2 and system.sites[other_site].covered == 0) \
                or (system.sites[site].covered == 0 and system.sites[other_site].covered == 2):
            return True
        else:
            return False

    def get_rate(self, system, site, other_site):
        EadsO_ = PD_EV_CONSTANTS['EadsO']
        if system.sites[site].covered == 2:
            Ncovs = [system.sites[n].covered for n in system.neighbors[site]]
            E = max(0., EadsO_ - get_repulsion(2, Ncovs, 0))
            system.sites[site].covered = 0
            system.sites[other_site].covered = 2
            Nothercovs = [system.sites[n].covered for n in system.neighbors[other_site]]
            Eother = max(0, EadsO_ - get_repulsion(2, Nothercovs, 0))
            system.sites[site].covered = 2 
            system.sites[other_site].covered = 0
            dE = max(E - Eother, 0.)
        else:
            Ncovs = [system.sites[n].covered for n in system.neighbors[other_site]]
            E = max(0., EadsO_ - get_repulsion(2, Ncovs, 0))
            system.sites[site].covered = 2
            system.sites[other_site].covered = 0
            Nothercovs = [system.sites[n].covered for n in system.neighbors[site]]
            Eother = max(0, EadsO_ - get_repulsion(2, Nothercovs, 0))
            system.sites[site].covered = 0 
            system.sites[other_site].covered = 2
            dE = max(E - Eother, 0.)
        Eact = dE + PD_EV_CONSTANTS['EdiffO']
        R = self.alpha * self.dZ * np.exp(-Eact / (kB * self.params['T'])) * kB * self.params['T'] / h
        return R

    def do_event(self, system, site, other_site):
        old_site = system.sites[site].covered
        old_othersite = system.sites[other_site].covered
        system.sites[site].covered = old_othersite
        system.sites[other_site].covered = old_site

    def get_involve_other(self):
        return True


class COOxEvent(EventBase):
    """CO oxidation event class.
    The event is CO* + O* -> CO2(g)+2*.
    The event is possible if the site is 
    CO-covered and the neighbor is O-covered.
    The rate comes from transition state theory.
    Performing the event removes a CO+O from the sites.

    """

    def __init__(self, params):
        EventBase.__init__(self, params)
        self.Zratio = 0.
        self.recompute()

    def recompute(self):
        Zads = get_Zvib(self.params['T'], modes_COads) * get_Zvib(self.params['T'], modes_Oads)
        Zts = get_Zvib(self.params['T'], modes_TS_COOx)
        self.Zratio = Zts / Zads

    def possible(self, system, site, other_site):
        if (system.sites[site].covered == 1 and system.sites[other_site].covered == 2) or \
              (system.sites[site].covered == 2 and system.sites[other_site].covered == 1):
            return True
        else:
            return False

    def get_rate(self, system, site, other_site):
        ECO = PD_EV_CONSTANTS['EadsCO']
        EO = PD_EV_CONSTANTS['EadsO']
        # Find the Nearest neighbor repulsion
        if system.sites[site].covered == 1:
            Ncovs_CO = [system.sites[n].covered for n in system.neighbors[site] ]
            Ncovs_O = [system.sites[n].covered for n in system.neighbors[other_site]]
        else:
            Ncovs_CO = [system.sites[n].covered for n in system.neighbors[other_site] ]
            Ncovs_O = [system.sites[n].covered for n in system.neighbors[site]]
        ECO -= get_repulsion(1, Ncovs_CO, 0)
        EO -= get_repulsion(2, Ncovs_O, 0)
        Ea = max(0., get_Ea(ECO, EO, Ea_const=PD_EV_CONSTANTS['Ea_const']))
        return self.alpha * self.Zratio * np.exp(-Ea /(kB * self.params['T'])) * kB * self.params['T'] / h

    def do_event(self, system, site, other_site):
        system.sites[site].covered = 0
        system.sites[other_site].covered = 0

    def get_involve_other(self):
        return True


