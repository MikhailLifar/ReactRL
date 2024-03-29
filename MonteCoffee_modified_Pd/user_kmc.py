"""Defines the NeighborKMC class used to run a MonteCoffee simulation.

The module defines the main simulation class (NeighborKMC), which is used
to run the simulation. The main engine is found in base.kmc.

See Also
--------
Module: base.kmc

"""

from __future__ import print_function

# import copy
import os
import numpy as np
import matplotlib.pyplot as plt
import ase.io

from MonteCoffee_changed.NeighborKMC.base.kmc import NeighborKMCBase
from MonteCoffee_changed.NeighborKMC.base.logging import Log
from MonteCoffee_changed.NeighborKMC.base.basin import scale_rate_constant
from user_constants import *
from user_sites import Site
from user_events import *
# import pyclbr
import h5py


class NeighborKMC(NeighborKMCBase):
    """Controls the kMC simulation.
            
    Calls constructor of NeighborKMCBase objects, and  
    loads in the user-specified event-list in load_events().

    The variable self.evs_exec is initialized as a list to
    count the number of times each event-type is executed.  
    
    Parameters
    -----------
    system: System
        The system instance to perform the simulation on.
    parameters: dict
        Parameters used, which are dumped to the log file.
        Example: parameters = {'pCO':1E2,'T':700,'Note':'Test simulation'}
    events: list(classobj)
        A list pointing to the event classes that defines the events.
        The order of list is kept consistently throughout the simulation.
        For example, given the event classes:

        .. code-block:: python

            class AdsEvent(EventBase):
                def __init__(self):
                ...

            class DesEvent(EventBase):
                def __init__(self):
                ...

        One should define events as a list of class names as

        >>> events = [AdsEvent, DesEvent]

    rev_events: dict
        Specifying which events are reverse to each other, following the order `self.events`.
        This dict is used for temporal acceleration.
        For example, if we have an adsorption and desorption event that are each others reverse, a
        third non-reversible event, and a diffusion event that is its own reverse:

        >>> events = [AdsEvent, DesEvent, ThirdEvent, DiffusionEvent]

        Then rev_events is defined as

        >>> rev_events = {0:1,3:3}.

    Example
    --------
    Assume that we have defined a System object (system), a list of event **classes** (events), and the
    dict of reverse events (rev_events). Then a NeighborKMC object is instantiated and simulation is run as:

    .. code-block:: python

        nkmc = NeighborKMC(system=system,
                           parameters=params,
                           events=events,
                           rev_events=rev_events)
        nkmc.run_kmc()

    See Also
    ---------
    Module: base.kmc
    Module: base.basin

    """

    def __init__(self, system, parameters=None, events=None, rev_events=None):
        if events is None:
            events = []
        if rev_events is None:
            rev_events = dict()

        self.events = [ev(parameters) for ev in events]
        self.reverses = None  # Set later
        self.load_reverses(rev_events)
        self.evs_exec = np.zeros(len(self.events))
        self.system_evolution = [[] for i in range(4)]

        NeighborKMCBase.__init__(self, system=system, kmc_parameters=parameters,
                                 options_dir=os.path.dirname(os.path.abspath(__file__)))

        self.log = None
        self.stepN_CNT = 0
        self.stepNMC = 0
        self.stepSaveN = 0

        self.rescaleN = self.ne
        self.rescaleStep = 0

        self.COxO_prev_count = 0

    def reset(self):
        # log initialize
        if self.verbose:
            print('Loading logging and counters...')

        logparams = {}
        logparams.update(self.kmc_parameters)
        logparams.update({"tend": 'undefined',
                          "Nsites": self.system.Nsites,
                          "Number of events": len(self.events),
                          "Number of site-types (stypes)": len(list(set([m.stype for m in self.system.sites])))
                          })
        accelparams = {"on": self.use_scaling_algorithm, "Ns": self.Ns, "Nf": self.Nf, "ne": self.ne}
        self.log = Log(logparams, accelparams)

        # Save txt files with site information:
        with open("siteids.txt", "wb") as f2:
            np.savetxt(f2, [m.ind for m in self.system.sites])

        with open("stypes.txt", "wb") as f2:
            np.savetxt(f2, [m.stype for m in self.system.sites])

        if self.save_coverages:
            f = h5py.File('detail_site_event_evol.hdf5', 'w')
            d = f.create_dataset("time", (1,), maxshape=(None,), chunks=True, dtype='float')
            d = f.create_dataset("site", (1,), maxshape=(None,), chunks=True, dtype='int')
            d = f.create_dataset("othersite", (1,), maxshape=(None,), chunks=True, dtype='int')
            d = f.create_dataset("event", (1,), maxshape=(None,), chunks=True, dtype='int')
            f.close()

        # Initialize time and step counters
        self.stepN_CNT = 0
        self.stepNMC = 0
        self.stepSaveN = 0

        self.rescaleStep = 0

        if self.verbose:
            print('\nRunning simulation.')

    def load_reverses(self, rev_events):
        """Prepares the reverse_event dict.
               
        Method makes the dict self.reverses two sided, and performs
        a check that each event only has one reverse in the end.

        Parameters
        -----------
        rev_events: dict
            Specifying which events are reverse to each other, as described in
            the constructor of NeighborKMC.

        Raises
        -------
        Warning:
            If an reversible event has more than one reverse.

        """

        self.reverses = dict(rev_events)

        for k in rev_events:
            val = self.reverses[k]
            self.reverses[val] = k

        # Make sure each event only has one reverse
        if sorted(self.reverses.keys()) != list(set(self.reverses.keys())) or \
                sorted(self.reverses.values()) != list(set(self.reverses.values())):
            raise Warning('Error in user_kmc.NeighborKMC.load_reverses(). An event has more than one reverse.')

    def set_pressures(self, pO2, pCO):
        self.kmc_parameters['pO2'] = pO2
        self.kmc_parameters['pCO'] = pCO

    def step_forward(self, step_time):
        print('Step started')
        t0 = self.t
        while self.t - t0 < step_time:

            self.frm_step()

            # Log every self.LogSteps step.
            if self.stepN_CNT >= self.LogSteps:
                covs = self.system.get_coverages(self.Nspecies)
                # if self.verbose:
                #     print("Time : ", self.t, "\t Covs :", covs)
                self.log.dump_point(self.stepNMC, self.t, self.evs_exec, covs)

                self.times.append(self.t)
                #               self.MCstep.append(stepNMC)

                covs_cur = [s.covered for s in self.system.sites]
                self.covered.append(covs_cur)
                self.stepN_CNT = 0

            self.stepSaveN += 1

            # Save every self.SaveSteps steps.
            if self.stepSaveN == self.SaveSteps:
                self.plot_snapshot(f'./snapshots/snapshot_t({self.t:.3f}).png')
                self.save_txt()
                self.stepSaveN = 0.

            # MY CODE. Trying to rescale constants
            self.rescaleStep += 1
            if self.rescaleStep == self.rescaleN:
                scale_rate_constant(self)

            self.stepN_CNT += 1
            self.stepNMC += 1
        print('Step ended')

    def get_COxO_events_last_step(self):
        count = self.evs_exec[-1] - self.COxO_prev_count
        self.COxO_prev_count += count
        return count

    def finalize(self):
        self.log.dump_point(self.stepNMC, self.t, self.evs_exec, [])  # forgivable crutch
        self.save_txt()

    def save_txt(self):
        """Saves txt files containing the simulation data.

        Calls the behind-the-scenes save_txt() method of the base class.
        The user should add any optional writes in this method, which
        is called every self.SaveSteps steps.

        Example
        --------
        Add the following line to the end of the method:

        >>> np.savetxt("user_stype_ev.txt", self.stype_ev[0])

        """

        Log.save_txt(self)

    def run_kmc(self, tend):
        """Runs a kmc simulation.

        Method starts the simulation by initializing the log,
        initializes lists to keep track of time and step
        numbers.

        It saves information about the site-indices in `siteids.txt`,
        and the site-types in `stypes.txt`.

        While the simulation runs (self.t < self.tend),
        Monte Carlo steps are performed by calling self.frm_step().

        Every self.LogSteps, a line is added to the simulation
        log.


        """

        self.reset()

        while self.t < tend:

            self.frm_step()

            # Log every self.LogSteps step.
            if self.stepN_CNT >= self.LogSteps:
                covs = self.system.get_coverages(self.Nspecies)
                if self.verbose:
                    print("Time : ", self.t, "\t Covs :", covs)

                self.log.dump_point(self.stepNMC, self.t, self.evs_exec, covs)

                self.times.append(self.t)
                # self.MCstep.append(stepNMC)

                covs_cur = [s.covered for s in self.system.sites]
                self.covered.append(covs_cur)
                self.stepN_CNT = 0

            self.stepSaveN += 1

            # Save every self.SaveSteps steps.
            if self.stepSaveN == self.SaveSteps:
                self.save_txt()
                self.stepSaveN = 0.

                # plot snapshots
                self.plot_snapshot(f'./snapshots/snapshot_t({self.t:.3f}).png')

            self.stepN_CNT += 1
            self.stepNMC += 1

        self.finalize()

    def plot_snapshot(self, filepath):
        # TODO rewrite sites to numpy array instead of list
        m, n, _ = self.system.shape
        surface = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                surface[i][j] = self.system.sites[i * n + j].covered

        fig, ax = plt.subplots(figsize=(m * 16 / (m + n), n * 16 / (m + n)))

        where_co = np.where(surface == 1)
        where_o = np.where(surface == 2)

        marker_size = 0.8 * 16 / (m + n)
        ax.scatter(*where_co, c='r', marker='o', label='CO', s=marker_size)
        ax.scatter(*where_o, c='b', marker='o', label='O', s=marker_size)
        ax.set_title(f'Kinetic Monte Carlo simulation\nsurface state, time {self.t:.5f}')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        fig.legend(loc='outside lower center', ncol=2, fancybox=True)

        fig.savefig(filepath, dpi=400, bbox_inches='tight')
        plt.close(fig)

    # def run_steps(self, full_time, step_time):
    #     self.tend = full_time
    #
    #     self.reset()
    #
    #     while self.t < self.tend:
    #         self.step_forward(step_time)
    #
    #     self.log.dump_point(self.stepNMC, self.t, self.evs_exec)
    #     self.save_txt()

