import sys
import os
import random
import json

# import numpy as np
import matplotlib.pyplot as plt
import warnings

import h5py
import numpy as np

from ase.build import fcc111

# TODO: This 'append' statements don't look well
sys.path.append(f'{os.path.expanduser("~")}/RL_22_07_MicroFluidDroplets')
OPTIONS_PATH = f'{os.path.expanduser("~")}/RL_22_07_MicroFluidDroplets/repos/MonteCoffee_modified_Pd'
sys.path.append(OPTIONS_PATH)

from MonteCoffee_changed.NeighborKMC.base.kmc import NeighborKMCBase
from MonteCoffee_changed.NeighborKMC.base.logging import Log
from MonteCoffee_modified_Pd.user_sites import Site
from MonteCoffee_modified_Pd.user_system import System
from MonteCoffee_modified_Pd.user_events import (OAdsEvent, ODesEvent,
                                                 COAdsEvent, CODesEvent, COOxEvent, CODiffEvent, ODiffEvent,
                                                 PD_EV_CONSTANTS)
from .BaseModel import *


class MCKMCModel(BaseModel, NeighborKMCBase):
    """
    This class substitutes NeighborKMC class from user_kmc.py
    """

    names = {'input': ['O2', 'CO'], 'output': ['CO2_rate', 'O2_Pa', 'CO_Pa', 'CO2_count'], }

    bottom = {'input': dict(), 'output': dict(), }
    top = {'input': dict(), 'output': dict(), }
    bottom['input']['CO'] = bottom['input']['O2'] = 0.
    bottom['output']['CO_Pa'] = bottom['output']['O2_Pa'] = 0.
    bottom['output']['CO2_rate'] = bottom['output']['CO2_count'] = 0.  # count is a bad option because it depends on the size and
    # top['input']['CO'] = top['input']['O2'] = 3.e+3

    model_name = 'KMC_CO_O2_Pt'

    def __init__(self,
                 surf_shape,
                 logDir=None,
                 snapshotPeriod=None,
                 diffusion_level=0.,
                 OORepLim=0.3,
                 OCORepLim=1.1,
                 init_covs=None,
                 **params):
        """
        The Model is based on the NeighborKMC class from Pt(111) example from MonteCoffee package

        :param surf_shape:
        :param logDir:
        :param parameters:
        """
        BaseModel.__init__(self, params)
        if diffusion_level > 5.e-2:
            file_id = f'{diffusion_level + 1.e-6:.2f}'
            with open(f'./data/PdDynamicAdvParams_diff({file_id}).txt') as fread:
                PD_EV_CONSTANTS.update(json.load(fread))  # TODO crutch parameterization, prune to errors
            self.events_clss = [COAdsEvent, CODesEvent, OAdsEvent, ODesEvent, CODiffEvent, ODiffEvent, COOxEvent]
            self.reverse_events_dict = {0: 1, 2: 3, 4: 4, 5: 5}
        else:
            self.events_clss = [COAdsEvent, CODesEvent, OAdsEvent, ODesEvent, COOxEvent]
            self.reverse_events_dict = {0: 1, 2: 3}
        PD_EV_CONSTANTS.update({'OORepLim': OORepLim, 'OCORepLim': OCORepLim})
        self.kmc_parameters_dict = {'pCO': 1.e-4, 'pO2': 1.e-4, 'T': 440.,
                                    'Name': 'COOx Pt(111) reaction simulation',
                                    'reverses ': self.reverse_events_dict,
                                    'Events': self.events_clss}

        for p in params:
            if p in self.kmc_parameters_dict:
                self.kmc_parameters_dict[p] = params[p]

        random.seed(0)
        # INITIALIZE SYSTEM OBJECT
        a = 4.00  # Lattice Parameter (not related to DFT!) # TODO is it relevant for Pd?
        neighbour_cutoff = a / np.sqrt(2.) + 0.05  # Nearest neighbor cutoff
        Pt_surface_ase_obj = fcc111("Pd", a=a, size=surf_shape)
        # Create a site for each surface-atom:
        Nsites = len(Pt_surface_ase_obj)
        if init_covs is None:
            sites = [Site(stype=0, covered=0, ind=i) for i in range(Nsites)]
        else:
            init_site_covs = np.random.choice(np.arange(len(init_covs)), size=Nsites, p=init_covs)
            sites = [Site(stype=0, covered=cov, ind=i) for i, cov in enumerate(init_site_covs)]
        # Instantiate a system, events, and simulation.
        system = System(atoms=Pt_surface_ase_obj, sites=sites, shape=surf_shape)
        # Set the global neighborlist based on distances:
        system.set_neighbors(neighbour_cutoff, pbc=True)

        # EVENTS
        self.events = [ev(self.kmc_parameters_dict) for ev in self.events_clss]
        self.reverses = None  # Set later
        self.load_reverses(self.reverse_events_dict)
        self.evs_exec = np.zeros(len(self.events))
        # self.system_evolution = [[] for i in range(4)]

        # SIMULATION
        NeighborKMCBase.__init__(self, system=system, kmc_parameters=self.kmc_parameters_dict,
                                 options_dir=OPTIONS_PATH)
        self.timeOffset = 0.  # needed to account for difference between PC and KMC time

        self.size = np.prod(surf_shape)
        self.params['surf_shape'] = 'x'.join(map(str, surf_shape))

        self.COxO_prev_count = 0

        # LIMITS
        for g in 'CO', 'O2':
            if f'{g}_bottom' in params:
                self.bottom['input'][g] = self.bottom['output'][f'{g}_Pa'] = params[f'{g}_bottom']

        self.top['input']['O2'] = self.top['output']['O2_Pa'] = params['O2_top']
        self.top['input']['CO'] = self.top['output']['CO_Pa'] = params['CO_top']
        self.top['output']['CO2_rate'] = params['CO2_rate_top']  # empirically chosen
        self.top['output']['CO2_count'] = params['CO2_count_top']  # empirically chosen
        self.fill_limits()

        _, covCO, covO = self.system.get_coverages(self.Nspecies)
        self.plot = {'thetaCO': covCO, 'thetaO': covO}

        self.add_info = self.get_add_info()

        self.logDir = logDir
        self.saveLog = self.logDir is not None

        self.log = None
        self.stepN_CNT = 0
        self.stepNMC = 0
        self.stepSaveN = 0

        self.snapshotPeriod = snapshotPeriod
        assert (self.snapshotPeriod is None) or (self.logDir is not None)
        self.snapshotTime = 0.

        # self.rescaleN = self.ne
        # self.rescaleStep = 0

    def get_add_info(self):
        s = f'Model name: {self.model_name}\n' + ('-' * 10) + '\n'
        for name in self.params:
            s += f'{name}: {self[name]}\n'
        return s

    def assign_constants(self, **kw):
        for k, v in kw.items():
            self.params[k] = v

    def new_values(self):
        for p, v in self.params.items():
            if p in self.kmc_parameters_dict:
                self.kmc_parameters_dict[p] = v

    def assign_and_eval_values(self, **kw):
        self.assign_constants(**kw)
        self.new_values()
        self.add_info = self.get_add_info()

    def plotSnapshot(self, filepath):
        # TODO rewrite sites to numpy array instead of list
        m, n, _ = self.system.shape
        surface = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                surface[i][j] = self.system.sites[i * n + j].covered

        filename, _ = os.path.splitext(filepath)
        np.save(f'{filename}.npy', surface)

        fig, ax = plt.subplots(figsize=(m * 16 / (m + n), n * 16 / (m + n)))

        where_co = np.where(surface == 1)
        where_o = np.where(surface == 2)
        covs = self.system.get_coverages(self.Nspecies)

        marker_size = 200. * 16 / (m + n)
        ax.scatter(*where_co, c='r', marker='o', label='CO', s=marker_size)
        ax.scatter(*where_o, c='b', marker='o', label='O', s=marker_size)
        ax.set_title(f'Kinetic Monte Carlo simulation, surface state\n'
                     f'time: {self.t:.5f}; covs: free {covs[0]:.2f}, O {covs[2]:.2f}, CO {covs[1]:.2f}')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        fig.legend(loc='outside lower center', ncol=2, fancybox=True)

        fig.savefig(filepath, dpi=400, bbox_inches='tight')
        plt.close(fig)

    def update(self, data_slice, delta_t, save_for_plot=False):
        # set pressures
        pO2, pCO = 1.e-4 * data_slice  # TODO DO NOT FORGET THIS
        if abs(self.kmc_parameters['pO2'] - pO2) + abs(self.kmc_parameters['pCO'] - pCO) > 1.e-10:
            self.kmc_parameters['pO2'] = pO2
            self.kmc_parameters['pCO'] = pCO
            for e in self.events:
                e.recompute()
            self.frm_init()

        covs = self.system.get_coverages(self.Nspecies)

        # perform time step
        t0 = self.t
        while self.t + self.timeOffset - t0 < delta_t:
            kmc_dt = self.frm_step(throw_exception=False, tstep_up_bound=delta_t)
            covs = self.system.get_coverages(self.Nspecies)

            if covs[2] == 1.:
                warnings.warn('The surface is completely occupied by oxygen')

            if self.saveLog:

                # LOGGING. SHOULD BE INSIDE WHILE LOOP
                # Log every self.LogSteps step.
                if self.stepN_CNT >= self.LogSteps:

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

                self.stepN_CNT += 1
                self.stepNMC += 1

            # snapshot every snapshotPeriod s
            if self.snapshotPeriod is not None:
                self.snapshotTime += kmc_dt
                if self.snapshotTime > self.snapshotPeriod:
                    self.plotSnapshot(f'{self.logDir}/snapshot_t({self.t}).png')
                    self.snapshotTime = 0.

        self.timeOffset = self.t + self.timeOffset - t0 - delta_t

        if self.t - t0 > 0.1 * delta_t:
            if self.t - t0 > delta_t:
                warnings.warn('!!!Simulation step is too large!!!')
            else:
                warnings.warn('Simulation step is quite large')

        # # TO SEE PROCESS IS ALIVE
        # if np.random.uniform() > 0.95:
        #     print(self.t)

        # NOTE: SINCE THE NUMBER OF PT ATOMS ARE CURRENTLY SMALL
        # AND KMC ARE BASED ON RANDOMNESS
        # THE 'RATE' VALUES ARE OF BIG RANDOM DEVIATION
        # get CO2 formation rate (number of reactions / atom / delta_t)

        if save_for_plot:
            _, self.plot['thetaCO'], self.plot['thetaO'] = covs

        count = self.evs_exec[-1] - self.COxO_prev_count
        self.COxO_prev_count += count
        rate = count / self.size / delta_t
        self.model_output = np.array([rate, data_slice[0], data_slice[1], count])
        return self.model_output

    def reset(self):
        if self.saveLog:
            if self.log:
                self.finalize()

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
            self.log = Log(logparams, accelparams, logs_fold_path=self.logDir)

            self.log.write_line('Rates\n')
            for e in self.events:
                self.log.write_line(f'{type(e)}: {e.get_rate(self.system, 0, 0)}\n')

            # Save txt files with site information:
            foldpath = self.logDir
            np.savetxt(f'{foldpath}/time.txt', [])
            np.savetxt(f'{foldpath}/coverages.txt', [])
            np.savetxt(f'{foldpath}/evs_exec.txt', [])
            np.savetxt(f'{foldpath}/mcstep.txt', [])

            with open(f'{foldpath}/siteids.txt', 'wb') as f2:
                np.savetxt(f2, [m.ind for m in self.system.sites])

            with open(f'{foldpath}/stypes.txt', 'wb') as f2:
                np.savetxt(f2, [m.stype for m in self.system.sites])

            if self.save_coverages:
                f = h5py.File(f'{foldpath}/detail_site_event_evol.hdf5', 'w')
                d = f.create_dataset("time", (1,), maxshape=(None,), chunks=True, dtype='float')
                d = f.create_dataset("site", (1,), maxshape=(None,), chunks=True, dtype='int')
                d = f.create_dataset("othersite", (1,), maxshape=(None,), chunks=True, dtype='int')
                d = f.create_dataset("event", (1,), maxshape=(None,), chunks=True, dtype='int')
                f.close()

            # Initialize time and step counters
            self.stepN_CNT = 0
            self.stepNMC = 0
            self.stepSaveN = 0

            # self.rescaleStep = 0

            if self.verbose:
                print('\nRunning simulation.')

        else:
            warnings.warn('Logging is turned off')

        for s in self.system.sites:
            s.covered = 0  # turn all sites empty

        self.snapshotTime = 0.
        self.COxO_prev_count = 0

        NeighborKMCBase.reset(self)

    def load_reverses(self, rev_events):
        """Prepares the reverse_event dict.
        Method makes the dict self.reverses two sided, and performs
        a check that each event only has one reverse in the end.
        """

        self.reverses = dict(rev_events)
        for k in rev_events:
            val = self.reverses[k]
            self.reverses[val] = k
        # Make sure each event only has one reverse
        if sorted(self.reverses.keys()) != list(set(self.reverses.keys())) or \
                sorted(self.reverses.values()) != list(set(self.reverses.values())):
            raise Warning('Error in user_kmc.NeighborKMC.load_reverses(). An event has more than one reverse.')

    def finalize(self):
        self.log.dump_point(self.stepNMC, self.t, self.evs_exec, [])
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

        Log.save_txt(self, foldpath=self.logDir)
