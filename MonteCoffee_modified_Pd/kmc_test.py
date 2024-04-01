"""Script that runs a full example of CO oxidation.
 
"""
import random
import json

import numpy as np
from scipy.optimize import fsolve

from ase.io import write
from ase.build import fcc111

import sys, os
sys.path.append(f'{os.path.expanduser("~")}/RL_22_07_MicroFluidDroplets')
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from user_sites import Site
from user_system import System

from user_kmc import NeighborKMC
from user_events import (OAdsEvent, ODesEvent,
                         COAdsEvent, CODesEvent, COOxEvent, CODiffEvent, ODiffEvent,
                         PD_EV_CONSTANTS)


def periodic_policy(sim: NeighborKMC):
    # pressures = {'CO': (4.e-5, 10.e-5), 'O2': (10.e-5, 4.e-5), }
    pressures = {'CO': (0.887 * 1.e-4, 0.83 * 1.e-4, 0.0528 * 1.e-4, 0.0301 * 1.e-4, ),
                 'O2': (0.0867 * 1.e-4, 0.112 * 1.e-4, 0.844 * 1.e-4, 0.919 * 1.e-4, ), }  # dynamic advantage policy
    dt = 10.
    for i in range(10):
        for j in range(4):
            sim.set_pressures(pressures['O2'][j], pressures['CO'][j])
            sim.step_forward(dt)
            rs = np.zeros(len(sim.events))
            for i0, ev in enumerate(sim.events):
                average_r = 0.
                for i1, s in enumerate(sim.system.sites):
                    NNCur = sim.system.neighbors[i1]
                    rs_site = [ev.get_rate(sim.system, i1, other_s) for other_s in NNCur if ev.possible(sim.system, i1, other_s)]
                    if rs_site:
                        average_r = average_r * (1 - 1/(i1 + 1)) + (1/(i1 + 1)) * np.mean(rs_site)
                rs[i0] = average_r
            print(rs)
            print(f'CO2 out: {sim.get_COxO_events_last_step()}')


def run_original():
    tend = 4.  # End time of simulation (s)
    T = 800.   # Temperature of simulation in K
    pCO = 2E3  # CO pressure in Pa
    pO2 = 1E3  # O2 pressure in Pa
    a = 4.00  # Lattice Parameter (not related to DFT!)
    neighbour_cutoff = a / np.sqrt(2.) + 0.05  # Nearest neighbor cutoff
    # Clear up old output files.
    # ------------------------------------------
    np.savetxt("time.txt", [])
    np.savetxt("coverages.txt", [])
    np.savetxt("evs_exec.txt", [])
    np.savetxt("mcstep.txt", [])

    os.system("rm detail_site_event_evol.hdf5")

    # Define the sites from ase.Atoms.
    # ------------------------------------------
    size = (5, 5, 1)
    Pt_surface_ase_obj = fcc111("Pt", a=a, size=size)
    Pt_surface_ase_obj.write('surface.traj')
    # Create a site for each surface-atom:
    sites = [Site(stype=0, covered=0, ind=i) for i in range(len(Pt_surface_ase_obj))]
    # Instantiate a system, events, and simulation.
    # ---------------------------------------------
    p = System(atoms=Pt_surface_ase_obj, sites=sites)

    # Set the global neighborlist based on distances:
    p.set_neighbors(neighbour_cutoff, pbc=True)

    events = [COAdsEvent, CODesEvent, OAdsEvent, ODesEvent,
              CODiffEvent, ODiffEvent,
              COOxEvent]
    # Specify what events are each others' reverse.
    reverse_events = {0: 1, 2: 3,
                      4: 4, 5: 5
                      }

    parameters = {'pCO': pCO, 'pO2': pO2, 'T': T,
                  'Name': 'COOx Pt(111) reaction simulation',
                  'reverses': reverse_events,
                  'Events': events}

    # Instantiate simulator object.
    random.seed(10)  # needed to get fully reproducible result
    sim = NeighborKMC(system=p,
                      parameters=parameters,
                      events=events,
                      rev_events=reverse_events)

    # Run the simulation.
    sim.run_kmc(tend)  # previously used method
    print("Simulation end time reached ! ! !")


def runDynamicAdvParameters():
    """Runs the test of A adsorption and desorption over a surface.

    First, constants are defined and old output files cleared.
    Next, the sites, events, system and simulation objects
    are loaded, and the simulation is performed.

    Last, the results are read in from the generated.txt files,
    and plotted using matplotlib.

    """
    # Define constants.
    # ------------------------------------------
    T = 440.   # 440.; Temperature of simulation in K
    pCO = 1.e-4  # 5.e-4; CO pressure in Pa
    pO2 = 1.e-4  # 5.e-4; O2 pressure in Pa
    a = 4.00  # Lattice Parameter (not related to DFT!); What is lattice parameter for Pd?
    neighbour_cutoff = a / np.sqrt(2.) + 0.05  # Nearest neighbor cutoff; arbitrary
    # Clear up old output files.
    # ------------------------------------------
    np.savetxt("time.txt", [])
    np.savetxt("coverages.txt", [])
    np.savetxt("evs_exec.txt", [])
    np.savetxt("mcstep.txt", [])
 
    os.system("rm detail_site_event_evol.hdf5")

    # Define the sites from ase.Atoms.
    # ------------------------------------------
    shape = (5, 5, 1)
    Pt_surface_ase_obj = fcc111("Pt", a=a, size=shape)
    Pt_surface_ase_obj.write('surface.traj')
    # Create a site for each surface-atom:
    random.seed(0)
    sites = [Site(stype=0, covered=0, ind=i) for i in range(len(Pt_surface_ase_obj))]
    # Instantiate a system, events, and simulation.
    # ---------------------------------------------
    p = System(atoms=Pt_surface_ase_obj, sites=sites, shape=shape)

    # Set the global neighborlist based on distances:
    p.set_neighbors(neighbour_cutoff, pbc=True)
    
    events = [COAdsEvent, CODesEvent, OAdsEvent, ODesEvent,
              CODiffEvent, ODiffEvent,
              COOxEvent]
    # Specify what events are each others' reverse.
    reverse_events = {0: 1, 2: 3, 4: 4, 5: 5}

    parameters = {'pCO': pCO, 'pO2': pO2, 'T': T,
                  'Name': 'COOx Pt(111) reaction simulation',
                  'reverses': reverse_events,
                  'Events': events}

    # Instantiate simulator object.
    random.seed(10)  # needed to get fully reproducible result
    sim = NeighborKMC(system=p,
                      parameters=parameters,
                      events=events,
                      rev_events=reverse_events)

    # # get rates
    # with open('PdDynamicAdvParams.txt', 'r') as fread:
    #     PD_EV_CONSTANTS.update(json.load(fread))
    # for e in sim.events:
    #     print(type(e), e.get_rate(p, 0, 0))

    # restore energies given rates
    evs_order = [
        'COAds', 'CODes', 'OAds', 'ODes',
        'CODiff', 'ODiff', 'COOx'
    ]

    diffusion_target = 1.e-7
    rates_target = {
        'COAds': 0.17582,
        'CODes': 0.1,
        'OAds': 0.074495,
        'ODes': 1.e-6,
        'CODiff': diffusion_target,
        'ODiff': diffusion_target,  # TODO - questionable, may be better different diffusion for both species
        'COOx': 0.1,
    }

    def F(param_v, param_name, ev_type, target_rate):
        PD_EV_CONSTANTS[param_name] = param_v
        return sim.events[evs_order.index(ev_type)].get_rate(sim.system, 0, 0) - target_rate

    PD_EV_CONSTANTS['s0CO'] = fsolve(F, (0.9, ), args=('s0CO', 'COAds', rates_target['COAds']))[0]
    PD_EV_CONSTANTS['EadsCO'] = fsolve(F, (1.36, ), args=('EadsCO', 'CODes', rates_target['CODes']))[0]
    PD_EV_CONSTANTS['s0O'] = fsolve(F, (0.1, ), args=('s0O', 'OAds', rates_target['OAds']))[0]
    PD_EV_CONSTANTS['EadsO'] = fsolve(F, (1.36, ), args=('EadsO', 'ODes', rates_target['ODes']))[0]
    PD_EV_CONSTANTS['EdiffCO'] = fsolve(F, (0.8, ), args=('EdiffCO', 'CODiff', rates_target['CODiff']))[0]
    PD_EV_CONSTANTS['EdiffO'] = fsolve(F, (0.8, ), args=('EdiffO', 'ODiff', rates_target['ODiff']))[0]
    PD_EV_CONSTANTS['Ea_const'] = fsolve(F, (PD_EV_CONSTANTS['Ea_const'], ), args=('Ea_const', 'COOx', rates_target['COOx']))[0]
    print(PD_EV_CONSTANTS)
    with open(f'{os.path.expanduser("~")}/RL_22_07_MicroFluidDroplets/data/PdDynamicAdvParams_diff({diffusion_target:.2f}).txt', 'w') as fwrite:
        json.dump(PD_EV_CONSTANTS, fwrite)

    # Run the simulation.
    # sim.run_kmc(100.)
    # sim.reset()
    # periodic_policy(sim)
    # sim.finalize()
    # print("Simulation end time reached ! ! !")


def plot_res():
    pass


def main():
    # run_original()
    runDynamicAdvParameters()


if __name__ == '__main__':
    main()

