"""Script that runs a full example of CO oxidation.
 
"""
import random

import numpy as np
from ase.io import write
from ase.build import fcc111

import sys, os
sys.path.append('/home/mikhail/RL_22_07_MicroFluidDroplets')
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from user_sites import Site
from user_system import System

from user_kmc import NeighborKMC
from user_events import (OAdsEvent, ODesEvent,
                         COAdsEvent, CODesEvent, COOxEvent, CODiffEvent, ODiffEvent)


def periodic_policy(sim: NeighborKMC):
    pressures = {'CO': (4.e-5, 10.e-5), 'O2': (10.e-5, 4.e-5), }
    dt = 10
    for i in range(10):
        for j in range(2):
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
    T = 440.   # 300., 440.; Temperature of simulation in K
    pCO = 5.e-4  # 5.e-4; CO pressure in Pa
    pO2 = 5.e-4  # 5.e-4; O2 pressure in Pa
    a = 4.00  # Lattice Parameter (not related to DFT!)
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
    # sim.reset()
    # periodic_policy(sim)
    # sim.finalize()
    print("Simulation end time reached ! ! !")


def main():
    run_original()


if __name__ == '__main__':
    main()

