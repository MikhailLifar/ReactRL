"""Defines the System Class, derived from base.system module.

The System is supposed to be a singleton that
is passed to a singleton NeighborKMC object.

See Also  
---------
Module: base.system
Module: user_sites

"""

import numpy as np
from MonteCoffee.NeighborKMC.base.system import SystemBase
from scipy.spatial.distance import cdist


class System(SystemBase):
    """Class defines a collection of sites and connected atoms.
            
    Calls the base class system.py constructor, 
    sets the global neighborlist from the individual site's
    neighborlist.

    Attributes
    -----------
    atoms: ase.Atoms
        Can (optionally) be passed to connect an ASE.Atoms
        object to the system. This can be useful for visualization
        of the simulation, for example by setting the ase.Atoms tag
        according to coverages.
    sites: list(Site)
        The sites that constitute the system.

    See Also
    ---------
    Module: base.system

    """

    def __init__(self, atoms=None, sites=[]):
        SystemBase.__init__(self, atoms=atoms, sites=sites)

    def set_neighbors(self, Ncutoff, pbc=False):
        """Sets neighborlists of self.sites by distances.

        Loops through the sites and using self.atoms, the
        method adds neighbors to the sites that are within a
        neighbor-distance (Ncutoff).

        Parameters
        -----------
        Ncutoff: float
            The cutoff distance for nearest-neighbors in angstroms
        pbc: bool
            If the neighborlist should be computed with periodic boundary
            conditions. To make a direction aperiodic, introduce a vacuum larger
            than Ncutoff in this direction in self.atoms.

        Raises
        ---------
        Warning:
            If self.atoms is not set, because then distances cannot
            be used to determine neighbors.

        """
        if self.atoms is None:
            raise Warning("Tried to set neighbor-distances in user_system.set_neighbors() with self.atom = None")

        # Set the neighbor list for each site using distances.
        # ------------------------------------------

        # original
        # for i, s in enumerate(self.sites):
        #     for j, sother in enumerate(self.sites):
        #         # Length of distance vector:
        #         dcur = self.atoms.get_distance(s.ind, sother.ind, mic=pbc)
        #         # If the site is a neighbor:
        #         if dcur < Ncutoff and j != i:
        #             s.neighbors.append(j)

        # 2x faster
        # for i, s in enumerate(self.sites[:-1]):
        #     for j, s_other in enumerate(self.sites[i+1:]):
        #         # Length of distance vector:
        #         d_cur = self.atoms.get_distance(s.ind, s_other.ind, mic=pbc)
        #         # If the site is a neighbor:
        #         if d_cur < Ncutoff:
        #             s.neighbors.append(j + i + 1)
        #             s_other.neighbors.append(i)

        # powered 1, does not work
        # posns = self.atoms.get_positions()
        # mask = cdist(posns, posns) < Ncutoff
        # idxs = np.arange(0, len(self.sites))
        # for i in idxs:
        #     mask[i, i] = False
        #     neighbors_idxs = idxs[mask[i, :]]
        #     for j in neighbors_idxs:
        #         self.sites[i].neighbors.append(j)

        # powered 2, this works! acceleration is high enough, 40x40x4 for about 10 seconds!
        idxs = np.array([s.ind for s in self.sites])
        for i, s in enumerate(self.sites):
            neighbors_idxs = idxs[(self.atoms.get_distances(i, idxs, mic=pbc) < Ncutoff) & (idxs != i)]
            for j in neighbors_idxs:
                self.sites[i].neighbors.append(j)

        if len(self.neighbors) == 0:
            self.neighbors = [s.neighbors for s in self.sites]
            self.verify_nlist()

    def cover_system(self, species, coverage):
        """Covers the system with a certain species.
            
        Randomly covers the system with a species, at a
        certain fractional coverage.
    
        Parameters
        ----------
        species: int
            The species as defined by the user (e.g. empty=0,CO=1).
        coverage: float
            The fractional coverage to load lattice with.

        """
        n_covered = int(np.round(coverage * len(self.system.sites)))
        chosen_sites = np.random.choice(len(self.system.sites), n_covered)
        for c in chosen_sites:
            self.system.sites[c].covered = species
