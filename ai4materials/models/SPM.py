import numpy as np
from ase import Atoms

class SPM():
    
    def __init__(grid=None, box_size=None):
        
        if grid == None:
            grid = structure.positions
            
        self.grid = grid
        self.box_size = box_size
    
    def extract_laes(structure, grid, box_size):
        
        if not structure.__class__ = ase.atoms.Atoms:
            raise NotImplementedError("Current implementation requires ASE atoms object")
        
        
        local_aes = multiple_queries(structure, grid, box_size)
        
        return lcal_laes
                
    @njit            
    def get_mask(array, x_min, y_min, z_min,
                          x_max, y_max, z_max):
        
        mask = ((array[:, 0] > xmin) & (array[:,  1] > ymin) 
                & (array[:, 2] > zmin) & (array[:, 0] < xmax) 
                & (array[:, 1] < ymax) & (array[:, 2] < zmax))
        
        return mask
    
    def multiple_queries(structure, grid, box_size):
        """
        Wrapper for get_mask, sort data, onl pass subset to get_mask

        Returns
        -------
        None.
        """
        
        array = structure.positions
        species = structure.get_chemical_symbols()
        
        # Sort the array on the first dimension
        sorted_array = array[np.argsort(array[:, 0])]
        sorted_species = species[np.argsort(array[:, 0])]
        count = 0
        
        filtered_laes = []
    
        for point in grid:
            xmin, xmax = point[0] - delta, point[0] + delta
            ymin, ymax = point[1] - delta, point[1] + delta
            zmin, zmax = point[2] - delta, point[2] + delta
    
            min_index = np.searchsorted(sorted_array[:, 0], xmin, side='left')
            max_index = np.searchsorted(sorted_array[:, 0], xmax, side='right')
    
            mask = boolean_index_numba_multiple(sorted_array[min_index:max_index], 
                                                xmin, xmax, ymin, ymax, zmin, zmax)
            
            filtered_positions = sorted_array[min_index:max_index][mask]
            filtered_species = sorted_species[min_index, max_index][mask]
            
            filtered_laes.append(Atoms(positions=filtered_positions,
                                       species=filtered_species))
            
        return filtered_laes