from collections import Iterable
import torch

def make_grid(grid_size, cell_size=None, grid_offset=None):
    """Construct an N-dimensional grid"""

    # Handle default or non-tuple cell_sizes
    if cell_size is None:
        cell_size = [1.] * len(grid_size)
    elif not isinstance(cell_size, Iterable):
        cell_size = [cell_size] * len(grid_size)
    
    # By default the grid offset is set to zero
    if grid_offset is None:
        grid_offset = [0.] * len(grid_size)
    
    coords = [torch.arange(0, gs, cs) + off for gs, cs, off 
              in zip(grid_size, cell_size, grid_offset)]
    grid = torch.meshgrid(*coords[::-1])[::-1]
    return torch.stack(grid, -1)