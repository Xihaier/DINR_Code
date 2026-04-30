import os
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import mrcfile
import vtk

from vtk.util import numpy_support as vtk_numpy


def numpy_to_mrc(
    data: np.ndarray,
    out_path: str,
    voxel_size: tuple = (1.0, 1.0, 1.0),
    origin: tuple = (0.0, 0.0, 0.0),
    order: str = 'zyx'
):
    """
    Save a numpy array as MRC/CCP4 .map file for ChimeraX visualization.
    
    Args:
        data: Input numpy array to save.
        out_path: Output file path for the .mrc file.
        voxel_size: Voxel size in Angstroms (x, y, z).
        origin: Map origin in Angstroms (x0, y0, z0).
        order: Axis order in input data ('zyx', 'xyz', 'xzy', 'yzx').
    
    Raises:
        ValueError: If unknown axis order is specified.
    """
    # Convert to float32 and handle invalid values
    arr = np.asarray(data, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Transpose to MRC order (z, y, x)
    axis_order_map = {
        'xyz': (2, 1, 0),
        'xzy': (1, 2, 0),
        'yzx': (2, 0, 1),
        'zyx': None,
    }
    
    order_lower = order.lower()
    if order_lower not in axis_order_map:
        raise ValueError(f"Unknown axis order: {order}. Must be one of {list(axis_order_map.keys())}")
    
    if axis_order_map[order_lower] is not None:
        arr = np.transpose(arr, axis_order_map[order_lower])
    
    arr = np.ascontiguousarray(arr)
    
    # Write MRC file
    with mrcfile.new(out_path, overwrite=True) as mrc:
        mrc.set_data(arr)
        mrc.voxel_size = voxel_size
        mrc.header.origin = origin
        mrc.header.label[0] = b'Written by numpy_to_mrc() for ChimeraX'


def numpy_to_vti(
    data: np.ndarray,
    out_path: str,
    spacing: tuple = (1.0, 1.0, 1.0),
    origin: tuple = (0.0, 0.0, 0.0),
    scalar_name: str = "scalars"
):
    """
    Save a numpy array as VTK ImageData (.vti) file for ParaView/VTK visualization.
    
    Args:
        data: Input 3D numpy array with shape (nz, ny, nx).
        out_path: Output file path for the .vti file.
        spacing: Grid spacing (dx, dy, dz).
        origin: Grid origin (x0, y0, z0).
        scalar_name: Name for the scalar field in VTK.
    
    Raises:
        ImportError: If VTK package is not installed.
        ValueError: If input array is not 3D.
    """
    if data.ndim != 3:
        raise ValueError(f"Expected 3D array, got {data.ndim}D array with shape {data.shape}")
    
    nz, ny, nx = data.shape
    
    # Convert to float32 and ensure C-contiguous
    arr = np.ascontiguousarray(data, dtype=np.float32)
    
    # Create VTK image data structure
    image = vtk.vtkImageData()
    image.SetDimensions(nx, ny, nz)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    
    # Convert numpy array to VTK array
    vtk_array = vtk_numpy.numpy_to_vtk(
        num_array=arr.ravel(order="C"),
        deep=True,
        array_type=vtk.VTK_FLOAT
    )
    vtk_array.SetName(scalar_name)
    image.GetPointData().SetScalars(vtk_array)
    
    # Write to VTI file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(out_path)
    writer.SetInputData(image)
    writer.SetDataModeToBinary()
    writer.Write()


def visualize_sampled_data(sampled_indices, grid_size, epoch, method_name, save_dir):
    """
    Visualize sampled data points on the original grid and save the plot.

    Args:
        sampled_indices (torch.Tensor): Indices of sampled points.
        grid_size (int): Size of the 2D grid (e.g., 128).
        epoch (int): Current epoch number.
        method_name (str): Name of the sampling method.
        save_dir (str): Directory to save the plots.

    Returns:
        None: Displays and saves the plot.
    """
    # Convert 1D indices to 2D grid coordinates
    sampled_indices = sampled_indices.cpu().numpy()
    rows, cols = np.divmod(sampled_indices, grid_size)

    # Create a blank grid and mark sampled points
    sampled_grid = np.zeros((grid_size, grid_size), dtype=np.int32)
    sampled_grid[rows, cols] = 1  # Set sampled points to 1

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Visualize the sampled points
    with plt.style.context('science'):
        plt.figure(figsize=(3.5, 3.5))
        plt.imshow(sampled_grid, cmap="GnBu", origin="lower")
        plt.title(f"{method_name} (Epoch {epoch})")

        # Remove x-ticks, y-ticks, x-label, and y-label
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("")
        plt.ylabel("")

        # Save the plot
        file_name = f"{method_name}_epoch{epoch:03d}.png".replace(" ", "")
        file_path = os.path.join(save_dir, file_name)
        plt.savefig(file_path, dpi=300)
        plt.close()  # Close the plot to free memory