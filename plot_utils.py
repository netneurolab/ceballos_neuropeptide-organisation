import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mpc
from matplotlib.cm import ColormapRegistry
from nilearn.plotting import plot_surf
from neuromaps.datasets import fetch_atlas
from neuromaps.parcellate import Parcellater
from neuromaps.images import dlabel_to_gifti, annot_to_gifti

def custom_surf_plot(data, space='fsLR', density='32k', template='inflated', cmap='coolwarm', dpi=100,
                     parcellation=None, cbar_label=None, cbar_ticks=None, hemi=None,
                     vmin=None, vmax=None):
    """
    Custom surface plot ROI-wise or vertex-wise data in fsLR or fsaverage space.
    
    Parameters
    ----------
    data : array_like or tuple
        ROI-wise or vertex-wise data. If tuple, assumes (left, right) 
        hemisphere.
    density : str
        Density of surface plot, can be '8k', '32k' or '164k'.
    template : str
        Type of surface plot. Can be 'inflated', 'veryinflated', 'sphere' or 
        'midthickness' (fsLR, civet) / 'pial' (fsaverage).
    cmap : str
        Colormap.
    dpi : int
        Resolution of plot.
    parcellation : Path or tuple, optional
        Path to an parcellation in .dlabel.nii or .annot format. If tuple, 
        assumes (left, right) GIFTI objects.
    cbar_label: str, optional
        Colorbar label.
    cbar_ticks: list, optional
        Colorbar ticks.
    hemi : str, optional
        Hemisphere to plot. Can be 'left' or 'right'.
    vmin/vmax : int, optional
        Minimun/ maximum value in the plot.
    """
    
    space = space.lower()
    
    if parcellation is not None:
        if not isinstance(parcellation, tuple):
            parcellation = dlabel_to_gifti(parcellation) if space=='fslr' else \
                           annot_to_gifti(parcellation)
        surf_masker = Parcellater(parcellation, space, 
                                  resampling_target='parcellation')
        data = surf_masker.inverse_transform(data)
        l_data, r_data = data[0].agg_data(), data[1].agg_data()
    else:
        if not isinstance(data, tuple):
            raise ValueError("Data input must be tuple of vertex-wise values. \
                             Alternatively, provide 'parcellation' option to use ROI data.")
        l_data, r_data = data[0], data[1]
        
    if None in (vmin, vmax):
        # Handle NaNs in left hemisphere data
        l_min, l_max = np.nanmin(l_data), np.nanmax(l_data)
        l_data = np.nan_to_num(l_data, nan=l_min)
        
        # Handle NaNs in right hemisphere data
        r_min, r_max = np.nanmin(r_data), np.nanmax(r_data)
        r_data = np.nan_to_num(r_data, nan=r_min)
        
        # min/max values in the data
        vmin = np.min([l_min, r_min])
        vmax = np.max([l_max, r_max])
        
    if cbar_ticks is None:
        cbar_ticks = ['min', 'max']
    
    # Fetch surface template for plot
    surfaces = fetch_atlas(space, density)
    lh, rh = surfaces[template]
    
    if hemi == None:
        # Plot both hemispheres
        fig, ax = plt.subplots(nrows=1,ncols=4,subplot_kw={'projection': '3d'}, 
                            figsize=(12, 4), dpi=dpi)
        
        plot_surf(lh, l_data, threshold=-1e-14, cmap=cmap, alpha=1, view='lateral', 
                colorbar=False, axes=ax.flat[0], vmin=vmin, vmax=vmax)
        plot_surf(lh, l_data, threshold=-1e-14, cmap=cmap, alpha=1, view='medial', 
                colorbar=False, axes=ax.flat[1], vmin=vmin, vmax=vmax)

        plot_surf(rh, r_data, threshold=-1e-14, cmap=cmap, alpha=1, view='lateral', 
                colorbar=False, axes=ax.flat[2], vmin=vmin, vmax=vmax)
        p = plot_surf(rh, r_data, threshold=-1e-14, cmap=cmap, alpha=1, view='medial', 
                    colorbar=True, axes=ax.flat[3], vmin=vmin, vmax=vmax)

        p.axes[-1].set_ylabel(cbar_label, fontsize=10, labelpad=0.5)
        p.axes[-1].set_yticks([vmin, vmax])
        p.axes[-1].set_yticklabels(cbar_ticks)
        p.axes[-1].tick_params(labelsize=7, width=0, pad=0.1)
        plt.subplots_adjust(wspace=-0.05)
        p.axes[-1].set_position(p.axes[-1].get_position().translated(0.08, 0))
        
    elif hemi == 'left' or hemi == 'right':
        # Plot one hemisphere
        fig, ax = plt.subplots(nrows=1,ncols=2,subplot_kw={'projection': '3d'}, 
                            figsize=(8, 4), dpi=dpi)
        
        h = lh if hemi == 'left' else rh
        h_data = l_data if hemi == 'left' else r_data
        
        plot_surf(h, h_data, threshold=-1e-14, cmap=cmap, alpha=1, view='lateral', 
                colorbar=False, axes=ax.flat[0], vmin=vmin, vmax=vmax)
        p = plot_surf(h, h_data, threshold=-1e-14, cmap=cmap, alpha=1, view='medial', 
                    colorbar=True, axes=ax.flat[1], vmin=vmin, vmax=vmax)
        
        p.axes[-1].set_ylabel(cbar_label, fontsize=10, labelpad=0.5)
        p.axes[-1].set_yticks([vmin, vmax])
        p.axes[-1].set_yticklabels(cbar_ticks)
        p.axes[-1].tick_params(labelsize=7, width=0, pad=0.1)
        plt.subplots_adjust(wspace=-0.05)
        p.axes[-1].set_position(p.axes[-1].get_position().translated(0.08, 0))

def sequential_blue(N=100, return_palette=False, n_colors=8):
    """
    Generate a sequential blue colormap.

    Parameters
    ----------
    N : int, optional
        Number of colors in the colormap. Default is 100.
    return_palette : bool, optional
        If True, return a seaborn color palette instead of a colormap. Default is False.
    n_colors : int, optional
        Number of colors in the palette. Only applicable if return_palette is True. Default is 8.

    Returns
    -------
    colormap or color palette
        A matplotlib colormap or seaborn color palette.

    Examples
    --------
    Generate a sequential blue colormap with 50 colors:
    >>> cmap = sequential_blue(N=50)

    Generate a seaborn color palette with 5 colors:
    >>> palette = sequential_blue(return_palette=True, n_colors=5)
    """
    
    # taken from https://coolors.co/f4f5f5-e8eaed-bed5e1-93bfd5-2b7ea1-2b6178
    clist = ["d2d6da","e8eaed","bed5e1","93bfd5","2b7ea1","2b6178"]
    hex = [f'#{c}' for c in clist]
    rgb = list(map(mpc.to_rgb, hex))
    if return_palette:
        return sns.color_palette(rgb, n_colors=n_colors)
    else:
        return mpc.LinearSegmentedColormap.from_list('custom', rgb, N=N)

def sequential_green(N=100, return_palette=False, n_colors=8):
    """
    Generate a sequential green colormap.

    Parameters
    ----------
    N : int, optional
        Number of colors in the colormap. Default is 100.
    return_palette : bool, optional
        If True, return a seaborn color palette instead of a colormap. Default is False.
    n_colors : int, optional
        Number of colors in the palette. Only applicable if return_palette is True. Default is 8.

    Returns
    -------
    colormap or color palette
        A matplotlib colormap or seaborn color palette.

    Examples
    --------
    Generate a sequential blue colormap with 50 colors:
    >>> cmap = sequential_green(N=50)

    Generate a seaborn color palette with 5 colors:
    >>> palette = sequential_green(return_palette=True, n_colors=5)
    """
    
    clist = ["e7f0ee","c4dcd2","a4c5b8","79aa94","4c8a70","206246","114d33","013721"]
    hex = [f'#{c}' for c in clist]
    rgb = list(map(mpc.to_rgb, hex))
    if return_palette:
        return sns.color_palette(rgb, n_colors=n_colors)
    else:
        return mpc.LinearSegmentedColormap.from_list('custom', rgb, N=N)

def custom_coolwarm(N=100, return_palette=False, n_colors=8):
    """
    Generate an adapted version of seaborn's coolwarm colormap.

    Parameters
    ----------
    N : int, optional
        Number of colors in the colormap. Default is 100.
    return_palette : bool, optional
        If True, return a seaborn color palette instead of a colormap. Default is False.
    n_colors : int, optional
        Number of colors in the palette. Only applicable if return_palette is True. Default is 8.

    Returns
    -------
    colormap or color palette
        A matplotlib colormap or seaborn color palette.

    Examples
    --------
    Generate a sequential blue colormap with 50 colors:
    >>> cmap = custom_coolwarm(N=50)

    Generate a seaborn color palette with 5 colors:
    >>> palette = custom_coolwarm(return_palette=True, n_colors=5)
    """
        
    clist = ["2a6179","3d758d","75aec7","c2d4dc","ebebeb","e5d3d1","ea9085","c86356","b73a2a"]
    hex = [f'#{c}' for c in clist]
    rgb = list(map(mpc.to_rgb, hex))
    if return_palette:
        return sns.color_palette(rgb, n_colors=n_colors)
    else:
        return mpc.LinearSegmentedColormap.from_list('custom', rgb, N=N)

def categorical_cmap(N=None, return_palette=False, n_colors=8):
    """
    Create a categorical colormap.

    Parameters:
        N (int, optional): Number of colors in the colormap. If None, the number of colors is determined by the length of the input color list. Default is None.
        return_palette (bool, optional): If True, return a color palette instead of a colormap. Default is False.
        n_colors (int, optional): Number of colors in the returned palette. Only applicable if return_palette is True. Default is 8.

    Returns:
        matplotlib.colors.Colormap or list: If return_palette is False, returns a matplotlib colormap. If return_palette is True, returns a list of colors.
    """
    
    clist = ["ea6b5d","65a488","498eab"]
    hex = [f'#{c}' for c in clist]
    rgb = list(map(mpc.to_rgb, hex))
    N = len(rgb) if N==None else N
    if return_palette:
        return sns.color_palette(rgb, n_colors=n_colors)
    else:
        return mpc.LinearSegmentedColormap.from_list('custom', rgb, N=N)


def divergent_green_orange(N=100, return_palette=False, n_colors=8):
    """
    Generate a divergent green-orange colormap.

    Parameters
    ----------
    N : int, optional
        Number of colors in the colormap. Default is 100.
    return_palette : bool, optional
        If True, return a seaborn color palette instead of a colormap. Default is False.
    n_colors : int, optional
        Number of colors in the palette. Only applicable if return_palette is True. Default is 8.

    Returns
    -------
    colormap or color palette
        A matplotlib colormap or seaborn color palette.

    Examples
    --------
    Generate a sequential blue colormap with 50 colors:
    >>> cmap = divergent_green_orange(N=50)

    Generate a seaborn color palette with 5 colors:
    >>> palette = divergent_green_orange(return_palette=True, n_colors=5)
    """
        
    clist = ["0c6c55","308675","53a094","97bdb7","f6f6f6","f7ccbd","f7a384","f8794b","f84f12"]
    hex = [f'#{c}' for c in clist]
    rgb = list(map(mpc.to_rgb, hex))
    if return_palette:
        return sns.color_palette(rgb, n_colors=n_colors)
    else:
        return mpc.LinearSegmentedColormap.from_list('custom', rgb, N=N)    

def cmap_from_hex(clist, N=100, return_palette=False, n_colors=8):
    """
    Create a custom colormap from a list of hexadecimal color codes.

    Parameters
    ----------
    clist : list
        A list of hexadecimal color codes.
    N : int, optional
        The number of colors in the colormap. Default is 100.
    return_palette : bool, optional
        Whether to return the color palette as well. Default is False.
    n_colors : int, optional
        The number of distinct colors to extract from the colormap. Default is 8.

    Returns
    -------
    cmap : matplotlib.colors.ListedColormap
        The custom colormap.
    palette : list, optional
        The color palette extracted from the colormap, if `return_palette` is True.
    """

    hex_list = [f'#{c}' for c in clist]
    rgb = list(map(mpc.to_rgb, hex_list))
    if return_palette:
        return sns.color_palette(rgb, n_colors=n_colors)
    else:
        return mpc.LinearSegmentedColormap.from_list('custom', rgb, N=N)

def schaefer_cmap(include_nota=False):
    """
    Create a custom colormap based on the Schaefer atlas.

    Parameters
    ----------
    include_nota : bool, optional
        Whether to include the "None of the above" category in the colormap. 
        Default is False.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        The custom colormap.

    """
    
    if include_nota:
        rgb = np.array([(119, 17, 128),  # Vis
                        (70, 128, 179),  # SomMot
                        (4, 117, 14),  # DorsAttn
                        (200, 56, 246),  # SalVentAttn
                        (223, 249, 163),  # Limbic
                        (232, 147, 31),  # Cont
                        (218, 24, 24),  # Default
                        (255, 255, 255)  # None of the above
                        ]) / 255
    else:
        rgb = np.array([(119, 17, 128),  # Vis
                        (70, 128, 179),  # SomMot
                        (4, 117, 14),  # DorsAttn
                        (200, 56, 246),  # SalVentAttn
                        (223, 249, 163),  # Limbic
                        (232, 147, 31),  # Cont
                        (218, 24, 24)  # Default
                        ]) / 255
    return mpc.LinearSegmentedColormap.from_list('custom', rgb, N=len(rgb))