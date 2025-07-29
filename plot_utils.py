import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mpc
from matplotlib.transforms import ScaledTranslation
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

def divergent_green_yellow_orange(N=100, return_palette=False, n_colors=8):
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
        
    clist = ["0c6c55","308675","53a094","97bdb7","ffeed0","f7ccbd","f7a384","f8794b","f84f12"]
    hex = [f'#{c}' for c in clist]
    rgb = list(map(mpc.to_rgb, hex))
    if return_palette:
        return sns.color_palette(rgb, n_colors=n_colors)
    else:
        return mpc.LinearSegmentedColormap.from_list('custom', rgb, N=N)    


def cozy_adventure(n_colors=16):
    """
    Generate a divergent green-orange colormap.

    Parameters
    ----------
    n_colors : int, optional
        Number of colors in the colormap. Default is 16.
        
    Returns
    -------
    color palette
        A seaborn color palette.
        
    Examples
    --------
    Generate a seaborn color palette with 5 colors:
    >>> palette = cozy_adventure(n_colors=5)
    """
    
    clist = ["ffd887", "eb9361", "da5e4e", "ab2330", "dfffff", "b5de89", "6aab7c", "26616b", "a2dceb", 
             "759ed0", "434ea8", "2a2140", "e1a7c5", "ab7ac6", "735bab", "3b3772"]
    hex = [f'#{c}' for c in clist]
    rgb = list(map(mpc.to_rgb, hex))
    
    return sns.color_palette(rgb, n_colors=n_colors)

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


def split_barplot(df, x=None, y=None, top=None, equal_scale=False, figsize=(6, 10), dpi=100, 
                  colors=['#308675', '#f8794b']):
    """
    Create two barplots from a DataFrame, one for positive and one for negative values.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data to plot.
    x : str
        The name of the column containing the x-axis data. Default is None.
    y : str
        The name of the column containing the y-axis data. Default is None.
    top : int
        The number of top values to plot. Default is None.
    equal_scale : bool (experimental)
        Whether to use the same number of yticks for both barplots. Default is False.
    figsize : tuple
        The size of the figure.
    dpi : int
        The resolution of the figure.
    colors : list
        The colors to use for the negative and positive loadings.
    """

    # If dataframe does not contain column 'err' define with 0
    if 'err' not in df.columns:
        df['err'] = np.zeros(len(df))

    # Split data into positive and negative loadings
    negative_df = df[df[x] < 0].reset_index()
    positive_df = df[df[x] > 0].reset_index()
    
    negative_df = negative_df.sort_values(x, ascending=True)
    positive_df = positive_df.sort_values(x, ascending=False)

    # If top is given, only plot the top n values
    if isinstance(top, int):        
        negative_df = negative_df.head(top)
        positive_df = positive_df.head(top)
        
        # if equal_scale is True raise error
        if equal_scale:
            raise ValueError("Cannot use 'top' and 'equal_scale' together.")

    # Find the maximum absolute value in the original dataframe for x-axis scaling
    max_abs_index = df[x].abs().idxmax()
    max_value = np.abs(df.at[max_abs_index, x])
    max_err = df.at[max_abs_index, 'err']
    # Add small value to have space between axis and bars
    max_err = max_err + max_value * 0.02 
    axes_lim = max_value + max_err

    # Error values for the barplot
    negative_err = negative_df['err'].values
    positive_err = positive_df['err'].values

    # Plotting
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize, dpi=dpi)

    # Negative barplot
    sns.barplot(ax=axes[0], x=x, y=y, data=negative_df, xerr=negative_err, color=colors[0], 
                error_kw=dict(ecolor='black', lw=1, capsize=1, capthick=1, alpha=0.7))
    axes[0].set_xlim(-axes_lim, 0)

    # Positive barplot
    sns.barplot(ax=axes[1], x=x, y=y, data=positive_df, xerr=positive_err, color=colors[1], 
                error_kw=dict(ecolor='black', lw=1, capsize=1, capthick=1, alpha=0.7))
    axes[1].set_xlim(0, axes_lim)

    # Set ytick labels for the positive subplot on the right side
    axes[1].yaxis.tick_right()
    axes[1].yaxis.set_label_position("right")

    if equal_scale:
        # TODO: Fix y-axis labels. Right now using ScaledTranslation to move labels, but a fixed offset would be better.
        #       Simply calling set_yticks on both axes does not work
        max_yticks = max(len(negative_df), len(positive_df))
        more_neg = len(negative_df) > len(positive_df)
        
        if more_neg:
            # turn autoscale for right axis off
            axes[1].autoscale(False)
            # set ytick majors are at same positions as left axis
            yticklabels = positive_df[y].tolist() + [''] * (max_yticks - len(positive_df))
            axes[1].set_yticks(axes[0].get_yticks() + 0.5,
                               labels=yticklabels)
            # turn yticks off
            axes[0].tick_params(axis='y', which='both', left=False, right=False)
            axes[1].tick_params(axis='y', which='both', left=False, right=False)
            
            # move labels back to their position
            for label in axes[1].get_yticklabels():
                label.set_transform(label.get_transform() + 
                                    ScaledTranslation(0, 0.1, axes[1].figure.dpi_scale_trans))
                        
            # OLD APPROACH
            # axes[1].sharey(axes[0])
            # g = axes[1].get_shared_y_axes();
            # g.remove(g.get_siblings(axes[1])[0])
            # axes[0].set_yticklabels(negative_df[y].tolist())
            # axes[1].set_yticklabels(positive_df[y].tolist() + [''] * (max_yticks - len(positive_df)))
        else:
            # turn autoscale for right axis off
            axes[0].autoscale(False)
            # set ytick majors are at same positions as left axis
            yticklabels = negative_df[y].tolist() + [''] * (max_yticks - len(negative_df))
            axes[0].set_yticks(axes[1].get_yticks() + 0.5,
                               labels=yticklabels)
            # turn yticks off
            axes[0].tick_params(axis='y', which='both', left=False, right=False)
            axes[1].tick_params(axis='y', which='both', left=False, right=False)
            
            # move labels back to their position
            for label in axes[0].get_yticklabels():
                label.set_transform(label.get_transform() + 
                                    ScaledTranslation(0, 0.1, axes[0].figure.dpi_scale_trans))          

    return fig, axes