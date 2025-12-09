import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_08(
    method,
    step_size_local,
    step_size_global,
    f,
    g,
    resolution_x,
    resolution_y,
    x_min,
    x_max,
    y_min,
    y_max,
    tol = 1e-16, 
    kmax = 100,
    method_name = 'iBFGS',
):
    
    values_local  = np.zeros((resolution_x, resolution_y), dtype = int)
    values_global = np.zeros((resolution_x, resolution_y), dtype = int)

    step_x = (x_max - x_min) / (resolution_x - 1)
    step_y = (y_max - y_min) / (resolution_y - 1)

    for row in range(resolution_x):
        for col in range(resolution_y):
            x = x_min + (row * step_x)
            y = y_min + (col * step_y)
            x_0 = np.array([x, y])
            values_local [row, col] = len(method(f, g, x_0, step_size_local,  None, tol, kmax)) - 1
            values_global[row, col] = len(method(f, g, x_0, step_size_global, None, tol, kmax)) - 1

    xs = np.arange(x_min, x_max, step_x)
    ys = np.arange(y_min, y_max, step_y)
    X, Y = np.meshgrid(xs, ys)
    Z = [f(np.array([x, y])) for x, y in zip(X, Y)]

    min_iter = np.min([values_local, values_global])
    max_iter = np.max([values_local, values_global])

    plt.figure(figsize=(18, 4))
    plt.subplot(131)
    plt.title('f')
    plt.contour(X, Y, Z, levels = 40)

    plt.subplot(132)
    plt.title(f'Lokales {method_name}-Verfahren')
    plt.imshow(
        values_local.T, 
        extent=[x_min, x_max, y_min, y_max],
        vmin = min_iter,
        vmax = max_iter,
        origin = 'lower',
    )
    
    ax = plt.subplot(133)
    plt.title(f'Globales {method_name}-Verfahren')
    im = ax.imshow(
        values_global.T,
        extent=[x_min, x_max, y_min, y_max],
        vmin = min_iter,
        vmax = max_iter,
        origin = 'lower',
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax = cax)

    plt.autoscale(False)

    #plt.tight_layout()