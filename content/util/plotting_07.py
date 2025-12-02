import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_07(
    newton_local,
    newton_global,
    step_size,
    f,
    g,
    h,
    resolution_x,
    resolution_y,
    x_min,
    x_max,
    y_min,
    y_max,
    tol = 1e-16, 
    kmax = 50,
    delta = 0.01,
    p = 3,
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
            values_local [row, col] = len(
                newton_local (f, g, h, x_0, tol, kmax)
            ) - 1
            values_global[row, col] = len(
                newton_global(f, g, h, x_0, tol, kmax, step_size, delta, p)
            ) - 1

    xs = np.arange(x_min, x_max, step_x)
    ys = np.arange(y_min, y_max, step_y)
    X, Y = np.meshgrid(xs, ys)
    Z = [f(np.array([x, y])) for x, y in zip(X, Y)]

    min_iter = np.min([values_global, values_local])
    max_iter = np.max([values_global, values_local])

    plt.figure(figsize=(18, 4))
    plt.subplot(131)
    plt.title('f')
    plt.contour(X, Y, Z, levels = 40)

    plt.subplot(132)
    plt.title('Lokales Newton-Verfahren')
    plt.imshow(
        values_local.T, 
        extent=[x_min, x_max, y_min, y_max],
        vmin = min_iter,
        vmax = max_iter,
        origin = 'lower',
    )
    
    ax = plt.subplot(133)
    plt.title('Globales Newton-Verfahren')
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