import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_10(
    method,
    f,
    g,
    h,
    resolution_x,
    resolution_y,
    x_min,
    x_max,
    y_min,
    y_max,
    tol = 1e-8, 
    kmax = 50,
    delta_1 = 0.25,
    delta_2 = 0.75,
    sigma_1 = 0.5,
    sigma_2 = 2.0,
    Delta_0 = 0.1,
    Delta_min = 1e-3,
    min_shift = 1e-2,
    max_inner = 20,
):
    
    values = np.zeros((resolution_x, resolution_y), dtype = int)

    step_x = (x_max - x_min) / (resolution_x - 1)
    step_y = (y_max - y_min) / (resolution_y - 1)

    for row in range(resolution_x):
        for col in range(resolution_y):
            x = x_min + (row * step_x)
            y = y_min + (col * step_y)
            x_0 = np.array([x, y])
            values[row, col] = len(method(
                f, g, h, x_0, 
                tol, kmax, delta_1, delta_2, sigma_1, sigma_2,
                Delta_0, Delta_min, min_shift, max_inner,
            )) - 1

    xs = np.arange(x_min, x_max, step_x)
    ys = np.arange(y_min, y_max, step_y)
    X, Y = np.meshgrid(xs, ys)
    Z = [f(np.array([x, y])) for x, y in zip(X, Y)]

    min_iter = np.min(values)
    max_iter = np.max(values)

    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.title('f')
    plt.contour(X, Y, Z, levels = 40)

    ax = plt.subplot(122)
    plt.title(f'Trust-Region-Verfahren')
    im = ax.imshow(
        values.T,
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