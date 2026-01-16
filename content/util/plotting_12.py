import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backend_bases

def plot_iterates(algorithm, f, g, h, df, dg, dh, x_0, global_optimizer, xlims = (-3, 3), ylims = (-3, 3)):

    figure = plt.figure(figsize = (9.5, 9.0))

    xs, ys, values_f = precompute_function_values(f, xlims, ylims)
    values_g, values_h = None, None
    if g is not None:
        _,   _, values_g = precompute_function_values(g, xlims, ylims)
    if h is not None:
        _,   _, values_h = precompute_function_values(h, xlims, ylims)

    draw_background(figure, xs, ys, values_f, values_g, values_h)

    iterates = algorithm(f, g, h, df, dg, dh, x_0, global_optimizer)
    
    iterates_plt = draw_iterates(figure, iterates)

    figure.canvas.mpl_connect(
        'button_press_event', 
        lambda event : button_press(
            event,
            lambda x_0_new : update_iterates(
                figure, algorithm, f, g, h, df, dg, dh, x_0_new, global_optimizer,
                *iterates_plt,
            ),
        ),
    )

    plt.show()

def draw_background(figure, xs, ys, values_f, values_g, values_h):
    plt.figure(figure.number)
    plt.clf()

    if values_g is not None:
        for i in range(values_g.shape[0]):
            plt.contourf(xs, ys, values_g[i, :, :], [0, np.max(values_g)], colors = ['#dddddd',], extend = 'max')

    plt.contour(xs, ys, values_f[0, :, :], 40)

    if values_g is not None:
        for i in range(values_g.shape[0]):
            plt.contour (xs, ys, values_g[i, :, :], [0,], colors = ['k'], linewidths = [3,])

    if values_h is not None:
        for i in range(values_h.shape[0]):
            plt.contour (xs, ys, values_h[i, :, :], [0,], colors = ['k'], linewidths = [3,])

    plt.axis('equal')
    plt.draw()

def draw_iterates(figure, iterates):
    plt.figure(figure.number)
    iterates_plt_1, = plt.plot(iterates[ :,0], iterates[ :,1], 'm-')
    iterates_plt_2, = plt.plot(iterates[ :,0], iterates[ :,1], 'm.')
    iterates_plt_3, = plt.plot(iterates[ 0,0], iterates[ 0,1], 'ro')
    iterates_plt_4, = plt.plot(iterates[-1,0], iterates[-1,1], 'go')
    plt.draw()
    return iterates_plt_1, iterates_plt_2, iterates_plt_3, iterates_plt_4

def update_iterates_plt(figure, iterates,
    iterates_plt_1, iterates_plt_2, iterates_plt_3, iterates_plt_4,
):
    plt.figure(figure.number)
    iterates_plt_1.set_data(iterates[ :,0], iterates[ :,1])
    iterates_plt_2.set_data(iterates[ :,0], iterates[ :,1])
    iterates_plt_3.set_data(iterates[ 0,0], iterates[ 0,1])
    iterates_plt_4.set_data(iterates[-1,0], iterates[-1,1])
    plt.draw()

def update_iterates(figure, algorithm, f, g, h, df, dg, dh, x_0, global_optimizer,
    iterates_plt_1, iterates_plt_2, iterates_plt_3, iterates_plt_4,
):
    iterates = algorithm(f, g, h, df, dg, dh, x_0, global_optimizer)
    update_iterates_plt(figure, iterates, iterates_plt_1, iterates_plt_2, iterates_plt_3, iterates_plt_4)

def button_press(button_event, draw_function):
    if (button_event.button != 1):
        return
    if not(button_event.xdata and button_event.ydata):
        return
    draw_function(np.array([button_event.xdata, button_event.ydata]))

def precompute_function_values(f, xlims, ylims, scale_factor = 100.0):
    nx = int((xlims[1] - xlims[0]) * scale_factor + 0.5) + 1
    ny = int((ylims[1] - ylims[0]) * scale_factor + 0.5) + 1
    xs = np.linspace(*xlims, nx)
    ys = np.linspace(*ylims, ny)
    n = f(np.array([xs[0], ys[0]])).shape[0]
    values = np.zeros((n, nx, ny))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            values[:, j, i] = f(np.array([x, y]))
    return xs, ys, values