import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backend_bases

def plot_step_sizes(descent_algorithm, optimal_step_size, armijo_step_size, wolfe_powell_step_size, A, b, x_0, tol, kmax):

    reference_solution = np.linalg.solve(A, -b)
    xg, yg, values = precompute_function_values(A, b)
    iterates_exact = descent_algorithm(A, b, x_0, tol, kmax, optimal_step_size)

    armijo_range_beta  = np.linspace(0.05, 0.95, 19)
    armijo_range_gamma = np.linspace(0.05, 0.95, 19)

    armijo_counts = precompute_counts(
        descent_algorithm, armijo_step_size, A, b, x_0, tol, kmax, armijo_range_beta, armijo_range_gamma,
    )
    
    wp_range_gamma = np.linspace(0.025, 0.475, 19)
    wp_range_rho   = np.linspace(0.025, 0.975, 39)

    wp_counts = precompute_counts(
        descent_algorithm, wolfe_powell_step_size, A, b, x_0, tol, kmax, wp_range_gamma, wp_range_rho, True,
    )

    idx_armijo_beta, idx_armijo_gamma = np.unravel_index(armijo_counts.argmin(), armijo_counts.shape)
    idx_wp_gamma, idx_wp_rho = np.unravel_index(wp_counts.argmin(), wp_counts.shape)
    armijo_opt_beta = armijo_range_beta[idx_armijo_beta]
    armijo_opt_gamma = armijo_range_gamma[idx_armijo_gamma]
    wp_opt_gamma = wp_range_gamma[idx_wp_gamma]
    wp_opt_rho = wp_range_rho[idx_wp_rho]
    
    print(f'Optimale Parameter (Armijo): beta={round(armijo_opt_beta, 2)}, gamma={round(armijo_opt_gamma, 2)}, Anzahl Iterationen: {int(armijo_counts.min())}.')
    print(f'Optimale Parameter (Wolfe-Powell): gamma={round(wp_opt_gamma, 2)}, rho={round(wp_opt_rho, 2)}, Anzahl Iterationen: {int(wp_counts.min())}.')
    
    figure = plt.figure(figsize = (10.0, 10.0))
    axes_armijo, axes_wp = plot_main(figure, armijo_counts, wp_counts)
    
    def draw_function_armijo(parameters): 
        def step_size(A, b, x, d):
            return armijo_step_size(A, b, x, d, parameters[0], parameters[1])
        title = f'Armijo, $\\gamma={round(parameters[0], 4)}, \\beta={round(parameters[1], 4)}$'
        draw_descent(
            axes_armijo, reference_solution, iterates_exact, xg, yg, values,
            lambda A, b, x_0, tol, kmax : descent_algorithm(A, b, x_0, tol, kmax, step_size),
            A, b, x_0, tol, kmax, title, 'Armijo',
        )
    def draw_function_wp(parameters): 
        def step_size(A, b, x, d):
            return wolfe_powell_step_size(A, b, x, d, parameters[0], parameters[1])
        title = f'Wolfe-Powell, $\\gamma={round(parameters[0], 4)}, \\rho={round(parameters[1], 4)}$'
        draw_descent(
            axes_wp, reference_solution, iterates_exact, xg, yg, values,
            lambda A, b, x_0, tol, kmax : descent_algorithm(A, b, x_0, tol, kmax, step_size),
            A, b, x_0, tol, kmax, title, 'Wolfe-Powell',
        )
    figure.canvas.mpl_connect(
        'button_press_event', lambda event : button_press(
            event, 
            draw_function_armijo,
            draw_function_wp,
        ),
    )
    plt.show()


def draw_descent(axes, reference_solution, iterates_exact, xg, yg, values, algorithm, A, b, x_0, tol, kmax, title, label):
    # computation
    iterates = algorithm(A, b, x_0, tol, kmax)
    
    # plotting
    plt.sca(axes)
    axes.clear()
    plt.title(title)
    function_values = [0.5 * np.inner(x, np.dot(A, x)) + np.inner(b, x) for x in iterates_exact]
    function_values = sorted(set(function_values))

    plt.contour(xg, yg, values, function_values)
    
    plt.plot(iterates_exact[:,0], iterates_exact[:,1], 'r-o', label='exakt')
    plt.plot(iterates[:,0], iterates[:,1], 'g-o', label=label)

    plt.draw()


def plot_main(figure, armijo_counts, wp_counts):
    plt.suptitle('Gradientenverfahren', fontsize = 20)
    
    plt.subplot2grid((2, 2), (0, 0))#, rowspan=10, colspan=16)
    plt.title('Armijo')
    plt.imshow(armijo_counts, cmap='jet', aspect='equal', interpolation='none', extent=(0.0,1.0,0.0,1.0), origin='lower')
    plt.colorbar()
    plt.axis('equal')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel('$\\gamma$')
    plt.ylabel('$\\beta$')

    plt.subplot2grid((2, 2), (0, 1))#, rowspan=10, colspan=8)
    plt.title('Wolfe-Powell')
    plt.imshow(wp_counts.T, cmap='jet', aspect='equal', interpolation='none', extent=(0.0, 0.5, 0.0, 1.0), origin='lower')
    plt.colorbar()
    plt.axis('equal')
    plt.xlim(0.0, 0.5)
    plt.ylim(0.0, 1.0)
    plt.xlabel('$\\gamma$')
    plt.ylabel('$\\rho$')

    axes_armijo = plt.subplot2grid((2, 2), (1, 0))#, rowspan = 8, colspan = 8)
    axes_wp = plt.subplot2grid((2, 2), (1, 1))#, rowspan = 8, colspan = 8)

    return axes_armijo, axes_wp
    
def precompute_counts(
    descent_algorithm, step_size, A, b, x_0, tol, kmax, range_parameter_1, range_parameter_2, ensure_2_greater_1 = False,
):
    counts = np.inf * np.ones((len(range_parameter_1), len(range_parameter_2)), dtype=int)
    for i, parameter_1 in enumerate(range_parameter_1):
        for j, parameter_2 in enumerate(range_parameter_2):
            if ensure_2_greater_1 and parameter_1 >= parameter_2:
                continue
            counts[i, j] = len(descent_algorithm(
                A, b, x_0, tol, kmax, lambda A, b, x, d : step_size(A, b, x, d, parameter_1, parameter_2),
            )) - 1
    return counts

def precompute_function_values(A, b):
    xa = -2.5
    xb =  2.5
    ya = -1.5
    yb =  1.5
    sf = 100.0
    nx = int((xb - xa) * sf + 0.5) + 1
    ny = int((yb - ya) * sf + 0.5) + 1
    xg = np.linspace(xa, xb, nx)
    yg = np.linspace(ya, yb, ny)
    x1 = np.ones(nx)
    y1 = np.ones(ny)
    xm = np.outer(y1, xg).reshape(nx*ny)
    ym = np.outer(yg, x1).reshape(nx*ny)
    xyg = np.vstack([xm, ym])
    values = [0.5 * np.inner(xyg[:,i], np.dot(A, xyg[:,i])) + np.inner(b, xyg[:,i])  for i in range(len(xyg[1,:]))]
    return xg, yg, np.reshape(values, (ny, nx))

def button_press(button_event, draw_function_armijo, draw_function_wp):
    if (button_event.button != 1):
        return

    if not(button_event.xdata and button_event.ydata):
        return
    
    if button_event.inaxes.title.get_text() == 'Armijo':
        draw_function_armijo(np.array([button_event.xdata, button_event.ydata]))

    if button_event.inaxes.title.get_text() == 'Wolfe-Powell':
        draw_function_wp(np.array([button_event.xdata, button_event.ydata]))