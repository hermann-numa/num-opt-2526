import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backend_bases

def plot_steepest_descent(algorithm, A, b, x_0, tol, kmax):

    reference_solution = np.linalg.solve(A, -b)
    figure = plt.figure(figsize = (15.0, 9.0))

    xg, yg, values = precompute_function_values(A, b)    
    
    draw_descent(figure, reference_solution, xg, yg, values, algorithm, A, b, x_0, tol, kmax)
    
    figure.canvas.mpl_connect(
        'button_press_event', lambda event : button_press(
            event, 
            lambda x_0_new : draw_descent(
                figure, reference_solution, xg, yg, values, algorithm, A, b, x_0_new, tol, kmax,
            ),
        ),
    )

    plt.show()

def draw_descent(figure, reference_solution, xg, yg, values, algorithm, A, b, x_0, tol, kmax):
    # computation
    iterates = algorithm(A, b, x_0, tol, kmax)
    title = f'{len(iterates)-1} Iterationsschritte, Abstand zur Optimalstelle: {np.linalg.norm(iterates[-1,:] - reference_solution)}'
    
    # plotting
    plt.figure(figure.number)
    plt.clf()
    plt.title(title)
    function_values = [0.5 * np.inner(x, np.dot(A, x)) + np.inner(b, x) for x in iterates]
    function_values = sorted(set(function_values))

    plt.contour(xg, yg, values, function_values)
    
    plt.plot(iterates[:,0], iterates[:,1], 'k-o')

    plt.draw()


def button_press(button_event, draw_function):
    if (button_event.button != 1):
        return

    if not(button_event.xdata and button_event.ydata):
        return
        
    draw_function(np.array([button_event.xdata, button_event.ydata]))

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