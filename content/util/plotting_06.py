import time
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

def compute_and_plot_06(cg, pcg, m, mul_A, mul_pre, tol, maxit):
    #  Aufgabe initialisieren:
    n = m * m
    
    b = np.zeros(n)
    middle = int(n / 2) + int(m / 2)
    b = place_point(b, middle)
    b = place_point(b, middle - 1)
    b = place_point(b, middle + 1)
    upper_half = int(n / 4)
    b = place_point(b, upper_half)
    b = place_point(b, upper_half)
    b = place_point(b, upper_half)

    A = 4.0 * np.eye(n) - (np.diag(np.ones(n-1), k=-1) + np.diag(np.ones(n-1), k=1))
    A -= (np.diag(np.ones(n - m), k = m) + np.diag(np.ones(n - m), k = -m))
    ref_solution = np.linalg.solve(A, -1*b)

    cg_iterates,  cg_gradient_norms = cg(b, mul_A, tol, maxit)
    pcg_iterates, pcg_gradient_norms = pcg(b, mul_A, mul_pre, tol, maxit)

    err_cg  = norm(cg_iterates  - ref_solution, axis=1)
    err_pcg = norm(pcg_iterates - ref_solution, axis=1)

    steps_cg  = norm(cg_iterates[1:,:]  - cg_iterates[:-1,:],  axis=1) 
    steps_pcg = norm(pcg_iterates[1:,:] - pcg_iterates[:-1,:], axis=1)

    #Plots
    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    plt.title('Abstand der Iterierten vom Optimum', fontsize=16)
    plt.semilogy(err_cg, 'b-', linewidth=2)
    plt.semilogy(err_pcg,'r-', linewidth=2)
    plt.xlabel('Iteration $k$', fontsize=14)
    plt.ylabel('absoluter Fehler  $\\|x^{(k)} - \\hat{x}\\|$', fontsize=14)
    plt.legend(['cg','pcg'])

    plt.subplot(122)
    plt.title('Norm der Gradienten', fontsize=16)
    plt.semilogy(cg_gradient_norms,  'b-', linewidth=2)
    plt.semilogy(pcg_gradient_norms, 'r-', linewidth=2)
    plt.xlabel('Iteration $k$', fontsize=14)
    plt.ylabel('$\\|\\nabla f(x^{(k)})\\|$', fontsize=14)
    plt.legend(['cg', 'pcg'])

    plt.tight_layout()
    
    return b, cg_iterates[-1,:], pcg_iterates[-1,:]

def place_point(b, position):
    n = b.shape[0]
    m = int(np.sqrt(n))
    for i in range(0, 5):
        for j in range(0, 5):
            b[position+(i-2)*m+(j-2)] += 0.25
    for i in range(0, 3):
        for j in range(0, 3):
            b[position+(i-1)*m+(j-1)] += 0.25
    b[position] += 0.5

    return b