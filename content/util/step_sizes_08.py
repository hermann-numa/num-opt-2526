def wolfe_powell_step_size(f, g, x, d, gamma = 1e-4, rho = 0.9):
    '''
        Wolfe-Powell-Schrittweitenregel

        Parameter:
            A, b        : Parameter der Funktion f
            x           : Aktueller Wert der Iteration
            d           : Aktuelle Richtung des Abstiegsverfahrens
            gamma, rho  : Parameter der Wolfe-Powell-Schritweitenregel

        Rückgabewert:
            alpha: Wolfe-Powell-Schrittweite
    '''
    
    if gamma <= 0 or gamma >= 0.5:
        raise ValueError(f'Der Parameter gamma muss im Intervall (0, 0.5) liegen, aber gamma={gamma}!')
    if rho <= gamma or rho >= 1:
        raise ValueError(f'Der Parameter rho muss im Intervall (gamma, 1) liegen, aber rho={rho} und gamma={gamma}!')  

    tol = 1e-12
    alpha = 1
    f_x = f(x)
    g_x = g(x)
    c = g_x @ d
    if c >= 0:
        return alpha
    phi = lambda a : f(x + a*d)
    phi_prime = lambda a : g(x + a*d) @ d
    psi = lambda a : phi(a) - f_x - a*gamma*c

    if phi(alpha) <= 0:
        if phi_prime(alpha) >= rho * c:
            return alpha
        alpha_min = alpha
        while psi(alpha) <= 0:
            alpha *= 2
        alpha_max = alpha
    else:
        alpha_max = alpha
        while (psi(alpha) > 0 or phi_prime(alpha) >= rho * c) and alpha > tol:
            alpha /= 2 
        alpha_min = alpha
    
    alpha = 0.5 * (alpha_min + alpha_max)
    while (psi(alpha) > 0 or phi_prime(alpha) < rho * c) and (alpha_max - alpha_min) > tol:
        if psi(alpha) > 0:
            alpha_max = alpha
        else:
            alpha_min = alpha
        alpha = 0.5 * (alpha_min + alpha_max)

    return alpha

def constant_step_size(f, g, x, d):
    return 1