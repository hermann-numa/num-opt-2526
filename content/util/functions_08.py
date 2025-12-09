import numpy as np

def  rosenbrock_funktion(x):
    return 100.0 *(x[1] - x[0]**2)**2 + (1.0 - x[0])**2

def  rosenbrock_gradient ( x ) :
    return np.array(
        [
            400.0 * x[0] * (x[0]**2 - x[1]) + 2.0 * (x[0] - 1.0),
            200.0 * (x[1] - x[0]**2),
        ],
    )

def  rosenbrock_hesse(x):
    return np.array(
        [
            [
                1200.0 * x[0]**2 - 400.0 * x[1] + 2.0, 
                -400.0 * x[0],
            ],
            [
                -400.0 * x[0],
                200.0,
            ],
        ],
    )

def himmelblau_funktion(x):
    return (x[0]**2 + x[1] - 11.0 )**2 + (x[0] + x[1]**2 - 7.0)**2

def  himmelblau_gradient(x):
    return np.array(
        [
            4.0 * x[0] * (x[0]**2 + x[1] - 10.5) + 2.0 * x[1]**2 - 14.0, 
            4.0 * x[1] * (x[1]**2 + x[0] - 6.5 ) + 2.0 * x[0]**2 - 22.0,
        ],
    )
    
def  himmelblau_hesse(x):
    return np.array(
        [
            [
                12.0 * x[0]**2 + 4.0 * x[1] - 42.0,
                4.0 * (x[0] + x[1]),
            ],
            [
                4.0 * (x[0] + x[1]),
                12.0 * x[1]**2 + 4.0 * x[0] - 26.0,
            ],
        ],
    )

def sattel_funktion(x):
    return x[0] * (x[0]**2 - 2.0 * x[1]**2) + 0.15 * (x[0]**2 + x[1]**2 - 1.0)**2

def sattel_gradient(x):
    return np.array(
        [
            x[0]**2 * (0.6 * x[0] + 3.0) + x[1]**2 * (0.6 * x[0] - 2.0) - 0.6 * x[0],
            0.6 * x[1] * (x[0]**2 + x[1]**2 - 1) - 4.0 * x[0] * x[1],
        ]
    )

def sattel_hesse(x):
    xx = x[0]
    yy = x[1]
    x2 = xx * xx
    y2 = yy * yy
    xy = xx * yy
    return np.array(
        [
            [
                6.0 * x[0] + 1.8 * x[0]**2 + 0.6 * x[1]**2 - 1.0,
                1.2 * x[0] * x[1] - 4.0 * x[1],
            ],
            [
                1.2  * x[0] * x[1] - 4.0 * x[1],
                -4.0 * x[0] + 1.8 * x[1]**2 + 0.6 * x[0]**2 - 1.0,
            ],
        ],
    )
