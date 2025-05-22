import numpy as np
import matplotlib.pyplot as plt


def fun_phi (r) :
    return (r)


def lambda(x, y):
    return (y/fun_phi(0))

def s_create (r) :
    x = find_control_points () #fonction à faire 
    y = find_position() #fonction à faire 
    n = len(x)   
    acc = 0
    for i in range(0, n ):
        lamb = lambda( x[i], y[i])
        phi = fun_phi(abs(r-x[i]))
        acc = acc + lamb * phi 
    return acc 

    