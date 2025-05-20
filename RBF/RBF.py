import numpy as np
import matplotlib.pyplot as plt

'''
Methode 1 : Operation-intensive 
1) Solve for field vectors y^(x/y/z)
2) Update volume mesh points.
=> O(Ns^3)

Methode 2 : Memory-intensive approach 
1) The system solution is performed in a preprocessing stage
2) Definition of a single transfer matrix H (Nv*Ns) -> uv = Hus where H= psi * phi^-1 ∈ R^(Nv×Ns)
=> infeasible for large meshes

=>  we need to implement it using some data-reduction method


Choice of the points : Greedy full point selection algorithm

    procedure
    init: 
        xa <- {...}
        f^exact = u^(x/y/z)
    main
        while Nactive < Nred do 
            alpha_a = phi_a^-1 f_a^exact
            fêval = psi*alpha_a
            e = F(f^exact - f^eval)
            i_worst = arg max e(i) (for i =1 to i=Nx)
            xa <- x(iworst)
        end while
    output
        return xa
    end procedure 
''' 
