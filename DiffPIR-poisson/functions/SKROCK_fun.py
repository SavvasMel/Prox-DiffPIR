import math
import torch
import numpy as np

def SKROCK(X: torch.Tensor,Lipschitz_U,nStages: int,eta: float,dt_perc,gradU):

    # SK-ROCK parameters

    # First kind Chebyshev function

    T_s = lambda s,x: np.cosh(s*np.arccosh(x))

    # First derivative Chebyshev polynomial first kind

    T_prime_s = lambda s,x: s*np.sinh(s*np.arccosh(x))/np.sqrt(x**2 -1)

    # computing SK-ROCK stepsize given a number of stages

    # and parameters needed in the algorithm

    denNStag=(2-(4/3)*eta)

    rhoSKROCK = ((nStages - 0.5)**2) * denNStag - 1.5 # stiffness ratio

    dtSKROCK = dt_perc*rhoSKROCK/Lipschitz_U # step-size

    w0=1 + eta/(nStages**2) # parameter \omega_0

    w1=T_s(nStages,w0)/T_prime_s(nStages,w0) # parameter \omega_1

    mu1 = w1/w0 # parameter \mu_1

    nu1=nStages*w1/2 # parameter \nu_1

    kappa1=nStages*(w1/w0) # parameter \kappa_1

    # Sampling the variable X (SKROCK)

    Q=math.sqrt(2*dtSKROCK)*torch.randn_like(X) # diffusion term

    # SKROCK

    # SKROCK first internal iteration (s=1)

    XtsMinus2 = X.clone()

    Xts= X.clone() - mu1*dtSKROCK*gradU(X + nu1*Q) + kappa1*Q

    for js in range(2,nStages+1): # s=2,...,nStages SK-ROCK internal iterations

        XprevSMinus2 = Xts.clone()

        mu=2*w1*T_s(js-1,w0)/T_s(js,w0) # parameter \mu_js

        nu=2*w0*T_s(js-1,w0)/T_s(js,w0) # parameter \nu_js

        kappa=1-nu # parameter \kappa_js

        Xts= -mu*dtSKROCK*gradU(Xts) + nu*Xts + kappa*XtsMinus2

        XtsMinus2=XprevSMinus2

    return Xts # new sample produced by the SK-ROCK algorithm
