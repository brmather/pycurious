import numpy as np
from scipy.optimize import minimize
from scipy.special import gamma, kv, lambertw
import warnings

def func(x, Phi_exp, kh):
    """
    Function to minimise
    """
    beta = x[0]
    zt = x[1]
    dz = x[2]
    C = x[3]
    with warnings.catch_warnings() as w:
        warnings.simplefilter("ignore")
        Phi_syn = bouligand2009(beta, zt, dz, kh, C)
        if np.isnan(Phi_syn).any():
            Phi_syn.fill(1e5)

    misfit = np.linalg.norm((Phi_exp - Phi_syn))**2

    # print (Phi_syn), misfit
#     misfit = np.sum((Phi_exp - pycurious.bouligand2009(beta, zt, dz, kh, C))**2/np.sqrt(sigma2)**2)
    misfit += (beta - 5.44)**2/0.62**2
    misfit += (zt - 0.14)**2/0.28**2
    misfit += (dz - 22.73)**2/29.63**2
    misfit += (C - 11.88)**2/1.38**2
    return misfit

def bouligand2009(beta, zt, dz, kh, C=0.0):
    """
    Calculate the synthetic radial power spectrum of
    magnetic anomalies

    Equation (4) of Bouligand et al. (2009)

    Parameters
    ----------
     beta  : fractal parameter
     zt    : top of magnetic sources
     dz    : thickness of magnetic sources
     kh    : norm of the wave number in the horizontal plane
     C     : field constant (Maus et al., 1997)

    Returns
    -------
     Phi  : radial power spectrum of magnetic anomalies

    References
    ----------
     Bouligand, C., J. M. G. Glen, and R. J. Blakely (2009), Mapping Curie
       temperature depth in the western United States with a fractal model for
       crustal magnetization, J. Geophys. Res., 114, B11104,
       doi:10.1029/2009JB006494
     Maus, S., D. Gordon, and D. Fairhead (1997), Curie temperature depth
       estimation using a self-similar magnetization model, Geophys. J. Int.,
       129, 163-168, doi:10.1111/j.1365-246X.1997.tb00945.x
    """
    khdz = kh*dz
    coshkhdz = np.cosh(khdz)

    Phi1d = C - 2.0*kh*zt - (beta-1.0)*np.log(kh) - khdz
    A = np.sqrt(np.pi)/gamma(1.0+0.5*beta) * \
        (0.5*coshkhdz*gamma(0.5*(1.0+beta)) - \
        kv((-0.5*(1.0+beta)), khdz) * np.power(0.5*khdz,(0.5*(1.0+beta)) ))
    Phi1d += np.log(A)
    return Phi1d


S = np.array([ 22.16409774,  19.95258494,  18.27873722,  17.10575637,\
               16.53959747,  16.31539575,  15.69619005,  15.29953307,\
               14.83475976,  14.54031396,  14.33361716,  13.81764026,\
               13.5176055 ,  13.27386563,  13.03493328,  12.88581369,\
               12.61998358,  12.48616749,  12.11261083,  12.13079154,\
               11.85440661,  11.79244826,  11.66823202,  11.40231744,\
               11.32521296,  11.13634007,  11.10650999,  10.94822598,\
               10.78032794,  10.66593304,  10.55815845,  10.56805594,\
               10.33514462,  10.22026537,  10.22945756,  10.09275259,\
               10.11562101,   9.85061009,   9.87165772,   9.85976847,\
                9.73954992,   9.72021054,   9.52959744,   9.59582531,\
                9.50927273,   9.44691364,   9.39293966,   9.33097387,   9.33191784])

k = np.array([ 0.09237068,  0.15443902,  0.21486191,  0.282031  ,  0.33452175,\
               0.4020083 ,  0.46381582,  0.5290359 ,  0.59611689,  0.65875047,\
               0.71442694,  0.77929408,  0.84320843,  0.90939865,  0.96863721,\
               1.03155349,  1.09130906,  1.15699532,  1.22338685,  1.28109263,\
               1.34606572,  1.40861721,  1.46920824,  1.53421046,  1.59675833,\
               1.66109235,  1.7233503 ,  1.78492941,  1.84835837,  1.90933735,\
               1.973275  ,  2.03650008,  2.0983359 ,  2.16386249,  2.22416603,\
               2.2876692 ,  2.34841931,  2.41225232,  2.47774604,  2.5379226 ,\
               2.59980401,  2.66346692,  2.72735801,  2.79146076,  2.85220715,\
               2.91443307,  2.97643404,  3.04008782,  3.10369815])

w = 1.0

beta0 = 3.0
zt0 = 1.0
dz0 = 40.0
C0 = 5.0
x0 = np.array([beta0, zt0, dz0, C0])

lb = np.zeros_like(x0)
ub = np.zeros_like(x0)
# ub = np.array([50., 1, 100, 1e99])
ub = [None]*len(lb)

# xi = func([5, 0., 45., 6.], S, k)
xi = func(x0, S, k)

res = minimize(func, x0, args=(S, k), method='TNC', bounds=list(zip(lb,ub)))
print("beta={:.2f}, zt={:.2f}, dz={:.2f}, C={:.2f}".format(*res.x))