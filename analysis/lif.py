from numpy import *
from scipy.special import erfc, erf, erfcx, dawsn
from scipy.integrate import quad, dblquad
from scipy.optimize import newton

# collection of functions for LIF with OU stimulus
# derived from module fpt.py


def Psi2(s):
    return exp(s**2) * erfc(s)


def rate_whitenoise_benji(mu, sigma_x):
    """Rate of white-noise driven LIF neuron according Lindner PhD thesis p.48

    is more stable for large mu than Brunel version "rate_whitenoise()" below

    dx/dt=-x+mu+sqrt(2)*sigma_x*xi(t)

    """
    D = sigma_x**2
    a1 = (mu - 1) / sqrt(2 * D)
    a2 = mu / sqrt(2 * D)
    T, err = quad(func=Psi2, a=a1, b=a2)
    T = T * sqrt(pi)
    return 1.0 / T


def rate_whitenoise_benji2(Vrest, tau, D, Vth, Vreset):
    """
    tau*dV/dt=-V+mu+sqrt(2*D)*xi(t)
    Vreset, and Vth
    """
    mu = (Vrest - Vreset) / (Vth - Vreset)
    DD = D / (Vth - Vreset) ** 2 / tau
    a1 = (mu - 1) / sqrt(2 * DD)
    a2 = mu / sqrt(2 * DD)
    T, err = quad(func=Psi2, a=a1, b=a2)
    print(a1, a2, T, tau)
    T = T * sqrt(pi)
    return 1.0 / (T * tau)

# def cv_benji(mu, D):

#     def Psi2(x):
#         # exp(x^2)erfc(x)
#         return erfcx(x)
#     # calculate MFPT
#     a1 = (mu - 1) / sqrt(2 * D)
#     a2 = mu / sqrt(2 * D)
#     T, _ = quad(Psi2, a1, a2)
#     T *= sqrt(pi)

#     def psi4(y):
#         if y > a2: return 0.0
#         return exp(y**2)
#     def psi3(x):
#         z, _ = quad(psi4, a1, x)
#         return z
#     def outer(x):
#         #exp(-x^2)*(erfcx(x))^2 = exp(-x^2)*exp(2x^2)*erfc(x)^2 = exp(x^2)*erfc(x)^2
#         # used for numerical stability
#         return 2 * pi * exp(-x**2) * (erfcx(x)**2) * psi3(x)
#     B, _ = quad(outer, a1, 100) # Integrate to infinity (approx 100)
#     return sqrt(B) / T

def cv_benji(mu, D):
    # 1. Calculate Mean First Passage Time (T)
    # Using erfcx is safe here.
    def Psi2(x):
        return erfcx(x)
    
    a1 = (mu - 1) / sqrt(2 * D)
    a2 = mu / sqrt(2 * D)
    T, _ = quad(Psi2, a1, a2)
    T *= sqrt(pi)

    # 2. Calculate Variance Term (B)
    # We merge exp(-x^2) from 'outer' with exp(upper^2) from 'psi3'
    def outer(x):
        # Determine the upper limit of the inner integral
        upper = x if x < a2 else a2
        # Term 1: exp(upper^2) * dawson(upper) * exp(-x^2) becomes: exp(upper^2 - x^2) * dawson(upper)
        term1 = exp(upper**2 - x**2) * dawsn(upper)
        # Term 2: exp(a1^2) * dawson(a1) * exp(-x^2) becomes: exp(a1^2 - x^2) * dawson(a1)
        term2 = exp(a1**2 - x**2) * dawsn(a1)
        return 2 * pi * (erfcx(x)**2) * (term1 - term2)
    # Integrate to a sufficient upper bound (100 is usually enough for convergence)
    B, _ = quad(outer, a1, 100)
    return sqrt(B) / T

# def cv_benji(mu,sigma_x):
#     """CV of white-noise driven LIF neuron according Lindner PhD thesis p.48

#     dx/dt=-x+mu+sqrt(2)*sigma_x*xi(t)

#     """
#     def psi4(y):
#         return exp(y**2)*(y<a2)

#     def psi3(x):
#         a1=(mu-1)/sqrt(2*D)
#         z,err=quad(func=psi4,a=a1,b=x)

#     D=sigma_x**2
#     a1=(mu-1)/sqrt(2*D)
#     a2=mu/sqrt(2*D)
#     T,err=quad(func=Psi2,a=a1,b=a2)
#     T=T*sqrt(pi)
#     B=
#     return sqrt(V)/T


def Psi(s):
    """Psi function of Fourcaud, Brunel 2002"""
    return exp(s**2) * (1 + erf(s))


def rate_whitenoise(mu, sigma_x):
    """Rate of white-noise driven LIF neuron

    dx/dt=-x+mu+sqrt(2)*sigma_x*xi(t)

    """
    sigma = sqrt(2) * sigma_x
    a1 = -mu / sigma
    a2 = (1 - mu) / sigma
    T, err = quad(func=Psi, a=a1, b=a2)
    T = T * sqrt(pi)
    return 1 / T


def brunelrate(tau, mu, sigma_x):
    """Firing rate of LIF neuron driven by OUP with small tau (Fourcaud & Brunel 2002)"""
    alpha = 2.06525  # =-sqrt(2)*zeta(0.5)
    sigma = sqrt(2 * (tau + 1)) * sigma_x  # =sqrt(2*D)
    a1 = -mu / sigma + 0.5 * alpha * sqrt(tau)
    a2 = (1 - mu) / sigma + 0.5 * alpha * sqrt(tau)
    T, err = quad(func=Psi, a=a1, b=a2)
    T = T * sqrt(pi)
    return 1 / T


def brunelrate2(tau, mu, sigma):
    """Firing rate of LIF neuron driven by OUP with small tau (Fourcaud & Brunel 2002)"""
    alpha = 2.06525  # =-sqrt(2)*zeta(0.5)
    a1 = -mu / sigma + 0.5 * alpha * sqrt(tau)
    a2 = (1 - mu) / sigma + 0.5 * alpha * sqrt(tau)
    T, err = quad(func=Psi, a=a1, b=a2)
    T = T * sqrt(pi)
    return 1 / T


def n0(tau, sigma_x, mu):
    """Returns the Rice frequency of a LIF neuron driven by exponentially correlated noise with respect to threshold at x=1

    Dynamical model (LIF neuron):
    dx/dt=-x+y
    tau*dy/dt=-y+mu+sqrt(2*D)

    tau=Correlation time of OUP y(t)
    mu=constant input current
    sigma_x=membr. potential fluctuation
    D=(tau+1)*sigma_x^2=noise intensity
    sigma_y=D/tau

    """
    return exp(-0.5 * (1 - mu) ** 2 / sigma_x**2) / sqrt(tau) / 2 / pi


def umean(mu1, x0, taum, taus, dt):
    """
    returns u(t) and udot(t) if stimulus is filtered at synapses
    """
    x = zeros(len(mu1))
    y = zeros(len(mu1))
    E1 = exp(-dt / taus)
    E2 = exp(-dt / taum)
    for i in range(len(mu1) - 1):
        y[i + 1] = y[i] * E1 + 0.5 * dt / taus * (E1 * mu1[i] + mu1[i + 1])
        x[i + 1] = x[i] * E2 + 0.5 * dt / taum * (E2 * y[i] + y[i + 1])

    return [x + x0, (y - x) / taum]


def umean2(mu1, x0, taum, dt):
    """
    returns u(t) and udot(t) if current is directly applied to voltage equation, i.e. the stimulus is not filtered by the synapses
    """
    x = zeros(len(mu1))
    E1 = exp(-dt / taum)
    for i in range(len(mu1) - 1):
        x[i + 1] = x[i] * E1 + 0.5 * dt / taum * (E1 * mu1[i] + mu1[i + 1])
    return [x + x0, (mu1 - x) / taum]


def umean2var(mu1, x0, taum, dt):
    """
    returns u(t) and udot(t) if current is directly applied to voltage equation, i.e. the stimulus is not filtered by the synapses
    like umean2 but numerical differentiation to get udot
    """
    x = zeros(len(mu1))
    E1 = exp(-dt / taum)
    for i in range(len(mu1) - 1):
        x[i + 1] = x[i] * E1 + 0.5 * dt / taum * (E1 * mu1[i] + mu1[i + 1])
    return [x + x0, ttools.D1vec(x, dt)]


def n1(u, udot, sigma_x, sigma_v, theta):
    g = exp(-0.5 * (theta - u) ** 2 / sigma_x**2) / (sqrt(2 * pi) * sigma_x)
    h = sigma_v * (
        exp(-0.5 * udot**2 / sigma_v**2) / sqrt(2 * pi)
        + 0.5 * udot / sigma_v * erfc(-udot / sqrt(2) / sigma_v)
    )
    n1 = g * h
    return [n1, g, h]


def n1cond(u, udot, t, sigma_x, sigma_v):
    b = 1.0 - u
    bdot = -udot
    rxx = Rxx(sigma_x, sigma_v, t) / sigma_x**2
    rxv = Rxv(sigma_x, sigma_v, t) / sigma_x / sigma_v
    rvv = Rvv(sigma_x, sigma_v, t) / sigma_v**2
    A1 = 1.0 - rxx**2 - rxv**2
    A2 = 1.0 - rvv**2 - rxv**2
    A3 = rxv * (rxx - rvv)
    Q = A1 * A2 - A3**2
    kappa = A1 / sigma_v**2 / Q
    beta = (A1 * bdot / sigma_v**2 + A3 * b / sigma_x / sigma_v) / Q
    alpha = beta / sqrt(2 * kappa)
    B = exp(
        -0.5
        / Q
        * (
            A1 * bdot**2 / sigma_v**2
            + 2 * A3 * b * bdot / sigma_x / sigma_v
            + A2 * b**2 / sigma_x**2
        )
    ) / (2 * pi * sigma_x * sigma_v * sqrt(Q))
    return (
        B * Q * sigma_v**2 / A1 * (1 - sqrt(pi) * alpha * exp(alpha**2) * erfc(alpha))
    )


def h(udot, sigma_v):
    return exp(-0.5 * udot**2 / sigma_v**2) + sqrt(pi / 2) * udot / sigma_v * erfc(
        -udot / sqrt(2) / sigma_v
    )


def H(t, tau, sigma):
    return exp(0.5 * (sigma / tau) ** 2 - t / tau) * erfc(
        (sigma / tau - t / sigma) / sqrt(2)
    )


def Rxx(sigma_x, sigma_v, t):
    vx = sigma_x**2
    vv = sigma_v**2  # =D/gam
    tau = vx / vv
    gam = (tau + 1.0) / tau
    Om = sqrt(gam**2 / 4 - vv / vx)
    expo = exp(-0.5 * gam * t)
    return vx * (gam / 2 / Om * sinh(Om * t) + cosh(Om * t)) * expo


def Rxv(sigma_x, sigma_v, t):
    """
    Computes <x(0)v(t)> of LIF with exponential synapses
    """
    vx = sigma_x**2
    vv = sigma_v**2  # =D/gam
    tau = vx / vv
    gam = (tau + 1.0) / tau
    Om = sqrt(gam**2 / 4 - vv / vx)
    expo = exp(-0.5 * gam * t)
    return -vv / Om * expo * sinh(Om * t)


def Rvv(sigma_x, sigma_v, t):
    """
    Computes <v(0)v(t)> of LIF with exponential synapses
    """
    vx = sigma_x**2
    vv = sigma_v**2  # =D/gam
    tau = vx / vv
    gam = (tau + 1.0) / tau
    Om = sqrt(gam**2 / 4 - vv / vx)
    expo = exp(-0.5 * gam * t)
    return vv * expo * (-0.5 * gam / Om * sinh(Om * t) + cosh(Om * t))


def n2(t1, t2, u1, u2, udot1, udot2, sigma_x, sigma_v, theta):
    """Returns two-point distribution function for upcrossing process generated by LIF dynamics without resetting driven by OU noise

    Use dimensionless quantities for sigma_x, sigma_v, t1, t2, theta
    """

    def P4(x1, x2, x3, x4):
        x = array([x1, x2, x3, x4])
        if d != 0.0:
            y = linalg.solve(C4, x)
            a = -0.5 * dot(x, y)
            #            a=-0.5*dot(x,dot(iC4,x))
            z = log((2 * pi) ** 2) + 0.5 * log(d)
            # print a-z
            return exp(a - z)
        else:
            return 0.0

    def func1(v1):
        def func(v2):
            return (v1 + udot1) * (v2 + udot2) * P4(theta - u1, theta - u2, v1, v2)

        y, err = quad(func, -udot2, 4 * sigma_v)
        return y

    vx = sigma_x**2
    vv = sigma_v**2  # =D/gam
    tau = vx / vv
    gam = (tau + 1.0) / tau
    Om = sqrt(gam**2 / 4 - vv / vx)

    T = t2 - t1
    expo = exp(-0.5 * gam * T)
    c = vx * (gam / 2 / Om * sinh(Om * T) + cosh(Om * T)) * expo
    dc = -vv / Om * expo * sinh(Om * T)
    ddc = -vv * expo * (-0.5 * gam / Om * sinh(Om * T) + cosh(Om * T))

    C4 = array([[vx, c, 0, dc], [c, vx, -dc, 0], [0, -dc, vv, -ddc], [dc, 0, -ddc, vv]])
    d = linalg.det(C4)
    #    d2=dc**4-2*c*dc**2*ddc+c**2*ddc*2-c**2*vv**2-2*dc**2*vv*vx-ddc**2*vx**2+vv**2*vx**2
    #    iC4=array([[-dc**2*vv-ddc**2*vx+vv**vx,-dc**2*ddc+c*ddc**2-c*vv*2,-c*dc*vv-dc*ddc*vx,dc**3-c*dc*ddc-dc*vv*vx],[-dc**2*ddc+c*ddc**2-c*vv**2,-dc**2*vv-ddc**2*vx+vv**2*vx,-dc**3+c*dc*ddc+dc*vv*vx,c*dc*vv+dc*ddc*vx],[-c*dc*vv-dc*ddc*vx,-dc**3+c*dc*ddc+dc*vv*vx,-c**2*vv-dc**2*vx+vv*vx**2,c*dc**2-c**2*ddc+ddc*vx**2],[dc**3-c*dc*ddc-dc*vv*vx,c*dc*vv+dc*ddc*vx,c*dc**2-c**2*ddc+ddc*vx**2,-c**2*vv-dc**2*vx+vv*vx**2]])/d

    return quad(func1, -udot1, 4 * sigma_v)


def n2_var(t1, t2, u1, u2, udot1, udot2, sigma_x, sigma_v, theta):
    """Returns two-point distribution function for upcrossing process generated by LIF dynamics without resetting driven by OU noise

    Use dimensionless quantities for sigma_x, sigma_v, t1, t2, theta
    """

    def P4(x1, x2, x3, x4):
        x = array([x1, x2, x3, x4])
        if d != 0.0:
            # y=linalg.solve(C4,x)
            a = -0.5 * dot(x, dot(iC4, x))
            z = log((2 * pi) ** 2) + 0.5 * log(d)
            return exp(a - z)
        else:
            return 0.0

    def func(v1, v2):
        return (v1 + udot1) * (v2 + udot2) * P4(theta - u1, theta - u2, v1, v2)

    vx = sigma_x**2
    vv = sigma_v**2  # =D/gam
    tau = vx / vv
    gam = (tau + 1.0) / tau
    Om = sqrt(gam**2 / 4 - vv / vx)

    T = t2 - t1
    expo = exp(-0.5 * gam * T)
    c = vx * (gam / 2 / Om * sinh(Om * T) + cosh(Om * T)) * expo
    dc = -vv / Om * expo * sinh(Om * T)
    ddc = -vv * expo * (-0.5 * gam / Om * sinh(Om * T) + cosh(Om * T))

    C4 = array([[vx, c, 0, dc], [c, vx, -dc, 0], [0, -dc, vv, -ddc], [dc, 0, -ddc, vv]])
    d = linalg.det(C4)
    d2 = (
        dc**4
        - 2 * c * dc**2 * ddc
        + c**2 * ddc * 2
        - c**2 * vv**2
        - 2 * dc**2 * vv * vx
        - ddc**2 * vx**2
        + vv**2 * vx**2
    )
    iC4 = (
        array(
            [
                [
                    -(dc**2) * vv - ddc**2 * vx + vv**vx,
                    -(dc**2) * ddc + c * ddc**2 - c * vv * 2,
                    -c * dc * vv - dc * ddc * vx,
                    dc**3 - c * dc * ddc - dc * vv * vx,
                ],
                [
                    -(dc**2) * ddc + c * ddc**2 - c * vv**2,
                    -(dc**2) * vv - ddc**2 * vx + vv**2 * vx,
                    -(dc**3) + c * dc * ddc + dc * vv * vx,
                    c * dc * vv + dc * ddc * vx,
                ],
                [
                    -c * dc * vv - dc * ddc * vx,
                    -(dc**3) + c * dc * ddc + dc * vv * vx,
                    -(c**2) * vv - dc**2 * vx + vv * vx**2,
                    c * dc**2 - c**2 * ddc + ddc * vx**2,
                ],
                [
                    dc**3 - c * dc * ddc - dc * vv * vx,
                    c * dc * vv + dc * ddc * vx,
                    c * dc**2 - c**2 * ddc + ddc * vx**2,
                    -(c**2) * vv - dc**2 * vx + vv * vx**2,
                ],
            ]
        )
        / d
    )

    return dblquad(
        func,
        -udot1,
        4 * sigma_v,
        lambda v1: -udot2,
        lambda v1: 4 * sigma_v,
        epsabs=finfo(float).eps,
    )


def n2_simpson(t1, t2, u1, u2, udot1, udot2, sigma_x, sigma_v, theta, n):
    """Returns two-point distribution function for upcrossing process generated by LIF dynamics without resetting driven by OU noise

    Use dimensionless quantities for sigma_x, sigma_v, t1, t2, theta
    n=number of intervals for 1D-integration
    """

    def P4(x1, x2, x3, x4):
        x = array([x1, x2, x3, x4])
        if d != 0.0:
            # y=linalg.solve(C4,x)
            a = -0.5 * dot(x, dot(iC4, x)) / d
            return exp(a) / (4 * pi**2 * sqrt(d))
        else:
            return 0.0

    def func1(v1):
        def func(v2):
            return (v1 + udot1) * (v2 + udot2) * P4(theta - u1, theta - u2, v1, v2)

        y = ttools.simpson(func, -udot2, 4 * sigma_v, n)
        return y

    vx = sigma_x**2
    vv = sigma_v**2  # =D/gam
    tau = vx / vv
    gam = (tau + 1.0) / tau
    Om = sqrt(gam**2 / 4 - vv / vx)

    T = t2 - t1
    expo = exp(-0.5 * gam * T)
    c = vx * (gam / 2 / Om * sinh(Om * T) + cosh(Om * T)) * expo
    dc = -vv / Om * expo * sinh(Om * T)
    ddc = -vv * expo * (-0.5 * gam / Om * sinh(Om * T) + cosh(Om * T))

    C4 = array([[vx, c, 0, dc], [c, vx, -dc, 0], [0, -dc, vv, -ddc], [dc, 0, -ddc, vv]])
    d = linalg.det(C4)
    d2 = (
        dc**4
        - 2 * c * dc**2 * ddc
        + c**2 * ddc * 2
        - c**2 * vv**2
        - 2 * dc**2 * vv * vx
        - ddc**2 * vx**2
        + vv**2 * vx**2
    )
    iC4 = array(
        [
            [
                -(dc**2) * vv - ddc**2 * vx + vv**vx,
                -(dc**2) * ddc + c * ddc**2 - c * vv * 2,
                -c * dc * vv - dc * ddc * vx,
                dc**3 - c * dc * ddc - dc * vv * vx,
            ],
            [
                -(dc**2) * ddc + c * ddc**2 - c * vv**2,
                -(dc**2) * vv - ddc**2 * vx + vv**2 * vx,
                -(dc**3) + c * dc * ddc + dc * vv * vx,
                c * dc * vv + dc * ddc * vx,
            ],
            [
                -c * dc * vv - dc * ddc * vx,
                -(dc**3) + c * dc * ddc + dc * vv * vx,
                -(c**2) * vv - dc**2 * vx + vv * vx**2,
                c * dc**2 - c**2 * ddc + ddc * vx**2,
            ],
            [
                dc**3 - c * dc * ddc - dc * vv * vx,
                c * dc * vv + dc * ddc * vx,
                c * dc**2 - c**2 * ddc + ddc * vx**2,
                -(c**2) * vv - dc**2 * vx + vv * vx**2,
            ],
        ]
    )

    return ttools.simpson(func1, -udot1, 4 * sigma_v, n)


def Pvv(v1, v2, t1, t2, sigma_x, sigma_v):
    """
    Returns P4(x1,x2,v1,v2)|{x1=x2=0},t1-t2=T
    """

    vx = sigma_x**2
    vv = sigma_v**2  # =D/gam
    tau = vx / vv
    gam = (tau + 1.0) / tau
    Om = sqrt(gam**2 / 4 - vv / vx)

    T = t2 - t1
    expo = exp(-0.5 * gam * T)
    c = vx * (gam / 2 / Om * sinh(Om * T) + cosh(Om * T)) * expo
    dc = -vv / Om * expo * sinh(Om * T)
    ddc = -vv * expo * (-0.5 * gam / Om * sinh(Om * T) + cosh(Om * T))

    C4 = array([[vx, c, 0, dc], [c, vx, -dc, 0], [0, -dc, vv, -ddc], [dc, 0, -ddc, vv]])
    d = linalg.det(C4)
    d2 = (
        dc**4
        - 2 * c * dc**2 * ddc
        + c**2 * ddc * 2
        - c**2 * vv**2
        - 2 * dc**2 * vv * vx
        - ddc**2 * vx**2
        + vv**2 * vx**2
    )
    iC4 = (
        array(
            [
                [
                    -(dc**2) * vv - ddc**2 * vx + vv**vx,
                    -(dc**2) * ddc + c * ddc**2 - c * vv * 2,
                    -c * dc * vv - dc * ddc * vx,
                    dc**3 - c * dc * ddc - dc * vv * vx,
                ],
                [
                    -(dc**2) * ddc + c * ddc**2 - c * vv**2,
                    -(dc**2) * vv - ddc**2 * vx + vv**2 * vx,
                    -(dc**3) + c * dc * ddc + dc * vv * vx,
                    c * dc * vv + dc * ddc * vx,
                ],
                [
                    -c * dc * vv - dc * ddc * vx,
                    -(dc**3) + c * dc * ddc + dc * vv * vx,
                    -(c**2) * vv - dc**2 * vx + vv * vx**2,
                    c * dc**2 - c**2 * ddc + ddc * vx**2,
                ],
                [
                    dc**3 - c * dc * ddc - dc * vv * vx,
                    c * dc * vv + dc * ddc * vx,
                    c * dc**2 - c**2 * ddc + ddc * vx**2,
                    -(c**2) * vv - dc**2 * vx + vv * vx**2,
                ],
            ]
        )
        / d
    )

    x = array([0.0, 0.0, v1, v2])
    if d != 0.0:
        # y=linalg.solve(C4,x)
        # yy=dot(Vh.T,W*dot(U.T,x))
        # xx=dot(C4,yy)
        # print abs(yy-y),y
        a = -0.5 * dot(x, dot(iC4, x))
        z = log((2 * pi) ** 2) + 0.5 * log(d)
        # print a,z
        return exp(a - z)
    else:
        return 0.0


def beta(t, taum, taus):
    return (t > 0) * (exp(-t / taum) - exp(-t / taus)) / (taum - taus)


def lifadap_limitcycle(mu, gamma, tau, delta):
    """delta is the jump size"""

    def F(t):
        et = exp(-t / tau)
        eg = exp(-gamma * t)
        aa = delta / (1 - et)
        return mu / gamma * (1 - eg) - aa * tau / (1 - gamma * tau) * (eg - et) - 1

    def dF(t):
        et = exp(-t / tau)
        eg = exp(-gamma * t)
        aa = delta / (1 - et)
        return (
            mu * eg
            - aa * tau / (1 - gamma * tau) * (et / tau - gamma * eg)
            + delta * et / (1 - et) ** 2 / (1 - gamma * tau) * (eg - et)
        )

    Tdet = newton(F, (1 + delta) / mu, fprime=dF)
    aa = delta / (1 - exp(-Tdet / tau))
    return (Tdet, aa)


def prc_lifadap(t, mu, gamma, tau, delta):
    tt, aa = lifadap_limitcycle(mu, gamma, tau, delta)
    return exp(-gamma * (tt - t)) / (mu - gamma - aa + delta)


def rho_lifadap(lag, mu, tau, delta, Tmean):
    """delta is the jump size"""
    tt = Tmean
    a = exp(-tt / tau)
    aa = delta / (1 - a)
    theta = 1 - aa * exp(-tt) * (1 - exp(-tt * (1 / tau - 1))) / (1 - tau) / (
        mu - 1.0 - aa + delta
    )
    return (
        -a
        * (1 - theta)
        * (1 - a**2 * theta)
        / (1 + a**2 - 2 * a**2 * theta)
        * (a * theta) ** (lag - 1)
    )


def sumrho_lifadap(mu, tau, delta, Tmean):
    """delta is the jump size"""
    tt = Tmean
    a = exp(-tt / tau)
    aa = delta / (1 - a)
    theta = 1 - aa * exp(-tt) * (1 - exp(-tt * (1 / tau - 1))) / (1 - tau) / (
        mu - 1.0 - aa + delta
    )
    return (
        -a
        * (1 - theta)
        * (1 - a**2 * theta)
        / (1 + a**2 - 2 * a**2 * theta)
        / (1 - a * theta)
    )


########################################################
#                 green noise                          #
########################################################


def corrfunc_green(t, D, alpha, ta, ts):
    """
    correlation function not normalized
    """
    DD = D * ta**2 / (ta**2 - ts**2)
    lam = ts / ta
    a = 1 - lam**2 * (alpha + 1) ** 2
    b = alpha * (alpha + 2)
    return DD * (a / ts * exp(-t / ts) + b / ta * exp(-t / ta))


def isih_colored(t, C, mu, gamma, k):
    """
    k-th-order interval density for dot{V}=mu+eta(t), Vth=1, C is corrfunc of eta
    """
    dt = t[1] - t[0]
    eps2 = C[0] / mu**2
    C = C / C[0]
    Tm = log(mu / (mu - gamma)) / gamma
    gam2 = dt / Tm * cumsum(C)
    gam1 = dt / Tm * cumsum(gam2)
    return (
        exp(-((t / Tm - k) ** 2) / (4 * eps2 * gam1))
        / (2 * Tm * sqrt(4 * pi * eps2 * gam1**3))
        * (
            ((k - t / Tm) * gam2 + 2 * gam1) ** 2 / (2 * gam1)
            - eps2 * (gam2**2 - 2 * gam1 * C)
        )
    )


def kappa2_colored_1(t, C, mu, k):
    """
    2nd cumulant of k0th-order interval for dot{V}=mu+eta(t), Vth=1, C is corrfunc of eta
    t[0]=0
    """
    dt = t[1] - t[0]
    eps2 = C[0] / mu**2
    Tm = 1.0 / mu
    i = int(k * Tm / dt)
    c = C[: i + 1] / C[0]  # integration only up to Tm
    gam2 = dt * cumsum(c) / Tm
    gam1 = dt * sum(gam2) / Tm
    k2_0 = 2 * Tm**2 * eps2 * gam1
    k2_1 = 2 * Tm**2 * eps2**2 * (gam2[-1] ** 2 + c[-1] * gam1)
    return k2_0 + k2_1


def kappa2_colored_0(t, C, mu, k):
    """
    2nd cumulant of k0th-order interval for dot{V}=mu+eta(t), Vth=1, C is corrfunc of eta
    t[0]=0
    """
    dt = t[1] - t[0]
    eps2 = C[0] / mu**2
    Tm = 1.0 / mu
    i = int(k * Tm / dt)
    c = C[: i + 1] / C[0]  # integration only up to Tm
    gam2 = dt * cumsum(c) / Tm
    gam1 = dt * sum(gam2) / Tm
    k2_0 = 2 * Tm**2 * eps2 * gam1
    return k2_0


def rhon_colored_1(k, t, C, mu):
    """
    SCC for dot{V}=mu+eta(t), Vth=1, C is corrfunc of eta
    """
    k2 = kappa2_colored_1(t, C, mu, k + 1)
    k1 = kappa2_colored_1(t, C, mu, k)
    k0 = kappa2_colored_1(t, C, mu, k - 1)
    return (k2 - 2 * k1 + k0) / (2 * kappa2_colored_1(t, C, mu, 1))


def rhon_colored_0(k, t, C, mu):
    """
    SCC for dot{V}=mu+eta(t), Vth=1, C is corrfunc of eta
    """
    k2 = kappa2_colored_0(t, C, mu, k + 1)
    k1 = kappa2_colored_0(t, C, mu, k)
    k0 = kappa2_colored_0(t, C, mu, k - 1)
    return (k2 - 2 * k1 + k0) / (2 * kappa2_colored_0(t, C, mu, 1))
