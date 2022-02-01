# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from math import *
from typing import TextIO

import numpy as np
from numpy import ndarray


def lmbda():
    ''' returns lambda '''
    return sqrt(1.01)


def want_wilson():
    ''' True if Wilson's tns wanted '''
    return False


def want_vivaldo():
    '''True if Vivaldo's tns wanted'''
    return True


def n_band():
    ''' returns number of band levels; must be odd '''
    if lmbda() > 1.5:
        return 40
    else:
        return 10001


class Param:
    """
   Gamma
   """

    def __init__(self, pgamma):
        self._gamma = pgamma

    def gam(self) -> float:
        return self._gamma


def gam(value=1.e-10):
    """ returns Gamma """
    parameters = Param(value)
    return parameters.gam()


def ed():
    ''' returns epsilon_d '''
    return 0.0


# NRG / eNRG

def wilson_factor(n):
    """ return Wilson's ksi """
    lmb2 = np.power(lmbda(), 2.0)
    lm_pw_np1 = np.power(lmb2, -n - 1.0)
    lm_pw_2np1 = np.power(lmb2, -2.0 * n - 1)
    lm_pw_2np3 = np.power(lmb2, -2.0 * n - 3.)
    return (
            (1. - lm_pw_np1)
            / sqrt(1. - lm_pw_2np1)
            / sqrt(1. - lm_pw_2np3)
    )


def alamb():
    """returns NRG's correction factor A_Lambda """
    lamb2 = np.power(lmbda(), 2.0)
    ret = (lamb2 + 1.0) / (lamb2 - 1.0) * np.log(lmbda())

    return ret


def dtilde(nn: int):
    """returns effective bandwidth"""
    ret = 0.0
    if want_wilson():
        ret = 1.0 + np.power(lmbda(), -2.0)
        ret /= 2.0
        ret *= np.power(lmbda(), 1. - nn)

    elif want_vivaldo():
        ret = 1.0 - np.power(lmbda(), -2)
        ret /= 2.0 * np.log(lmbda())
        ret *= np.power(lmbda(), 1.0 - nn)
    else:
        ret = np.power(lmbda(), 0.5 - nn)

    return ret


def calc_tn(offset: int, n: int):
    ''' returns nth NRG codiagonal element, multiplied by lamb**(n)'''
    lamb = lmbda()
    if want_wilson() or want_vivaldo():
        ret = dtilde(1) * np.power(lamb, -n)
        if offset == 0:
            ret *= wilson_factor(n)
        elif offset == 1:  #
            ksi_z05_ = [0.902905,
                        1.39329,
                        2.20224,
                        2.27685,
                        2.06959,
                        2.01367,
                        2.00325,
                        2.00058,
                        2.00000,
                        2.00000,
                        2.00000]
            if n < 10:
                ret *= ksi_z05_[n]
            else:
                ret *= lamb
        else:  # wilson and offset > 1
            print(f"Unexpected offset ({offset}): calc_tn")
            exit(1)
        return ret

    # eNRG
    else:
        if n == 0:  # first coupling is -sqrt(2), except if offset = 0 => coupling is -1
            if offset == 0:
                return -1.0
            else:
                return -sqrt(2.)
        if n < offset:  # n > 0, but n <offset
            return -1.0
        else:  # n >= offset => start discretization
            nf = n - offset  # f_n counter
            return np.power(lamb, -nf - 0.5)


def ham(nn, offset) -> np.array([float, float]):
    '''constructs eNRG conduction-band hamiltonian '''
    ret__ = np.zeros([nn + 1, nn + 1])
    for col in range(nn):
        n = col
        ret__[col, col + 1] = ret__[col + 1, col] = (
            calc_tn(offset, n)
        )

    return ret__


def diag_ham(nn, offset):
    '''returns eigenvalues and eigenvectors of NRG conduction-band hamiltonian'''
    ham__ = ham(nn, offset)

    eval_, evect__ = np.linalg.eigh(ham__)
    evect__ = evect__.transpose()

    adim_erg_ = eval_ / dtilde(nn)

    return eval_, evect__, adim_erg_


def ham_pot(kk, nn, offset):
    """ return hamiltonian with potential scattering kk """
    ham__ = ham(nn, offset)
    if want_wilson():
        kk *= alamb()
    ham__[0][0] = 2. * kk

    return ham__


def diag_ham_pot(kk, nn, offset):
    """ return eigenvalues and eigenvectors of hamiltonian with potential scattering kk"""
    ham__ = ham_pot(kk, nn, offset)

    eval_, evect__ = np.linalg.eigh(ham__)
    evect__ = evect__.transpose()

    adim_erg_ = eval_ / dtilde(nn)

    return eval_, evect__, adim_erg_


def xray_mel_f0dagger(kk_i, kk_f, nn):
    """returns xray absorption rates for given initial/final scattering potentials kk_i/kk_f """
    particle = 2  # particle level, measured from Fermi level

    nfermi = int((nn + 1) / 2)
    npart = nfermi + 1

    evali_, evecti__, adim_ergi_ = diag_ham_pot(kk_i, nn, 0)
    evalf_, evectf__, adim_ergf_ = diag_ham_pot(kk_f, nn, 0)

    proj__: ndarray = np.zeros((npart, npart))
    for bra in range(npart):  # final state: all states below fermi level + one at fermi+particle
        for ket in range(npart):  # initial state: all states below fermi level
            if bra < nfermi:
                if ket < nfermi:
                    proj = np.dot(evectf__[bra], evecti__[ket])
                else:  # ket == nfermi => f_0^dagger
                    proj = evectf__[bra][0]
            else:  # bra >= nfermi => nfermi + particle
                if ket < nfermi:
                    proj = np.dot(evectf__[nfermi + particle], evecti__[ket])
                else:
                    proj = evectf__[nfermi + particle][0]
            proj__[bra, ket] = proj
    return np.linalg.det(proj__), evalf_[nfermi + particle]


def xray_mel_f0(kk_i, kk_f, nn):
    """returns xray emission rates for given initial/final scattering potentials kk_i/kk_f """
    hole = 2  # particle level, measured from Fermi level

    nfermi = int((nn + 1) / 2)
    npart = nfermi + 1

    evali_, evecti__, adim_ergi_ = diag_ham_pot(kk_i, nn, 0)
    evalf_, evectf__, adim_ergf_ = diag_ham_pot(kk_f, nn, 0)

    # instead of <F|f_0|I>, compute <I|f_0^\dagger|F>
    proj__: ndarray = np.zeros((npart, npart))
    for bra in range(npart):  # initial state: all states below fermi level + one
        for ket in range(npart):
            # final state: all states below fermi level except one at nhole
            if ket != nfermi - hole:
                proj = np.dot(evecti__[bra], evectf__[ket])
            else:
                proj = evecti__[bra][0]
            proj__[bra, ket] = proj
    return np.linalg.det(proj__), evalf_[nfermi - hole]


def xray_exp(delpi, delpf, want_f0dagger=True):
    """ returns xray-absorption exponent for phase shift, over pi"""
    kk_i = -tan(delpi * pi) / pi
    kk_f = -tan(delpf * pi) / pi

    nn_ = np.arange(5, 31, 2)

    erg_ = np.zeros(len(nn_))
    rate_ = np.zeros_like(erg_)

    ret_ = np.zeros_like(erg_)

    for jj, nn in enumerate(nn_):
        if want_f0dagger:
            mel, erg = xray_mel_f0dagger(kk_i, kk_f, nn)
        else:
            mel, erg = xray_mel_f0(kk_i, kk_f, nn)
        erg_[jj] = erg
        rate_[jj] = np.power(mel, 2.0) / erg
        if jj > 0:
            ret_[jj] = log(rate_[jj] / rate_[jj - 1]) / log(erg_[jj] / erg_[jj - 1])
    return ret_


def bic_mel_adagger(kk_i, kk_f, nn):
    """returns spectral-density matrix element for given pair of scattering potentials kk_i/kk_f
    assumes |I> = g_1^\dagger|\Omega_I> and |F> = g_part^\dagger|\Omega_F> """
    particle: int = 2  # particle level, measured from Fermi level
    lamb: float = lmbda()

    nfermi = int((nn + 1) / 2)
    npart = nfermi + 1

    evali_, evecti__, adim_ergi_ = diag_ham_pot(kk_i, nn, 0)
    evalf_, evectf__, adim_ergf_ = diag_ham_pot(kk_f, nn, 0)

    proj__: ndarray = np.zeros((npart, npart))
    for bra in range(npart):  # final state: all states below fermi level + one at fermi+particle
        for ket in range(npart):  # initial state: all states below fermi level + one at npart
            if bra < nfermi:
                proj = np.dot(evectf__[bra], evecti__[ket])
            else:  # bra >= nfermi => nfermi + particle
                proj = np.dot(evectf__[nfermi + particle], evecti__[ket])
            proj__[bra, ket] = proj
    return np.linalg.det(proj__), evalf_[nfermi + particle]


def bic_mel_a(kk_i, kk_f, nn):
    """returns spectral-density matrix element for given pair of scattering potentials kk_i/kk_f
     assumes |I> = g_fermi|\Omega_I> and |F> = g_hole|\Omega_F>"""
    hole = 2  # particle level, measured from Fermi level
    lamb: float = lmbda()

    nfermi = int((nn + 1) / 2)
    npart = nfermi - 1

    evali_, evecti__, adim_ergi_ = diag_ham_pot(kk_i, nn, 0)
    evalf_, evectf__, adim_ergf_ = diag_ham_pot(kk_f, nn, 0)

    # instead of <F|f_0|I>, compute <I|f_0^\dagger|F>
    proj__: ndarray = np.zeros((npart, npart))
    proj = 0  # keeps pycharm happy
    for bra in range(npart):  # initial state: all states below (nfermi-1)
        for ket in range(nfermi):
            # final state: all states below fermi level except one at nhole
            if ket != nfermi - hole:
                proj = np.dot(evecti__[bra], evectf__[ket])
            if ket < nfermi - hole:
                proj__[bra, ket] = proj
            elif ket > nfermi - hole:
                proj__[bra, ket - 1] = proj
    return np.linalg.det(proj__), evalf_[nfermi - hole]


def bic_exp(delpi, delpf, want_a_dagger=True):
    """ returns xray-absorption exponent for phase shift, over pi"""
    kk_i = -tan(delpi * pi) / pi
    kk_f = -tan(delpf * pi) / pi

    nn_ = np.arange(5, 31, 2)

    erg_ = np.zeros(len(nn_))
    rate_ = np.zeros_like(erg_)

    ret_ = np.zeros_like(erg_)

    for jj, nn in enumerate(nn_):
        if want_a_dagger:
            mel, erg = bic_mel_adagger(kk_i, kk_f, nn)
        else:
            mel, erg = bic_mel_a(kk_i, kk_f, nn)
        erg_[jj] = erg
        rate_[jj] = np.power(mel, 2.0) / erg
        if jj > 0:
            ret_[jj] = log(rate_[jj] / rate_[jj - 1]) / log(erg_[jj] / erg_[jj - 1])
    return ret_


def hamRes(offset):
    ''' constructs eNRG resonant level Hamiltonian '''
    dim = n_band() + 1  # n_band() == Niter + 1 (n=0,...,N-1)
    # dim = dimension (conduction band (n_band() + impurity)
    ret__: ndarray = np.zeros([dim, dim])

    nn = n_band() - 1  # nn == Niter
    if want_wilson():
        v = sqrt(gam() * alamb() / np.pi)
    else:
        v = sqrt(gam() * 2.0 * sqrt(2.0))

    for col in range(nn):
        t_n = calc_tn(offset, col)
        ret__[col, col + 1] = ret__[col + 1, col] = t_n
    ret__[0, dim - 1] = v
    ret__[dim - 1, 0] = v
    ret__[dim - 1, dim - 1] = ed()  # impurity
    return ret__


def diag_hamRes(offset):
    '''returns eigenvalues and eigenvectors of NRG conduction-band hamiltonian'''
    ham__ = hamRes(offset)

    eval_, evect__ = np.linalg.eigh(ham__)
    evect__ = evect__.transpose()

    return eval_, evect__, eval_ * np.power(lmbda(), n_band() - 1.5)


def conduct(kbT_, offset):
    ''' conductance of resonant level Hamiltonian for given offset '''
    scd_ = np.zeros_like(kbT_)
    set_ = np.zeros_like(kbT_)

    va, ve, varn = diag_hamRes(offset)
    dim = n_band() + 1
    for p in range(dim):
        alp02 = np.power(ve[p][0], 2.0)
        alpd2 = np.power(ve[p][dim - 1], 2.0)
        for it, kbt in enumerate(kbT_):
            bterg = va[p] / kbt
            if abs(bterg) > 20.0:
                continue
            try:
                btfermi = np.power(2.0 * cosh(0.5 * bterg), -2.0) / kbt
            except:
                print(f"{bterg}\t{kbt}")
                return scd_, set_

            scd_[it] += alp02 * btfermi
            set_[it] += alpd2 * btfermi

    return scd_, set_ * (pi * gam())


def g4(kbT_, offset):
    ''' conductances for resonant level Hamiltonian '''

    gscd0_, gset0_ = conduct(kbT_, offset)
    gscd1_, gset1_ = conduct(kbT_, offset + 1)

    return gscd0_, gscd1_, gset0_, gset1_


def g_ave(kbT_, offset):
    ''' averaged conductances for RL Hamiltonian '''
    tempor_ = g4(kbT_, offset)

    ret_ = 0.5 * (tempor_[0] + tempor_[1])

    return ret_


# linear basis
def ham_RL_linear(gam=1.e-2):
    ''' returns hamiltonian matrix for RL model on linear basis'''
    nn = n_band() - 1
    ret__ = np.zeros([nn + 2, nn + 2])

    delt = 2.0 / n_band()  # splitting between band levels
    param = Param(gam)
    v = sqrt(param.gam() / pi)
    n_fermi = 1 + int(nn / 2)
    # band with nn/2 positive
    #     and nn/2 negative levels, plus zero level

    for levl in np.arange(n_fermi - 1, 0, -1):
        pos = n_fermi - levl
        neg = n_fermi + levl
        ret__[pos][pos] = levl * delt
        ret__[neg][neg] = -levl * delt

    for levl in np.arange(1, nn + 2):
        rem = nn + 2 - levl
        if levl < rem:
            ret__[levl][0] = v * tanh(nn * levl / n_fermi)
            ret__[0][levl] = ret__[0][levl]
        elif levl > rem:
            ret__[levl][0] = v * tanh(rem * levl / n_fermi)
            ret__[0][levl] = ret__[0][levl]
        else:  # levl == nfermi + 1
            ret__[levl][0] = v
            ret__[0][levl] = ret__[0][levl]

    return ret__


def diag_ham_RL_linear(gam=1.e-2):
    '''returns eigenvalues/vectors of conduction-band hamiltonian with linear dispersion
    :rtype: vec_, vec__
    '''
    nn = n_band() - 1
    ham__ = ham_RL_linear(gam)

    eval_, evect__ = np.linalg.eigh(ham__)
    evect__ = evect__.transpose()

    return eval_, evect__


def conduct_RL_linear(kbT_):
    ''' conductance of resonant level Hamiltonian for linear dispersion '''
    scd_ = np.zeros_like(kbT_)
    set_ = np.zeros_like(kbT_)

    nn = n_band() - 1

    va, ve = diag_ham_RL_linear(nn)

    for p in range(nn + 2):
        alp02 = 0.
        bt02 = np.power(ve[p][0], 2.0)

        for n in range(1, nn + 2):
            alp02 += ve[p][n] ** 2
        for it, kbt in enumerate(kbT_):
            if abs(va[p]) / kbt > 10:
                continue
            bolt = exp(va[p] / kbt)
            fermi = bolt / (1.0 + bolt) ** 2 / kbt

            scd_[it] += alp02 * fermi

            set_[it] += bt02 * fermi

    return scd_, set_ * pi * gam()


# auxiliary functions

def phase_shifts(gm):
    """returns phase shifts of RL eigenvalues"""
    nn = n_band() - 1
    va, ve = diag_ham_RL_linear(gm)
    va0, ve0 = diag_ham_RL_linear(0.)

    ret_ = np.zeros_like(va)

    for n in range(nn + 2):
        if abs(va0[n]) < 1.e-12:
            ret_[n] = 0.0
        elif va0[n] < 0.0:
            ret_[n] = -pi * log(va[n] / va0[n]) / log(lmbda())
        else:
            ret_[n] = pi * log(va[n] / va0[n]) / log(lmbda())
    return ret_


def temps(lamb=2.0) -> object:
    ''' returns logarithmic sequence of temperatures '''
    niter = 15
    n_temps = 5  # number of temperatures in \lambda cycle
    t_fact = np.power(lamb, 1. / n_temps)
    ret_ = []

    temperature = 2.0
    for itr in range(niter):
        for nt in range(n_temps):
            ret_.append(temperature)
            temperature /= t_fact

    return np.array(ret_)


def print_gs(filename, t_, gb_, genrg_):
    ''' prints t-bind and e-nrg conductances
    :type genrg_: float[]
    :type gb_: float[]
    '''

    fud = open(filename, 'w')
    fud.write(f'# \Gamma = {gam() :7.3g}, U = 0, N = 10000\n')
    fud.write(f'# Temp\t G[t-bind]\t G[eNRG]\n')

    for it, t in enumerate(t_):
        g1 = gb_[it]
        g2 = genrg_[it]
        fud.write(f"{t_[it]: <11.8f}\t{g1: <11.7f}\t{g2: <11.7f}\n")
    fud.close()


def beta(ve, nn):
    """ return ve[0]**2"""
    ret_ = np.zeros(nn)
    for n in range(nn):
        ret_[n] = np.power(ve[n][0], 2.0)
    return ret_


def alpha(ve, nn):
    """ return \sum ve[0]**2"""
    # See PyCharm help at https://www.jetbrains.com/help/pycharm/
    ret_ = np.zeros(nn)
    for n in range(nn):
        alp2 = 0.0
        for p in range(1, nn):
            alp2 += np.power(ve[n][p], 2.0)
        ret_[n] = alp2
    return ret_


def phase_shift(ell, nn):
    """
    :param ell: level
    :param nn: dimension of conduction band
    :return: phase shift of level
    """
    return 0.0  # to keep compiler happy


def nk():
    """returns no of k points"""
    return 10000000


def del_over_pi(erg):
    """" erg = -2tcos(q), for some q. Vector kq is nearest k. Returns q-kq """
    q = acos(-erg / 2.0)

    zq = q * nk() / pi
    nq = np.floor(zq)
    delta_over_pi = zq - nq
    if delta_over_pi > 0.5:
        delta_over_pi -= 1.
        nq += 1
    return delta_over_pi, nq


def green(erg):
    """ compute sum over green's function's moments using nn k points"""
    nn = nk()

    nn1 = np.floor(2 * nk() / 5.)
    nn2 = np.floor(3 * nn1 / 2.)

    sum0 = 0.
    sum1 = 0.
    sum2 = 0.
    DK = 1.e-10  # avoids singularity
    n1_ = np.arange(nn1)
    n2_ = np.arange(nn2, nn)
    ergk1_ = -1. * np.cos(n1_ * pi / nn)
    ergk2_ = -1. * np.cos(n2_ * pi / nn)
    gk1_ = np.power(erg - ergk1_, -1.0)
    gk2_ = np.power(erg - ergk2_, -1.0)
    sum0 = np.sum(gk1_) + np.sum(gk2_)
    egk1_ = ergk1_ * gk1_
    egk2_ = ergk2_ * gk2_
    sum1 = np.sum(egk1_) + np.sum(egk2_)
    e2gk1_ = egk1_ * ergk1_
    e2gk2_ = egk2_ * ergk2_
    sum2 = np.sum(e2gk1_) + np.sum(egk2_)
    sum0 += 5. * erg / pi
    return sum0, sum1, sum2


# t_ = temps()
# conduct(t_, 0)
def correct_delta():
    """ reads data from file and returns correct phase shifts"""
    ud = open('ph_shift.txt', 'w')
    ps__ = np.loadtxt('phase_shift.txt')
    w_ = ps__.transpose()[0]
    d_ = ps__.transpose()[1]
    k_ = 2. * np.tan(pi * d_) / pi
    new_d_ = np.arctan(pi * k_) / pi
    simple_ = np.arctan(pi * w_) / pi

    for lin, w in enumerate(w_):
        ud.write(f'{w:7.4f}\t{new_d_[lin]:7.4f}\t{simple_[lin]:7.4f}\n')
    ud.close()

    return w_, new_d_


def directry():
    ret = '/Users/luizno/Dropbox/orienta/dr/luizHenrique/writeupLH/2021/'
    ret += 'l_v0/material/NRGdata/'
    return ret


def asymptotic(nn, q, ds, k, filename):
    """ returns ratio |<m|d^\daager\I>2/E_m, read from given file"""
    SMALL = 1.e-12
    gamk = -atan(k * pi) / pi

    dirctry = '/Users/luizno/Dropbox/orienta/dr/luizHenrique/writeupLH/2021/'
    subdirctry = 'l_v0/material/NRGdata/'
    fname = dirctry + subdirctry + filename + '.txt'
    data__ = np.genfromtxt(fname, dtype=None)
    alph = 0.0
    if q < 0:
        alph = 2.0 * np.power(gamk - 0.47, 2.0) - 1
    else:
        alph = 2.0 * np.power(1 + gamk - 0.47, 2.0) - 1
    print(f"alpha = {alph:7.5g}")
    rerg_ = np.zeros(len(data__))
    ros_ = np.zeros_like(rerg_)
    count = 0
    for ll, lin_ in enumerate(data__):
        if lin_[0] == nn and lin_[4] == q and lin_[5] == ds:
            if lin_[9] < SMALL:  # transition to threshold; special handling
                print(f'threshold: |<F|a^+|I>|2 = {lin_[10]:7.4g}')
            else:
                erg = dtilde(nn) * lin_[9]
                ratio = lin_[10] / erg / np.power(erg, alph)
                #                print(f'{lin_[7]:03d}\t{lin_[9]:6.3f}\t{ratio:7.4g}')
                rerg_[count] = lin_[9]
                ros_[count] = ratio
                count += 1
    return rerg_, ros_


def want_final_state(q):
    """ returns ordinal number of desired final state """
    if q < 0:
        return 0
    else:
        return 1


def compare(q, ds, k, filename):
    """ prints scaled asymptotic rates at nn = 100 and 200 """
    e100_, os100_ = asymptotic(100, q, ds, k, filename)
    e150_, os150_ = asymptotic(150, q, ds, k, filename)
    e200_, os200_ = asymptotic(200, q, ds, k, filename)

    state_no = want_final_state(q)
    print(f"100: {os100_[state_no]:9.5g}")
    print(f"150: {os150_[state_no]:9.5g}")
    print(f"200: {os200_[state_no]:9.5g}")
    print(f"ratio = {os200_[1] / os150_[1]:}")


def bitfield(n):
    return [int(digit) for digit in bin(n)[2:]]


def bin_ergs(eva_, q):
    ''' computes energies of all many-body eigenstates with charge q.
    input:
    eva_: vector containing single-particle eigenvalues closest to Fermi level
    q: number of electrons occupying levels of eva_ in many-body states
    output:
    bin_: binary representation of many-body eigenstates
    erg_: energy of many-body eigenstates
    '''
    nn = len(eva_)
    egrd = 0.0
    for eva in eva_:
        if eva < 0:
            egrd += eva
    egrd *= 2.0  # accounts for spin

    bin_up_ = np.zeros(nn, dtype=int)
    bin_do_ = np.zeros(nn, dtype=int)
    bin_ = []
    erg_ = []
    nbin = 1 << (2 * nn)
    for nb in range(nbin):
        if bin(nb).count("1") != q:
            continue
        b_ = bitfield(nb)
        lb = len(b_)
        erg = 0.0
        for j in range(lb):
            bj = b_[j]
            if j < nn:
                bin_do_[j] = bj
                erg += eva_[j] * bj
            else:
                bin_up_[j - nn] = bj
                erg += eva_[j - nn] * bj
        erg_.append(erg)
        bin_sum_ = bin_up_ + bin_do_
        bin_.append(bin_up_ + bin_do_)
    erg_ = np.array(erg_) - egrd
    bin_ = np.array(bin_)

    sort_ = np.sort(erg_)
    return sort_, bin_


def alpha_pm(kk, nn):
    """ returns {f_0^\dagger, g_k) coefficients
    input:
     kk == K
     nn == N (should be odd)
     output:
     alpha_p
     alpha_k
     """
    eva_, evect__, erg_ = diag_ham_pot(kk, nn, 0)
    nfermi = int((nn + 1) / 2)  # no levels below E_F
    temp__ = evect__.transpose()  # so that temp__[0] == f_0
    f0m_ = temp__[0][:nfermi]
    f0p_ = (temp__[0][::-1])[:nfermi]
    #    xp_ = np.arange(nfermi)
    #    renrm_ = np.power(lmbda(), xp_)
    #    alpha_p_ = renrm_ * f0p_
    #    alpha_m_ = renrm_ * f0m_

    return f0p_, f0m_


def rkky(kk, nn):
    """ retuns RKKY interaction (for R=0) with scattering potential K
    input:
    kk = scattering potential
    nn = N
    output:
    RKKY interaction
    """
    nfermi = nn + 1
    nfermi = (nn + 1) >> 1
    eva_, evect__, erg_ = diag_ham_pot(kk, nn, 0)
    ergm_ = eva_[:nfermi]
    ergp_ = (eva_[::-1])[:nfermi]
    ap, am = alpha_pm(kk, nn)

    ret = 0.0

    for n_p in np.arange(nfermi):
        ap2 = np.power(ap[n_p], 2.0)
        ep = ergp_[n_p]
        for n_m in np.arange(nfermi):
            am2 = np.power(ap[n_m], 2.0)
            em = ergm_[n_m]
            ret += ap2 * am2 / (ep - em)

    return 4.0 * ret


def rkky_int(eps_d, delta, gmma, nn):
    """ computes RKKY interaction from anderson Hamiltonian parameters """
    uu = delta - eps_d
    inv1: float = 1.0 / abs(eps_d)
    inv2: float = 1.0 / delta
    rj = 2.0 * gmma / pi * (inv1 + inv2)
    rk = 0.5 * gmma / pi * (inv1 - inv2)
    print(f'U = {uu:5.2g} => rJ = {rj:5.2g}\t rK = {rk:5.2g}')
    irkky = np.power(rj, 2.0) * rkky(rk, nn)
    return delta * irkky


def hald(delta, eps_d, uu, gamma):
    """returns Haldane's correction for delta==u+\epsilon_d,
    which equals zero at correct delta
    input:
    delta = trial eps_d +U
    eps_d = epsilon_d
    uu = U
    gamma = Gamma
    output:
    delta^* """
    ret = eps_d + uu - gamma / pi * np.log(uu / delta)
    return ret - delta


def iter_haldane(eps_d, uu, gamma, niter):
    """ iterates ed_haldane() for specified number of iterations niter """
    # start out with two trials: one an upper bound, the other a lower bound
    delta_hi: float = eps_d + uu
    delta_lo: float = gamma

    d_try: float = (delta_lo + delta_hi) / 2.0
    for n in range(niter):
        if abs(hald(delta_hi, eps_d, uu, gamma)) < 1.e-5 * abs(delta_hi):
            print(f"Hi: {delta_hi:}")
            return delta_hi

        if abs(hald(delta_lo, eps_d, uu, gamma)) < 1.e-5 * abs(delta_lo):
            print(f"Lo: {delta_lo:}")
            return delta_lo
        if hald(delta_hi, eps_d, uu, gamma) * hald(d_try, eps_d, uu, gamma) < 0:
            # [d_try, delta_up] brackets solution
            delta_lo = d_try
        else:
            # [delta_do, d_try] brackets solution
            delta_hi = d_try
        d_try = (delta_lo + delta_hi) / 2.0
        if abs(delta_hi - delta_lo) < delta_lo * 1.e-3:
            print(f"{n:}\t[{delta_lo:7.3g},{delta_hi:7.3g}]")
            return d_try
    return d_try

def delta_and_rkky(eps_d, uu, gmma):
    """ computes Haldane's correction and resulting RKKY interaction
    input:
    Anderson's parameters
    eps_d == epsilon_d,
    uu == U, and
    gmma == Gamma
    output:
    gap delta == (epsilon_d + U)^*
    rkky interaction
    """
    max_iter = nn = 99
    dlt = iter_haldane(eps_d, uu, gmma, max_iter)
    irkky = rkky_int(eps_d, dlt, gmma, nn)

    return dlt, irkky

def save_delta_rkky(eps_d, gmma):
    """" computes Delta == epsilon_d + U and I_RKKY
    for U running from -eps_d to -2*eps_d
    saves results in file delta_rkky_.gmma"""
    fud: TextIO = open(directry() + f"delta_rkky.{gmma:4.1g}.txt", "w")
    fud.write(f"# epsilon_d = {eps_d:7.4g}\tGamma={gmma:7.4g}\n")
    fud.write(f"# U\t\t Delta \t\t I_RkkY\n")
    for uu in np.linspace(-1.001*eps_d, -2.001*eps_d, 500):
        dlt, irkky = delta_and_rkky(eps_d, uu, gmma)
        fud.write(f"{uu:< 7.4g}\t{dlt:< 7.4g}\t{irkky:< 7.4g}\n")
    return
# import pdb
# pdb.set_trace()
# iter_haldane(-2e-5, 2.5e-5, 5e-7, 99)

def gamma_w(A=3.5, a=0.6, zim=1.3, zc=0.7, dd=0.5):
    """ compute \Gamma_W for sticking coefficient computation"""
#    dd *= (1-1./1.8) / log(1.8)
    vzc = A * exp(-zc*a) / sqrt(2)
    wzc = 27.2 / 2./ 4. / (zc+zim)

    ret = pi * vzc**2 * dd / (dd**2 + pi**2 * wzc**2)

    return ret

def nneighbor(t: float, r: float, nn:int):
    """
    returns next-neighbor-coupling Hamiltonian
    h0 = \sum_{0<n<nn} (t a_n^\dagger a_{n+1} + r a_n^\dagger a_{n+2} + H.c.)
"""
    h__ = np.zeros([nn, nn])
    for n in range(nn):
        if n < nn - 1 :
            h__[n][n+1] = h__[n+1][n] = t
        if n < nn - 2 :
            h__[n][n+2] = h__[n+2][n] = r
    return h__


def gram_schmidt(u__, m, v_):
    """ given orthonormal vectors u__[0],..., u__[m-1], and linearly independent
    vector v_, orthogonalize v_ to u vectors """
    proj_ = np.zeros(m)
    for n in range(m):
        unorm = np.linalg.norm(u__[n])
        projm = np.dot(u__[n], v_) / np.power(unorm, 2)
        v_ = v_ - projm * u__[n]
        print(n, ":", np.dot(v_, u__[m-1])/np.linalg.norm(v_))

    v_ = v_ / np.linalg.norm(v_)
    return v_

def tridHam(tau_, eta_):
    """ returns tridiagonal Hamiltonian with diagonal elements eta_ and
    codiagonal elements tau_
     Input: vectors tau_ (dimension n-1) and eta_ (dimension n)
     """
    dim = len(eta_)
    h__ = np.zeros([dim, dim])
    for n in range(dim):
        h__[n,n] = eta_[n]
        if n < dim  -1:
            h__[n, n+1] = h__[n+1, n] = tau_[n]
    return h__



def lanczos(t: float, r: float, nn: int):
    """ carries out Lanczos algorithm 
     to convert next-neighbor Hamiltonian to tridiagonal form
     Input Hamiltonian reads 
     h0 = \sum_{0<n<nn} (t a_n^\dagger a_{n+1} + r a_n^\dagger a_{n+2} + H.c.)
     output matrix is
     htrid = \sum_{0<n<nn} (tau_n c^\dagger c_{n+1}+H.c.) + \eta_n c_n^\dagger c_n
     returns vectors tau_ and eta_
     """
    h__ = nneighbor(t, r, nn)
# operational matrices
    v__ = np.zeros([nn,nn])
    w__ = np.zeros([nn,nn])
# output matrices
    eta_ = np.zeros([nn])
    tau_ = np.zeros([nn-1])

# iteration zero
    v__[0][0] = 1.  # seed state is a_0
    w__[0] = np.matmul(h__, v__[0])
    eta_[0] = np.dot(w__[0], v__[0])
    w__[0] = w__[0] - eta_[0] * v__[0]
    for j in range(1, nn):
        tj = tau_[j-1] = np.linalg.norm(w__[j-1])
        if tj < 1.e-10:
            print(f"*** small tau_[{j-1:}]: {tj:7.3g}")
            return tau_, eta_
        v__[j] = (1. / tj) * w__[j-1]
        w__[j] = np.matmul(h__, v__[j])
        eta_[j] = np.dot(w__[j], v__[j])
        w__[j] = w__[j] - eta_[j] * v__[j] - tj * v__[j-1]
        v__[j] = gram_schmidt(v__, j, v__[j])

    return tau_, eta_


