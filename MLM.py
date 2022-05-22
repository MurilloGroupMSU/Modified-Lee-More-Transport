
import numpy as np
import scipy.integrate as integrate
from scipy.special import zeta, gamma, factorial, hyp1f1, hyperu


def zbar(Z, AM, rho, T):
    """
    Finite Temperature Thomas Fermi Charge State using
    R.M. More, "Pressure Ionization, Resonances, and the
    Continuity of Bound and Free States", Adv. in Atomic
    Mol. Phys., Vol. 21, p. 332 (Table IV).

    Z = atomic number
    AM = atomic mass
    rho = density (g/cc)
    T = temperature (eV)
    """

    alpha = 14.3139
    beta = 0.6624
    a1 = 0.003323
    a2 = 0.9718
    a3 = 9.26148e-5
    a4 = 3.10165
    b0 = -1.7630
    b1 = 1.43175
    b2 = 0.31546
    c1 = -0.366667
    c2 = 0.983333

    R = rho/(Z*AM)
    T0 = T/Z**(4./3.)
    Tf = T0/(1+T0)
    A = a1*T0**a2+a3*T0**a4
    B = -np.exp(b0+b1*Tf+b2*Tf**7)
    C = c1*Tf+c2
    Q1 = A*R**B
    Q = (R**C+Q1**C)**(1/C)
    x = alpha*Q**beta

    return Z*x/(1 + x + np.sqrt(1 + 2.*x))


def FD_int_0(eta):
    '''Fermi integral of order 0, for which
    the exact result is easily expressible.'''
    
    return np.log(1 + np.exp(eta))


def FD_int_m1(eta):
    '''Fermi integral of order -1, for which
    the exact result is easily expressible.'''
    
    return np.exp(eta)/(1 + np.exp(eta))


def FD_int(eta, order, EPS=5e-5):
    '''Numerically computes Fermi integrals.
    
    Algorithm 745: Computation of the Complete and 
    Incomplete Fermi-Dirac Integral
    Michele Goano
    ACM Trans. Math. Soft. 21, 221 (1995)
    
    This function requires several special functions, 
    including Riemann zeta, factorial, gamma, confluent 
    hypergeometrics (Kummer and Tricomi); be sure to 
    import them if you move this function to another code.
    
    Since this is to be used with the conductivity model,
    I use the convention with no gamma-function scaling of 
    the FD integral (boring F), although Goano includes 
    it (fancy F). (See Lee-More (26).) If you want the convention 
    with the 1/gamma(1 + order), simply comment out the 
    gamma factor at each return statement.
    '''
    
    if eta <= -2:
        n = 0
        error = 9999
        new_sum = 0.0
        while error > EPS:
            n += 1
            temp_sum = new_sum +  (-1)**(n - 1)*np.exp(n*eta)/n**(1+order)
            error = np.abs(temp_sum - new_sum)/temp_sum
            new_sum = temp_sum
        # print(f"FD took {n} iterations.")
        return new_sum*gamma(1 + order) # remove gamma() for other convention
    
    elif eta > -2 and eta <= 2:
        n = 0
        error = 9999
        new_sum = 0.0
        while error > EPS:

            if n == order:
                zeta_factor = np.log(2)
            else:
                zeta_factor = (1 - 2**(n - order))*zeta(order + 1 - n)

            temp_sum = new_sum +  zeta_factor*eta**n/factorial(n)
            if zeta_factor != 0:
                error = np.abs(temp_sum - new_sum)/np.abs(temp_sum)
                new_sum = temp_sum
            n += 1

        return new_sum*gamma(1 + order) # remove gamma() for other convention
    
    elif eta > 2:
        n = 0
        error = 9999
        new_sum = 1.0
        base = eta**(order + 1)/factorial(order + 1)
        while error > EPS:
            n += 1
            temp_sum = new_sum + (-1)**(n-1)*((order + 1)*hyperu(1, order+2, n*eta) - hyp1f1(1, order + 2, -n*eta))
            error = np.abs(temp_sum - new_sum)/np.abs(temp_sum)
            # print(temp_sum, error)
            new_sum = temp_sum

        # print(f"FD took {n} iterations.")
        return base*new_sum*gamma(1 + order) # remove gamma() for other convention

    
def Ichimaru_chem_pot(theta):
    '''Ichimaru's fit to the chemical potential: Ichimaru, Volume II, page 87, (3.147)
    input: theta = T/E_F
    ouput: unitless ideal gas chemical potential
    '''
    
    A = 0.25954
    B = 0.072
    b = 0.858

    return -1.5*np.log(theta) + np.log(4/(3*np.sqrt(np.pi))) + (A*theta**(-b-1) + B*theta**(-b/2-1/2))/(1+ A*theta**(-b))


def A_alpha_fit(eta):
    '''Fit from the appendix of LM, used to check the numerical results.
    From (A2) and Table VII, p. 1284, LM [Phys. Fluids 27 (1984)]. 
    
    See below for a direct evaluation that avoids using a fit.
    
    Used mainly for electrical conductivity.
    
    This might be faster in some cases and therefore possibly preferred
    if extremely large numbers of calculations are needed.'''
    
    a_1 = 3.39
    a_2 = 3.47e-1
    a_3 = 1.29e-1
    b_2 = 5.11e-1
    b_3 = 1.24e-1

    y = np.log(1 + np.exp(eta))

    return (a_1 + a_2*y + a_3*y**2)/(1 + b_2*y + b_3*y**2)


def A_beta_fit(eta):
    '''Fit from the appendix of LM, used to check the numerical results.
    From (A2) and Table VII, p. 1284, LM [Phys. Fluids 27 (1984)]. See also
    limits (28b) and (30b).
        
    See below for a direct evaluation that avoids using a fit.
    
    Used mainly for thermal conductivity.
        
    This might be faster in some cases and therefore possibly preferred
    if extremely large numbers of calculations are needed.'''
    
    a_1 = 13.5
    a_2 = 9.76e-1
    a_3 = 4.37e-1
    b_2 = 5.10e-1
    b_3 = 1.26e-1

    y = np.log(1 + np.exp(eta))

    return (a_1 + a_2*y + a_3*y**2)/(1 + b_2*y + b_3*y**2)


def A_alpha(eta):
    '''Compute A^\alpha directly from numerical Fermi integrals.
    This is called D in my FIP paper to avoid conflict with other uses of A.'''
    
    return (4./3)*FD_int(eta,2)/((1 + np.exp(-eta))*FD_int(eta,0.5)**2)


def A_beta(eta):
    '''Compute A^\beta directly from numerical Fermi integrals.
    Not included in the FIP paper, which was only focussed on electrical
    conductivites.'''
    
    # return (20./9)*FD_int(eta,4)*(1 - 16*FD_int(eta,3)**2/(15*FD_int(eta,4)*FD_int(eta,2)))/((1 + np.exp(-eta))*FD_int(eta,0.5)**2)
    return (20./9)*FD_int(eta,4)*(1 - 16*FD_int(eta,3)**2/(15*FD_int(eta,2)*FD_int(eta,4)))/((1 + np.exp(-eta))*FD_int(eta,0.5)**2)
    # return (20./9)*FD_int(eta,4)#*(1 - 16*FD_int(eta,3)**2/(15*FD_int(eta,2)*FD_int(eta,4)))/(FD_int(eta,0.5)**2*(1 + np.exp(-eta)))


def effective_temperature(T_e, eta):
    '''Return the "Thomas-Fermi temperature", which yields
    T_e and (2/3)E_F in the classical and degenerate limits, respectively.
    T_e in eV and eta is dimensionless.'''
    
    return 2.0*T_e*FD_int(eta,0.5)/FD_int(eta,-0.5)


def LM_Coulomb_logarithm(n_e, T_e, n_i, T_i, eta, z_bar):
    '''This is the CL from LM, to my best guess of what they did.'''

    # b_max
    # It is not perfectly clear what LM meant by "Fermi temperature", so I used this:
    lambda_e = 1/np.sqrt(4*np.pi*n_e*1.44e-7/effective_temperature(T_e, eta)) # cm
    lambda_i = 1/np.sqrt(4*np.pi*n_i*1.44e-7/T_i) # cm
    a_i = (3/(4*np.pi*n_i))**(1/3) # I use this (a_i) for their R_0, since they didn't specify it.
    b_max = np.max([a_i, 1.0/np.sqrt(1/lambda_e**2 + 1/lambda_i**2)]) # LM rule for minimum CL

    # b_min
    b_min = np.min([z_bar*1.44e-7/(3.0*T_e), 5e-8/np.sqrt(T_e)]) # units are cm
#     b_min = np.min([z_bar*1.44e-7/(3.0*effective_temperature(T_e, eta)), 5e-8/np.sqrt(effective_temperature(T_e, eta))]) # what if LM used T_eff?


    return np.max([0.5*np.log(1 + b_max**2/b_min**2), 2.0])


def MSM_Coulomb_logarithm(n_e, T_e, n_i, T_i, eta, z_bar):
    '''
    This is my modification to the LM CL. Potentially important
    changes are made, such as using the effective electron temperature
    consistently.
    '''

    # b_max
    lambda_e = 1/np.sqrt(4*np.pi*n_e*1.44e-7/effective_temperature(T_e, eta)) # cm
    lambda_i = 1/np.sqrt(4*np.pi*n_i*1.44e-7/T_i) # cm
    a_e = (3/(4*np.pi*n_e))**(1/3)
    a_i = (3/(4*np.pi*n_i))**(1/3)
    b_max = 1/np.sqrt(1/(a_e**2 + lambda_e*2) + 1/(a_i**2 + lambda_i**2))

    # b_min
    b_min = np.sqrt((z_bar*1.44e-7/(3.0*effective_temperature(T_e, eta)))**2 + (5e-8/np.sqrt(effective_temperature(T_e, eta)))**2)

    return np.max([0.5*np.log(1 + b_max**2/b_min**2), 2.0]) # CL has floor of 2


def sigma_MSM_plasma(electron_temperature, ion_temperature, ion_density, eta, z_bar):
    '''Lee-More electrical conductivity, using my choices.
    inputs: temperatures in eV, density in 1/cc, others dimensionless
    Requires functions A^\alpha(eta) and TF_zbar.
    Base units are 1/s, or 1/(9*10^9 Ohm-m), but currently
    returns 1/(Ohm-cm).
    
    This function only evaluates the plasma part of the model. See
    below for the full (solid, liquid, plasma) version. 
    '''

    sigma_0 = ion_density*z_bar*1.44e-7*(2.997925e10)**2/511e3*A_alpha(eta) # units are 1/s^2
    degen_factor =  (1 + np.exp(-eta))*FD_int(eta, 0.5)

    Coulomb_log = MSM_Coulomb_logarithm(z_bar*ion_density, electron_temperature, ion_density, ion_temperature, eta, z_bar)

    tau_temporary = 3*np.sqrt(511e3)*electron_temperature**(3/2)/(2*np.sqrt(2)*2.997925e10*np.pi*z_bar**2*ion_density*(1.44e-7)**2*Coulomb_log) # units of s
    tau = tau_temporary*degen_factor
    sigma = sigma_0*tau # units are 1/s

    unit_factor = 9e11 # convert units to 1/(Ohm-cm)

    return sigma/unit_factor


def melt_temperature(Z_nuc, ion_density, original=False):
    '''LM use a fit due to Cowan for the melting temperature. I 
    re-fit Cowan's functional form to match the known melting temperature
    of 14 elements, while retaining his scalings. The goal was to increase
    the accuracy for common elements at their usual experimental
    "starting" conditions. There are many obvious ways of improving
    this functionality even further.'''
    
    mass_proton = 1.6737e-24 # grams
    
    if not original:
        c1 = 0.5
        c2 = 10
        c3 = 0.2
    else:
        c1 = 0.32
        c2 = 9
        c3 = 0.6       

    xi = c2*Z_nuc**0.3*ion_density*mass_proton # eqn (34) in LM
    b = c3*Z_nuc**(1/9)
    
    return c1*(xi/(1 + xi))**4*xi**(2*b - 2/3)
    
    
def sigma_MSM(electron_temperature, ion_temperature, ion_density, eta, z_bar, Z_nuc):
    '''Lee-More electrical conductivity, using my choices.
    inputs: temperatures in eV, density in 1/cc, others dimensionless
    Requires functions A^\alpha(eta) and TF_zbar.
    Base units are 1/s, or 1/(9*10^9 Ohm-m), but currently
    returns 1/(Ohm-cm).'''

    # conductivity divided by collision time (σ/τ) - see (23a)
    sigma_0 = ion_density*z_bar*1.44e-7*(2.997925e10)**2/511e3*A_alpha(eta) # units are 1/s^2
    
    # mean velocity, with degeneracy correction
    v_mean = np.sqrt(3*effective_temperature(electron_temperature, eta)/511e3)*2.997925e10 # cm/s
        
    # check for material phase

    melting_temperature = melt_temperature(Z_nuc, ion_density)
         
    R_0 = 2*(3/(4*np.pi*ion_density))**(1/3) # interatomic distance
    Gamma = 50*R_0*melting_temperature/ion_temperature # electron MFP, (32)
    
    unit_factor = 9e11 # convert units to 1/(Ohm-cm)
        
    if ion_temperature < melting_temperature:  # solid: Bloch-Gruneisen

        tau_solid = Gamma/v_mean # (32)
        
        return sigma_0*tau_solid/unit_factor
        
    else: # liquid, plasma and conductivity minimum
        
        # liquid
        phase_change_factor = 2.5 # change in resistivity upon melting (improve!)
        tau_liquid = Gamma/phase_change_factor/v_mean
        # minimum
        tau_minimum = R_0/v_mean # minimum MFP
        
        # plasma
        degen_factor =  (1 + np.exp(-eta))*FD_int(eta, 0.5)

        Coulomb_log = MSM_Coulomb_logarithm(z_bar*ion_density, electron_temperature, ion_density, ion_temperature, eta, z_bar)

        tau_temporary = 3*np.sqrt(511e3)*electron_temperature**(3/2)/(2*np.sqrt(2)*2.997925e10*np.pi*z_bar**2*ion_density*(1.44e-7)**2*Coulomb_log) # units of s
        tau_plasma = tau_temporary*degen_factor

        # choose which τ to use

        tau = np.max([tau_liquid, tau_minimum, tau_plasma])
        sigma = sigma_0*tau # units are 1/s

        return sigma/unit_factor


def sigma_LM(electron_temperature, ion_temperature, ion_density, eta, z_bar):
    '''Lee-More electrical conductivity, to my best guess of what they did.
    inputs: temperatures in eV, density in 1/cc, others dimensionless
    Requires functions A^\alpha(eta) and TF_zbar.
    Base units are 1/s, or 1/(9*10^9 Ohm-m), but currently
    returns 1/(Ohm-cm).'''

    sigma_0 = ion_density*z_bar*1.44e-7*(2.997925e10)**2/511e3*A_alpha(eta) # units are 1/s^2
    degen_factor =  (1 + np.exp(-eta))*FD_int(eta, 0.5)

    Coulomb_log = LM_Coulomb_logarithm(z_bar*ion_density, electron_temperature, ion_density, ion_temperature, eta, z_bar)

    tau_temporary = 3*np.sqrt(511e3)*electron_temperature**(3/2)/(2*np.sqrt(2)*2.997925e10*np.pi*z_bar**2*ion_density*(1.44e-7)**2*Coulomb_log) # units of s
    tau = tau_temporary*degen_factor
    sigma = sigma_0*tau # units are 1/s

    unit_factor = 9e11 # convert units to 1/(Ohm-cm)

    return sigma/unit_factor


def sigma(Z, A, rho, electron_temperature, ion_temperature = -9999):
    '''Call this function to get a conductivity prediction. This is a helper
    function that uses the functions above so that an 'external' user does not
    deal with <Z>, chemical potential, and so on.
    Z - no units
    rho - g/cc
    A - no units
    T - eV
    '''

    m_proton = 1.672622e-24
    if ion_temperature == -9999: 
        ion_temperature = electron_temperature
    ion_density = rho/(A*m_proton)

    z_bar = zbar(Z, A, rho, electron_temperature)
    n_e = z_bar*ion_density
    E_F = (0.197326e-4)**2*(3*np.pi**2*n_e)**(2/3)/(2*511e3)
    theta = electron_temperature/E_F
    eta = Ichimaru_chem_pot(theta)

    return sigma_MSM(electron_temperature, ion_temperature, ion_density, eta, z_bar, Z)
