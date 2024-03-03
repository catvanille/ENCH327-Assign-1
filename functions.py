# module containing functions for CO2 calculations
import numpy as np
import pandas as pd
from scipy.integrate import quad

def calc_PRZ(T,P):
    """
    Calculates the compressibility factor (Z) of CO2 using the Peng-Robinson equation of state (PR-EoS).
    Args:
        T (float): Temperature in Kelvin.
        P (float): Pressure in Pa.
    Returns:
        float: Compressibility factor.
    """

    R = 8.314  # J/mol-K
    Tc = 304.12  # K
    Pc = 73.74e5  # Pa
    w = 0.225

    # Reduced properties
    Pr = P / Pc
    Tr = T / Tc

    # PR-EoS parameters
    a0 = (0.45724 * (R * Tc) ** 2) / Pc
    b = 0.0778 * (R * Tc) / Pc
    kappa = 0.37464 + 1.54226 * w - 0.26992 * w ** 2
    alpha = (1 + kappa * (1 - np.sqrt(Tr))) ** 2
    a = a0 * kappa

    # Coefficients for cubic EoS
    A = 0.45724 * alpha * Pr / Tr ** 2
    B = 0.0778 * Pr / Tr
    a1 = 1
    a2 = B - 1
    a3 = A - 3 * B ** 2 - 2 * B
    a4 = -A * B + B ** 2 + B ** 3

    # Solve cubic equation for Z (positive real root)
    aa = np.array([a1, a2, a3, a4])
    zz = np.roots(aa)
    Z = zz[np.isreal(zz) == True][0] # Select the positive real root
    PRZ = np.real(Z)

    return PRZ

def res_enth(T, P, Z):
    """
    Calculates the residual enthalpy of CO2 using the Peng-Robinson equation of state (PR-EoS).
    Args:
        T (float): Temperature in Kelvin.
        P (float): Pressure in Pa.
    Returns:
        float: Residual enthalpy in kJ/mol.
    """

    R = 8.314  # J/mol-K
    Tc = 304.12  # K
    Pc = 73.74e5  # Pa
    w = 0.225

    # Reduced properties
    Pr = P / Pc
    Tr = T / Tc

    # PR-EoS parameters
    a0 = (0.45724 * (R * Tc) ** 2) / Pc
    b = 0.0778 * (R * Tc) / Pc
    kappa = 0.37464 + 1.54226 * w - 0.26992 * w ** 2
    alpha = (1 + kappa * (1 - np.sqrt(Tr))) ** 2
    a = a0 * kappa

    # Coefficients for cubic EoS
    A = 0.45724 * alpha * Pr / Tr ** 2
    B = 0.0778 * Pr / Tr
    a1 = 1
    a2 = B - 1
    a3 = A - 3 * B ** 2 - 2 * B
    a4 = -A * B + B ** 2 + B ** 3

    # Calculate residual enthalpy
    Hres = R * Tc * (Tr * (Z - 1) - 2.078 * (1 + kappa) * np.sqrt(alpha) * np.log((Z + 2.414 * B) / (Z - 0.414 * B)))
    Hres = Hres / 1000  # Convert to kJ/mol
    return Hres # kJ/mol

def calc_Z(T, P, v):
    """
    Calculates the compressibility factor (Z) based on Z= Pv/RT. For NIST data.
    Args:
        T (float): Temperature in Kelvin.
        P (float): Pressure in Pa."""
    
    R = 8.314  # J/mol-K
    Z = P*v/(R*T)

    return Z

def Cp(T):
    R = 8.314  # J/mol-K
    A = 5.457 
    B = 1.045e-3
    D = -1.157e5
    return R*( A + B*T + D*T**-2)

def calc_idealH(T1, T2):
    """
    Calculates the ideal gas enthalpy of CO2.
    Args:
        T1 (float): Initial temperature in Kelvin.
        T2 (float): Final temperature in Kelvin.
    Returns:
        float: Ideal gas enthalpy in J/mol.
    """
    H = quad(Cp, T1, T2) # J/mol
    
    return H[0]/1000 # kJ/mol

def percent_diff(NIST, PR):
    """
    Calculates the percent difference between NIST and PR-EoS vals.
    Args:
        NIST (float): NIST value.
        PR (float): PR-EoS value.
    Returns:
        float: Percent difference (absolute).
    """
    percent_diff = (abs(PR - NIST) / NIST) * 100
    return percent_diff

def calc_mol_flow(mass_flow = 400, molar_mass = 44.0095):
    """
    Calculates the molar flow rate.
    Args:
        mass_flow (float): Mass flow rate in tons/days.
        molar_mass (float): Molar mass in g/mol.
    Returns:
        float: Molar flow rate in mol/s.
    """
    mol_flow = mass_flow*1000*1000/molar_mass/24/3600
    return mol_flow # mol/s

def Hreal_out(Hin, Hs, isen_eff=0.85):
    """
    Calculates the enthalpy of the outlet stream of the turbine.
    Args:
        Hin (float): Enthalpy of the inlet stream in kJ/mol.
        Hs (float): Enthalpy of the isentropic process in kJ/mol.
        isen_eff (float): Isentropic efficiency.
    Returns:
        float: Enthalpy of the outlet stream in kJ/mol.
    """
    Hreal_out = (Hs- Hin + isen_eff*(Hin))/isen_eff
    return Hreal_out # kJ/mol

def calc_work(Hin, Hout, mol_flow):
    """
    Calculates the work on the compressor. 
    Args:
        Hin (float): Enthalpy of the inlet stream in kJ/mol.
        Hout (float): Enthalpy of the outlet stream in kJ/mol.
        mol_flow (float): Molar flow rate in mol/s.
    Returns:
        float: Work on the compressor in MW
    """
    work = mol_flow*(Hout - Hin)/1000
    return work # MW

def calc_Q(Tin, Tout, P, mol_flow):
    """
    Calculates the heat added to the system using the departure function and Cp.
    Assume constant pressure.
    Args:
        Tin (float): Inlet temperature in Kelvin.
        Tout (float): Outlet temperature in Kelvin.
        P (float): Pressure in Pa.
        mol_flow (float): Molar flow rate in mol/s.
    Returns:
        float: Heat added to the system in MW
    """
    Hin_res = res_enth(Tin, P, calc_PRZ(Tin, P))
    Hout_res = res_enth(Tout, P, calc_PRZ(Tout, P))
    H_ideal = calc_idealH(Tin, Tout)
    Q = mol_flow*(Hout_res - Hin_res + H_ideal)/1000
    return Q # MW