# Importing the required packages.

import sys
sys.path.append(r"C:\Users\SHINJINI MONDAL")
import pyUVProbe as pyuv
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from uncertainties import ufloat
from uncertainties import unumpy as unp
from IPython.display import display, Math, Latex

# Parameters for different analytes - PPO, bis-MSB, PPO+bis-MSB

ANALYTE_PARAMS = {
    "PPO": {
        "SLOPE": ufloat(0.002166648731422, 0.000006337640291), # Slope of calibration curve generated using UV-Vis data of 7 different concentrations.
        "INTERCEPT": 0.0, # Intercept for the calibration curve is forced to be 0.
        # Calibration areas
        "calib_areas": np.array([
            5.5024131675018, 5.520973015700291, 5.590602550422773,
            5.4786149507078035, 5.5110805759710875, 5.471310015816319,
            5.5513141263121115, 5.414075713058992, 5.427893587038852,
            5.405703879194334]), 
        # The start and end wavelengths over which the area under curve is to be integrated. The area under curve to be integrated for the input spectrum should be the same as wavelengths over which integration was done to generate the calibration curve.
        "start_wl": 325.0,
        "end_wl": 335.0
    },
    "bis-MSB": {
        "SLOPE": ufloat(0.125877358900729, 0.001827202420415),  # Slope of calibration curve generated using UV-Vis data of 7 different concentrations.
        "INTERCEPT": 0.0, # Intercept for the calibration curve is forced to be 0.
        # Calibration areas
        "calib_areas": np.array([
            57.4909021178828, 56.799350777235915, 56.9431289807156
        ]),
        # The start and end wavelengths over which the area under curve is to be integrated. The area under curve to be integrated for the input spectrum should be the same as wavelengths over which integration was done to generate the calibration curve.
        "start_wl": 315.0,
        "end_wl": 450.0
    },
    "PPO+bis-MSB": {
        "SLOPE": ufloat(0.333410168320650, 0.001612237505520),  # Slope of calibration curve generated using UV-Vis data of 7 different concentrations.
        "INTERCEPT": 0.0, # Intercept for the calibration curve is forced to be 0.
        # Calibration areas
        "calib_areas": np.array([
            43.37229689635569, 43.40152659121668, 43.555779329035424,
            43.35772034791685, 43.193699101354774, 43.338683465495706, 
            43.199621403162986, 43.15315196391505
        ]),
        # The start and end wavelengths over which the area under curve is to be integrated. The area under curve to be integrated for the input spectrum should be the same as wavelengths over which integration was done to generate the calibration curve.
        "start_wl": 360.0,
        "end_wl": 400.0
    }
}

# Function used to compute uncertainty as the sample standard deviation (σ, 68% CI) by measuring a particular sample 10 times

def compute_sigma(areas):
    return np.std(areas, ddof=1) # standard deviation

# Function to normalize the input spectrum at the desired wavlength taken as input from user.

def normalize_and_extract(wavelengths, absorbance, norm_wl):
    norm_idx = (np.abs(wavelengths - norm_wl)).argmin() # Normalization wavelength is taken as input from user.
    norm_val = absorbance[norm_idx] 
    return absorbance - norm_val # Subtracting the absorbance of entire spectrum with the absorbance at normalization wavelength.

# Function to integrate the area under the absorbance curve given as input in the form of a .spc file

def compute_area(wavelengths, absorbance, start_wl, end_wl):
    """
    Compute area under the curve.
    """
    x = np.asarray(wavelengths)
    y = np.asarray(absorbance)

    # Handle descending wavelengths
    if x[0] > x[-1]:
        x = x[::-1]
        y = y[::-1]

    # Replicate your st/ed selection
    st = np.where(x > start_wl)[0][0]   # first index where x > start_wl
    ed = np.where(x < end_wl)[0][-1]    # last index where x < end_wl

    return sp.integrate.simpson(y[st:ed], x[st:ed])


# Main function that takes input from user in the order - .spc file containing UV-Vis Data, start and stop wavlengths of the input spectrum plot, interval at which the data is taken, normalization wavelength.

def main():

    # Ask for analyte

    analyte_choice = input("Enter analyte (PPO / bis-MSB / PPO+bis-MSB): ").strip()
    if analyte_choice not in ANALYTE_PARAMS:
        print(f"Invalid choice. Choose from: {list(ANALYTE_PARAMS.keys())}")
        sys.exit(1)

    params = ANALYTE_PARAMS[analyte_choice]

    if len(sys.argv) != 6:
        print("Usage: python calculate_concentration.py <file.spc> <spectrum_start> <spectrum_stop> <interval> <norm_wl>")
        sys.exit(1)

    spc_file = sys.argv[1]
    spectrum_start = float(sys.argv[2])
    spectrum_stop = float(sys.argv[3])
    interval = float(sys.argv[4])
    norm_wl = float(sys.argv[5])

    # Determine number of points based on start and stop wavelengths and interval.

    if spectrum_start == 200 and spectrum_stop == 700 and interval == 0.5:
        npoints = 998
    elif spectrum_start == 250 and spectrum_stop == 700 and interval == 0.5:
        npoints = 901
    elif spectrum_start == 300 and spectrum_stop == 700 and interval == 0.5:
        npoints = 802
    else:
        npoints = ((spectrum_stop - spectrum_start)/interval) +1

    # Read the data file.

    spec = pyuv.Data(spc_file, int(npoints))

    print(f"Data read: {npoints} points")
    print(f"Attributes found: absorbance → {hasattr(spec, 'abs')}, wavelength → {hasattr(spec, 'wl')}")

    absorbance = np.array(spec.abs) # Read absorbance values
    wavelengths = np.array(spec.wl) # Read wavelength values

    # Remove invalid wavelength values based on analyte type

    if analyte_choice == "PPO":
        valid = (wavelengths >= 200) & (wavelengths <= 700)
    elif analyte_choice == "bis-MSB":
        valid = (wavelengths >= 300) & (wavelengths <= 700)
    elif analyte_choice == "PPO+bis-MSB":
        valid = (wavelengths >= 250) & (wavelengths <= 700)

    wavelengths = wavelengths[valid]
    absorbance = absorbance[valid]

    # Normalize and compute area

    absorbance = normalize_and_extract(wavelengths, absorbance, norm_wl)
    area = compute_area(wavelengths, absorbance, params["start_wl"], params["end_wl"])
    
    # Uncertainty calculation

    SLOPE = params["SLOPE"]

    # Standard Deviation and standard error of mean

    sigma = compute_sigma(params["calib_areas"])
    m, sigma_m = SLOPE.n, SLOPE.s
    std_error = sigma/np.sqrt(len(params["calib_areas"]))

    # Final propagated uncertainty

    C_nominal = m * area
    sigma_C = np.sqrt((area*sigma_m)**2 + sigma**2)
    C_with_uncertainty = ufloat(C_nominal, sigma_C)

    # Contribution breakdown

    contrib_slope = (area*sigma_m)**2
    contrib_std   = sigma**2

    perc_slope = 100 * contrib_slope / sigma_C 
    perc_std   = 100 * contrib_std / sigma_C 

    # Plot the input data and save it as a spectrum plot

    plt.figure(figsize=(10,8))
    plt.plot(wavelengths, absorbance, label='Normalized Spectrum')
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorbance (normalized)")
    plt.title(f"Spectrum: {spc_file} ({analyte_choice})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("spectrum_plot_{analyte_choice}.png")

    # Print results

    print(f"\n=== {analyte_choice} Results ===")
    print(f"\nArea under curve (from {params['start_wl']} to {params['end_wl']} nm) = {area:.5f}")
    print(f"Estimated Concentration = {C_with_uncertainty:.5uP} g/L\n")
    print(f"Uncertainty propagation through slope calibration: {sigma_m:.10f} g/L")
    print(f"Standard deviation: {sigma:.10f} g/L\n")
    print(f"Uncertainty Breakdown (variance contributions):")
    print(f"From slope: {contrib_slope:.10f} g/L ({perc_slope: 5f}%)")
    print(f"From standard deviation (68% C.I): {contrib_std:.10f} g/L ({perc_std:.5f}%)\n")
    print(f"Total combined uncertainty: {sigma_C:.10f} g/L\n")
    print(f"Standard error of the mean: {std_error:.10f} g/L")
    print(f"\nPlot saved as spectrum_plot_{analyte_choice}.png")

if __name__ == "__main__":
    main()

