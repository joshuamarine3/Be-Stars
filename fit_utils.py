import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from lmfit.models import VoigtModel, ExponentialModel, SplineModel, LinearModel, GaussianModel
from astropy.io import fits
import glob
from astropy.time import Time
import re
from copy import copy
from astropy.table import Table

def extract_datetime_from_filename(filename):
    """Extracts and parses the datetime from the filename."""
    match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})', filename)
    if match:
        datetime_str = match.group(1)
        date, time = datetime_str.split('T')
        formatted_time = time.replace('-', ':', 2).replace('_', '.')
        return datetime.strptime(f"{date}T{formatted_time}", "%Y-%m-%dT%H:%M:%S")
    return None

def organize_files_by_date(file_list):
    """Sorts files by extracted datetime."""
    files_with_dates = []
    for file in file_list:
        extracted_date = extract_datetime_from_filename(file)
        if extracted_date:
            files_with_dates.append((extracted_date, file))

    # Sort by datetime
    files_with_dates.sort(key=lambda x: x[0])

    # Extract sorted filenames
    sorted_files = [file for _, file in files_with_dates]
    return sorted_files

def calculate_equivalent_width(x, y, em_fit, bkg_fit, centroid, fwhm):
    # Define the region of interest (ROI) as +/- 4 * FWHM around the centroid
    roi_min = centroid - 4 * fwhm
    roi_max = centroid + 4 * fwhm

    # Mask the data within the ROI
    mask = (x >= roi_min) & (x <= roi_max)
    x_roi = x[mask]
    y_roi = y[mask]
    em_fit_roi = em_fit[mask] + bkg_fit[mask]
    bkg_fit_roi = bkg_fit[mask]

    # Continuum flux (use the background fit in this case)
    continuum = bkg_fit_roi

    # Compute the equivalent width
    ew = np.trapz(1 - (em_fit_roi / continuum), x_roi)

    # Visualization
    plt.figure(figsize=(10, 6))
    # plt.plot(x, y, label="Observed Flux", alpha=0.5, color="blue")
    plt.plot(x, em_fit + bkg_fit, label="Emission Fit", color="green", linestyle="--")
    plt.plot(x, bkg_fit, label="Background Continuum", color="orange", linestyle=":")

    # Highlight the region of interest (ROI)
    plt.fill_between(x_roi, em_fit_roi, bkg_fit_roi, color="gray", alpha=0.3, label=f"Integrated Region (EW = {ew:.2f})")

    # Indicate the centroid
    plt.axvline(centroid, color="red", linestyle="--", label=f"Centroid (x = {centroid:.2f})")

    # Add labels and legend
    plt.xlabel("Wavelength")
    plt.ylabel("Flux")
    plt.title("Equivalent Width Calculation")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

    return ew

def calculate_vr_ratio(x_halpha, y_halpha, fwhm, centroid_em1, centroid_em2):
    """
    Calculate the V/R ratio for a double-peaked emission profile.

    Parameters:
    - wave_fit: The best fit profile (array of y-values).
    - centroid_em1: Centroid of the first (blue) emission peak (float).
    - centroid_em2: Centroid of the second (red) emission peak (float).

    Returns:
    - vr_ratio: The calculated V/R ratio (float).
    """

    # Ensure centroids are correctly identified as blue and red peaks
    if centroid_em1 < centroid_em2:
        blue_centroid = centroid_em1
        red_centroid = centroid_em2
    else:
        blue_centroid = centroid_em2
        red_centroid = centroid_em1

    wv_per_ind = (x_halpha[-1] - x_halpha[0])/len(x_halpha)

    # Find the indices corresponding to the centroids
    blue_index = np.argmin(np.abs(x_halpha - blue_centroid))
    red_index = np.argmin(np.abs(x_halpha - red_centroid))
    blue_low = int(np.round(blue_index - 2))
    blue_high = int(np.round(blue_index + 2))
    red_low = int(np.round(red_index - 2))
    red_high = int(np.round(red_index + 2))

    print(f'Blue Calc Range: x = {blue_low*wv_per_ind + x_halpha[0]:.3f} to {blue_high*wv_per_ind + x_halpha[0]:.3f}')
    print(f'Red Calc Range: x = {red_low*wv_per_ind + x_halpha[0]:.3f} to {red_high*wv_per_ind + x_halpha[0]:.3f}')

    # Extract the intensities at the centroids
    # intensity_blue = np.median(y_halpha[blue_low:blue_high])
    # intensity_red = np.median(y_halpha[red_low:red_high])
    intensity_blue = np.max(y_halpha[blue_low:blue_high])
    intensity_red = np.max(y_halpha[red_low:red_high])


    # Calculate the V/R ratio
    vr_ratio = intensity_blue / intensity_red

    print(f"Blue Peak Intensity: {intensity_blue:.5f}")
    print(f"Red Peak Intensity: {intensity_red:.5f}")
    print(f"V/R Ratio: {vr_ratio:.5f}")

     # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(x_halpha, y_halpha, label="Best Fit", color="black")
    plt.axvline(blue_centroid, color="blue", linestyle="--", label=f"Blue Peak (λ = {blue_centroid:.2f} Å, I = {intensity_blue:.5f})")
    plt.axvline(red_centroid, color="red", linestyle="--", label=f"Red Peak (λ = {red_centroid:.2f} Å, I = {intensity_red:.5f})")
    plt.scatter([blue_centroid, red_centroid], [intensity_blue, intensity_red], color=["blue", "red"], zorder=5)
    plt.xlabel("Wavelength (Å)")
    plt.ylabel("Intensity")
    plt.title("Double-Peaked Emission Profile and V/R Ratio")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.show()

    return vr_ratio

def voigt_fwhm(sigma, gamma):
    """Calculate the FWHM of a Voigt profile given sigma and gamma."""
    # Approximation for Voigt FWHM
    fwhm_gaussian = 2 * np.sqrt(2 * np.log(2)) * sigma
    fwhm_lorentzian = 2 * gamma
    fwhm_voigt = 0.5346 * fwhm_lorentzian + np.sqrt(0.2166 * fwhm_lorentzian**2 + fwhm_gaussian**2)
    return fwhm_voigt

def voigt_height(sigma, gamma, amplitude):
    """Calculate the height of a Voigt profile given sigma, gamma, and amplitude."""
    # Approximation for Voigt height
    height_gaussian = 1 / (sigma * np.sqrt(2 * np.pi))
    height_lorentzian = 1 / (np.pi * gamma)
    height_voigt = height_gaussian * height_lorentzian * amplitude
    return height_voigt

def Be_double_peak_fit_recursive(x, y, x_guess, amp1, amp2, amp3, variability, min_absorption_width, separation, tellurics, max_attempts=20, chi2_threshold=0.0017):
    # attempts = 1

    # while attempts < max_attempts:
    #     adjustment = attempts//2 * (-1)**(attempts % 2)
    #     print(f"Attempt {attempts} with adjustment {adjustment}")
    #     # Adjust x_guess to explore parameter space
    #     adjusted_x_guess = x_guess + adjustment

    #     # Absorption component
    #     voigt_abs = VoigtModel(prefix='v_abs_')
    #     pars = voigt_abs.make_params(center=adjusted_x_guess, sigma=7, gamma=1)
    #     pars['v_abs_amplitude'].set(value=1, vary=True, max=0)
    #     pars['v_abs_gamma'].set(value=0.5, vary=True, min=0)
    #     pars['v_abs_sigma'].set(value=min_absorption_width*1.5, vary=True, min=min_absorption_width)
    #     # pars.add('v_abs_fwhm', expr='voigt_fwhm(v_abs_sigma, v_abs_gamma)', vary = True, min=min_absorption_width)
    #     # pars.add('v_abs_height', expr='voigt_height(v_abs_sigma, v_abs_gamma, v_abs_amplitude)', vary = True, min=0, max=-0.5)

    #     # Emission components
    #     voigt_em1 = VoigtModel(prefix='v_em1_')
    #     voigt_em2 = VoigtModel(prefix='v_em2_')
    #     pars.update(voigt_em1.make_params(center=adjusted_x_guess - separation / 2, sigma=1, gamma=1))
    #     pars.update(voigt_em2.make_params(center=adjusted_x_guess + separation / 2, sigma=1, gamma=1))
    #     pars['v_em1_amplitude'].set(value=amp1, vary=True)
    #     pars['v_em1_gamma'].set(value=0.5, vary=True, min=0)
    #     pars['v_em2_amplitude'].set(value=amp2, vary=True)
    #     pars['v_em2_gamma'].set(value=0.5, vary=True, min=0)

    #     for telluric, wavelength in tellurics.items():
    #         voigt_tell = VoigtModel(prefix=f'tell_{telluric}_')
    #         pars.update(voigt_tell.make_params())
    #         pars[f'tell_{telluric}_amplitude'].set(value=-0.02, vary=True, max=0, min = -0.05)
    #         pars[f'tell_{telluric}_sigma'].set(value=0.2, vary=True, min=0.1, max=2)
    #         pars[f'tell_{telluric}_center'].set(value=wavelength, vary=False)

    #     # Background component
    #     bkg = LinearModel(prefix='bkg_')
    #     pars.update(bkg.guess(y, x))

    #     # Combined model
    #     mod = voigt_abs + voigt_em1 + voigt_em2 + bkg
    #     for telluric in tellurics:
    #         voigt_tell = VoigtModel(prefix=f'tell_{telluric}_')
    #         mod += voigt_tell

    #     # Fit the data
    #     out = mod.fit(y, pars, x=x)

    #     chi2_red = out.redchi
    #     print(f'Reduced Chi-Squared Value: {chi2_red:.8f}')

    #     # Early stopping if chi-squared is below threshold
    #     if chi2_red > chi2_threshold:
    #         attempts += 1
    #         print(f"Poor fit: chi-squared above threshold. Adjusting x_guess to x = {adjustment + x_guess} Å.")
    #         continue

    #     # Extract centroids and FWHM
    #     centroid_em1 = out.params['v_em1_center'].value
    #     centroid_em2 = out.params['v_em2_center'].value
    #     fwhm_em1 = out.params['v_em1_fwhm'].value
    #     fwhm_em2 = out.params['v_em2_fwhm'].value
    #     # fwhm_em = (fwhm_em1 + fwhm_em2)

    #     fit_amp1 = out.params['v_abs_height'].value
    #     fit_amp2 = out.params['v_em1_height'].value
    #     fit_amp3 = out.params['v_em2_height'].value

    #     print(f'Fit Amplitudes (abs, em1, em2):{fit_amp1, fit_amp2, fit_amp3}')

    #     # Validate centroid conditions
    #     if 2 > abs(centroid_em1 - centroid_em2) > 3 * separation:
    #         attempts += 1
    #         print(f"Poor fit: Centroids invalid. Adjusting x_guess to x = {adjustment+ x_guess} Å.")
    #         continue
    #     # # Validate amplitude conditions
    #     # elif abs(fit_amp3-fit_amp1) > 2*abs(amp3-amp1) or abs(fit_amp2-fit_amp1) > 1.5*(amp2-amp1) or np.round(fit_amp3, 1) == 0 or np.round(fit_amp2, 1) == 0:
    #     #     attempts += 1
    #     #     print(f"Poor fit: Amplitudes too large. Adjusting x_guess to x = {adjustment+ x_guess} Å.")
    #     #     continue
    #     # Validate FWHM conditions
    #     elif fwhm_em1 > 3 * separation or fwhm_em2 > 2 * separation:
    #         attempts += 1
    #         print(f"Poor fit: FWHM too large. Adjusting x_guess to x = {adjustment+ x_guess} Å.")
    #         continue
    #     else:
    #         # Successful fit
    #         print(f"Fit succeeded on attempt {attempts + 1}. Centroids: {centroid_em1:.2f}, {centroid_em2:.2f}")
            
    #         return out
    

    # print("Failed to converge after maximum attempts.")
    # return None
    attempts = 1

    while attempts < max_attempts:
        adjustment = attempts//2 * (-1)**(attempts % 2)
        print(f"Attempt {attempts} with adjustment {adjustment}")
        # Adjust x_guess to explore parameter space
        adjusted_x_guess = x_guess + adjustment

        # Absorption component
        voigt_abs = VoigtModel(prefix='v_abs_')
        pars = voigt_abs.make_params(center=adjusted_x_guess, sigma=1, gamma=1)
        pars['v_abs_amplitude'].set(value=amp1, vary=True, max=0)
        pars['v_abs_gamma'].set(value=0.5, vary=True, min=0)

        # Emission components
        voigt_em1 = VoigtModel(prefix='v_em1_')
        voigt_em2 = VoigtModel(prefix='v_em2_')
        pars.update(voigt_em1.make_params(center=adjusted_x_guess - separation / 2, sigma=1, gamma=1))
        pars.update(voigt_em2.make_params(center=adjusted_x_guess + separation / 2, sigma=1, gamma=1))
        pars['v_em1_amplitude'].set(value=amp2, vary=True, min=0)
        pars['v_em1_gamma'].set(value=0.5, vary=True, min=0)
        pars['v_em2_amplitude'].set(value=amp3, vary=True, min=0)
        pars['v_em2_gamma'].set(value=0.5, vary=True, min=0)


        # Background component
        bkg = LinearModel(prefix='bkg_')
        pars.update(bkg.guess(y, x))

        # Combined model
        mod = voigt_abs + voigt_em1 + voigt_em2 + bkg

        # Fit the data
        out = mod.fit(y, pars, x=x)

        chi2_red = out.redchi
        print(f'Reduced Chi-Squared Value: {chi2_red:.8f}')

        # Early stopping if chi-squared is below threshold
        if chi2_red > chi2_threshold:
            attempts += 1
            print(f"Poor fit: chi-squared above threshold. Adjusting x_guess to x = {adjustment + x_guess} Å.")
            continue

        # Extract centroids and FWHM
        centroid_em1 = out.params['v_em1_center'].value
        centroid_em2 = out.params['v_em2_center'].value
        fwhm_em1 = out.params['v_em1_fwhm'].value
        fwhm_em2 = out.params['v_em2_fwhm'].value
        # fwhm_em = (fwhm_em1 + fwhm_em2)

        fit_amp1 = out.params['v_abs_height'].value
        fit_amp2 = out.params['v_em1_height'].value
        fit_amp3 = out.params['v_em2_height'].value

        print(f'Fit Amplitudes (abs, em1, em2):{fit_amp1, fit_amp2, fit_amp3}')

        # Validate centroid conditions
        if 2 > abs(centroid_em1 - centroid_em2) > 3 * separation:
            attempts += 1
            print(f"Poor fit: Centroids invalid. Adjusting x_guess to x = {adjustment+ x_guess} Å.")
            continue
        # Validate amplitude conditions
        elif abs(fit_amp3-fit_amp1) > 2*abs(amp3-amp1) or abs(fit_amp2-fit_amp1) > 1.5*(amp2-amp1) or np.round(fit_amp3, 1) == 0 or np.round(fit_amp2, 1) == 0:
            attempts += 1
            print(f"Poor fit: Amplitudes too large. Adjusting x_guess to x = {adjustment+ x_guess} Å.")
            continue
        # Validate FWHM conditions
        elif fwhm_em1 > 3 * separation or fwhm_em2 > 2 * separation:
            attempts += 1
            print(f"Poor fit: FWHM too large. Adjusting x_guess to x = {adjustment+ x_guess} Å.")
            continue
        else:
            # Successful fit
            print(f"Fit succeeded on attempt {attempts + 1}. Centroids: {centroid_em1:.2f}, {centroid_em2:.2f}")
            
            return out

    print("Failed to converge after maximum attempts.")
    return None

def Be_single_peak_fit_recursive(x, y, x_guess, amp1, variability, min_absorption_width, tellurics, max_attempts=20, chi2_threshold=0.0017):
    attempts = 1

    while attempts <= max_attempts:
        adjustment = attempts//2 * (-1)**(attempts % 2)
        print(f"Attempt {attempts} with adjustment {adjustment}")
        # Adjust x_guess to explore parameter space
        adjusted_x_guess = x_guess + adjustment

       # Absorption component
        # voigt_abs = VoigtModel(prefix='v_abs_')
        # pars = voigt_abs.make_params(center=adjusted_x_guess, sigma=1, gamma=1)
        # pars['v_abs_amplitude'].set(value=-.3, vary=True, max=0, min = -0.5)
        # pars['v_abs_gamma'].set(value=0.5, vary=True, min=0)
        # pars['v_abs_sigma'].set(value=min_absorption_width*1.5, vary=True, min=min_absorption_width)
        # Absorption component
        voigt_abs = VoigtModel(prefix='v_abs_')
        pars = voigt_abs.make_params(center=adjusted_x_guess, sigma=1, gamma=1)
        pars['v_abs_amplitude'].set(value=-.3, vary=True, max=0, min = -0.5)
        pars['v_abs_gamma'].set(value=0.5, vary=True, min=0)

        # Emission component
        # voigt_em = VoigtModel(prefix='v_em_')
        # pars.update(voigt_em.make_params(center=adjusted_x_guess, sigma=1, gamma=1))
        # pars['v_em_amplitude'].set(value=amp1, vary=True, min=amp1/variability, max = amp1*variability)
        # pars['v_em_gamma'].set(value=0.5, vary=True, min=0)

        voigt_em = VoigtModel(prefix='v_em_')
        pars.update(voigt_em.make_params(center=adjusted_x_guess, sigma=1, gamma=1))
        pars['v_em_amplitude'].set(value=amp1, vary=True, min=amp1/variability)
        pars['v_em_gamma'].set(value=0.5, vary=True, min=0)

        for telluric, wavelength in tellurics.items():
            voigt_tell = VoigtModel(prefix=f'tell_{telluric}_')
            pars.update(voigt_tell.make_params())
            pars[f'tell_{telluric}_amplitude'].set(value=-0.02, vary=True, max=0, min = -0.05)
            pars[f'tell_{telluric}_sigma'].set(value=0.2, vary=True, min=0.1, max=2)
            pars[f'tell_{telluric}_center'].set(value=wavelength, vary=False)

        # Background component
        bkg = LinearModel(prefix='bkg_')
        pars.update(bkg.guess(y, x))

        # Combined model
        mod = voigt_abs + voigt_em + bkg
        for telluric in tellurics:
            voigt_tell = VoigtModel(prefix=f'tell_{telluric}_')
            mod += voigt_tell

        # Fit the data
        out = mod.fit(y, pars, x=x)

        chi2_red = out.redchi
        print(f'Reduced Chi-Squared Value: {chi2_red:.8f}')

        # Early stopping if chi-squared is below threshold
        if chi2_red > chi2_threshold:
            adjustment += max_attempts/20
            attempts += 1
            print(f"Poor fit: chi-squared above threshold. Adjusting x_guess to x = {adjustment + x_guess} Å.")
            continue


        # Extract centroids and FWHM
        centroid_em = out.params['v_em_center'].value
        fwhm_em = out.params['v_em_fwhm'].value

        fit_amp1 = out.params['v_abs_height'].value
        fit_amp2 = out.params['v_em_height'].value

        print(f'Fit Amplitudes (abs, em):{fit_amp1, fit_amp2}')

        # Validate centroid conditions
        if abs(centroid_em - x_guess) > 10:
            attempts += 1
            print(f"Poor fit: Centroids invalid. Adjusting x_guess to x = {adjustment + x_guess} Å.")
            continue
        # Validate amplitude conditions
        # elif abs(fit_amp2-fit_amp1) > 2*(amp2-amp1) or np.round(fit_amp2, 1) == 0:
        #     attempts += 1
        #     print(f"Poor fit: Amplitudes too large. Adjusting x_guess to x = {adjustment + x_guess} Å.")
        #     continue
        # Validate FWHM conditions
        elif fwhm_em > 20:
            attempts += 1
            print(f"Poor fit: FWHM too large. Adjusting x_guess to x = {adjustment + x_guess} Å.")
            continue
        else:
            # Successful fit
            print(f"Fit succeeded on attempt {attempts + 1}. Centroid: {centroid_em:.2f}")

            return out

    print("Failed to converge after maximum attempts.")
    return None

def Be_absorption_fit_recursive(x, y, x_guess, min_absorption_width, tellurics, max_absorption_depth=-0.5, max_attempts=20, chi2_threshold=0.0017):
    attempts = 0

    while attempts < max_attempts:
        adjustment = attempts//2 * (-1)**(attempts % 2)
        print(f"Attempt {attempts + 1} with adjustment {adjustment}")
        # Adjust x_guess to explore parameter space
        adjusted_x_guess = x_guess + adjustment

        # Absorption component
        # voigt_abs = VoigtModel(prefix='v_abs_')
        # pars = voigt_abs.make_params(center=adjusted_x_guess, sigma=1, gamma=1)
        # pars['v_abs_amplitude'].set(value=-0.3, vary=True, max=0, min = -0.5)
        # pars['v_abs_gamma'].set(value=0.5, vary=True, min=0)
        # pars['v_abs_sigma'].set(value=min_absorption_width, vary=True, min=min_absorption_width)

        # Voigt background component
        voigt_abs = VoigtModel(prefix='v_abs_')
        pars = voigt_abs.make_params(center=adjusted_x_guess, sigma=5, gamma=1)
        pars['v_abs_sigma'].set(value=5, vary=True, min=0)
        pars['v_abs_gamma'].set(value=1, vary=True, min=0)
        pars['v_abs_amplitude'].set(value=-1, vary=True, max=0)
        # pars.add('v_abs_fwhm', value=min_absorption_width*2+1, vary=True, min=min_absorption_width)
        # pars['v_abs_gamma'].set(expr='1/(pi*v_abs_sigma*v_abs_fwhm*sqrt(2*pi))', vary=False)
        # pars.add('v_abs_height', value=-0.1, vary = True, max=0, min=max_absorption_depth)
        # pars['v_abs_amplitude'].set(expr='v_abs_height * v_abs_sigma * sqrt(2*pi) * (pi * v_abs_gamma)', vary=False)
        # pars.add('v_abs_fwhm', expr='0.5346 * (2 * v_abs_gamma) + sqrt(0.2166 * (2 * v_abs_gamma)**2 + (2 * sqrt(2 * log(2)) * v_abs_sigma)**2)', vary = True, min=min_absorption_width)
        # pars.add('v_abs_height', expr='(1 / (v_abs_sigma * sqrt(2 * pi))) * (1 / (pi * v_abs_gamma)) * v_abs_amplitude', vary = True, max=0, min=max_absorption_depth)
        # pars['v_abs_sigma'].set(value=min_absorption_width, vary=True, min=min_absorption_width)

        for telluric, wavelength in tellurics.items():
            voigt_tell = VoigtModel(prefix=f'tell_{telluric}_')
            pars.update(voigt_tell.make_params())
            pars[f'tell_{telluric}_amplitude'].set(value=-0.02, vary=True, max=0, min = -0.05)
            pars[f'tell_{telluric}_sigma'].set(value=0.2, vary=True, min=0.1, max=2)
            pars[f'tell_{telluric}_center'].set(value=wavelength, vary=False)

        # Background component
        bkg = LinearModel(prefix='bkg_')
        pars.update(bkg.guess(y, x))

        # Combined model
        mod = voigt_abs + bkg
        for telluric in tellurics:
            voigt_tell = VoigtModel(prefix=f'tell_{telluric}_')
            mod += voigt_tell

        # Fit the data
        out = mod.fit(y, pars, x=x)

        chi2_red = out.redchi
        print(f'Reduced Chi-Squared Value: {chi2_red:.8f}')

        # Early stopping if chi-squared is below threshold
        if chi2_red > chi2_threshold:
            attempts += 1
            print(f"Poor fit: chi-squared above threshold. Adjusting x_guess to x = {adjustment + x_guess} Å.")
            continue

        # Extract fit results
        comps = out.eval_components(x=x)
        bkg_fit = comps['bkg_']
        abs_fit = comps['v_abs_']

        # Extract centroids and FWHM
        centroid_abs = out.params['v_abs_center'].value
        fwhm_abs = out.params['v_abs_fwhm'].value

        fit_amp1 = out.params['v_abs_height'].value

        print(f'Fit Amplitudes (abs):{fit_amp1}')

        # Validate amplitude conditions
        # if fit_amp1 > 0 or abs(fit_amp1) > 2*abs(amp1) or np.round(fit_amp1, 1) == 0:
        #     attempts += 1
        #     print(f"Poor fit: Amplitudes too large. Adjusting x_guess to x = {adjustment + x_guess} Å.")
        #     continue
        # Validate FWHM conditions
        # if fwhm_abs > 30:
        #     attempts += 1
        #     print(f"Poor fit: FWHM too large. Adjusting x_guess to x = {adjustment + x_guess} Å.")
        #     continue
        # else:
            # Successful fit
        print(f"Fit succeeded on attempt {attempts + 1}. Centroids: {centroid_abs:.2f}")

        return out

    print("Failed to converge after maximum attempts.")
    return None

def Be_double_absorption_fit_recursive(x, y, x_guess, amp1, amp2, max_attempts=20, chi2_threshold=0.0017):
    attempts = 1

    while attempts <= max_attempts:
        adjustment = attempts//2 * (-1)**(attempts % 2)
        print(f"Attempt {attempts} with adjustment {adjustment}")
        # Adjust x_guess to explore parameter space
        adjusted_x_guess = x_guess + adjustment

        # Left Absorption component
        voigt_abs1 = VoigtModel(prefix='v_abs1_')
        pars = voigt_abs1.make_params(center=adjusted_x_guess, sigma=1, gamma=1)
        pars['v_abs1_amplitude'].set(value=amp1, vary=True, max=0)
        pars['v_abs1_gamma'].set(value=0.5, vary=True, min=0)

        # Right Absorption component
        voigt_abs2 = VoigtModel(prefix='v_abs2_')
        pars.update(voigt_abs2.make_params(center=adjusted_x_guess, sigma=1, gamma=1))
        pars['v_abs2_amplitude'].set(value=amp2, vary=True, max=0)
        pars['v_abs2_gamma'].set(value=0.5, vary=True, min=0)

        # Background component
        bkg = LinearModel(prefix='bkg_')
        pars.update(bkg.guess(y, x))

        # Combined model
        mod = voigt_abs1 + voigt_abs2 + bkg

        # Fit the data
        out = mod.fit(y, pars, x=x)

        chi2_red = out.redchi
        print(f'Reduced Chi-Squared Value: {chi2_red:.8f}')

        # Early stopping if chi-squared is below threshold
        if chi2_red > chi2_threshold:
            adjustment += max_attempts/20
            attempts += 1
            print(f"Poor fit: chi-squared above threshold. Adjusting x_guess to x = {adjustment + x_guess} Å.")
            continue


        # Extract centroids and FWHM
        centroid_abs1 = out.params['v_abs1_center'].value
        fwhm_abs1 = out.params['v_abs1_fwhm'].value

        fit_amp1 = out.params['v_abs1_amplitude'].value
        fit_amp2 = out.params['v_abs2_amplitude'].value
        print(f'Fit Amplitudes (abs, em):{fit_amp1, fit_amp2}')

        # Validate centroid conditions
        if abs(centroid_abs1 - x_guess) > 10:
            attempts += 1
            print(f"Poor fit: Centroids invalid. Adjusting x_guess to x = {adjustment + x_guess} Å.")
            continue
        # Validate amplitude conditions
        # elif abs(fit_amp1) > abs(2*amp1) or abs(fit_amp2) > abs(2*amp2):
        #     attempts += 1
        #     print(f"Poor fit: Amplitudes too large. Adjusting x_guess to x = {adjustment + x_guess} Å.")
        #     continue
        # # Validate FWHM conditions
        elif fwhm_abs1 > 30:
            attempts += 1
            print(f"Poor fit: FWHM too large. Adjusting x_guess to x = {adjustment + x_guess} Å.")
            continue
        else:
            # Successful fit
            print(f"Fit succeeded on attempt {attempts + 1}. Centroid: {centroid_abs1:.2f}")

            return out

    print("Failed to converge after maximum attempts.")
    return None

def calculate_equivalent_width_abs(x, y, best_fit, bkg_fit, centroid, fwhm):
    """
    Calculate the equivalent width (EW) of a spectral feature.

    Parameters:
    - x (array): Wavelength array.
    - y (array): Observed flux array.
    - best_fit (array): Best-fit flux array from the Voigt model.
    - bkg_fit (array): Background continuum fit array.
    - centroid (float): Centroid of the spectral feature.
    - fwhm (float): Full width at half maximum of the feature.

    Returns:
    - float: The calculated equivalent width (EW).
    """
    # Define the region of interest (ROI) as +/- 4 * FWHM around the centroid
    roi_min = centroid - 4 * fwhm
    roi_max = centroid + 4 * fwhm

    # Mask the data within the ROI
    mask = (x >= roi_min) & (x <= roi_max)
    x_roi = x[mask]
    y_roi = y[mask]
    best_fit_roi = best_fit[mask]
    bkg_fit_roi = bkg_fit[mask]
    # abs_fit_roi = abs_fit[mask]

    # Continuum flux (use the background fit in this case)
    continuum = bkg_fit_roi

    # Compute the equivalent width
    ew = np.trapz(1 - (best_fit_roi / continuum), x_roi)

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="Observed Flux", alpha=0.5, color="blue")
    plt.plot(x, best_fit, label="Best Fit", color="green", linestyle="--")
    # plt.plot(x, bkg_fit, label="Background Continuum", color="orange", linestyle=":")

    # Highlight the region of interest (ROI)
    plt.fill_between(x_roi, y_roi, bkg_fit_roi, color="gray", alpha=0.3, label=f"Integrated Region (EW = {ew:.2f})")

    # Indicate the centroid
    plt.axvline(centroid, color="red", linestyle="--", label=f"Centroid (x = {centroid:.2f})")

    # Add labels and legend
    plt.xlabel("Wavelength")
    plt.ylabel("Flux")
    plt.title("Equivalent Width Calculation")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

    return ew

# ----------------------------
# BeSSSpectra: WCS-free
# ----------------------------
def BeSSSpectra(file):
    """
    Load a 1-D BeSS spectrum from FITS.

    Returns:
        wavelengths: np.ndarray (linear solution from header)
        spectrum: np.ndarray
        header: FITS header
    """
    try:
        f = fits.open(file)
        header = f[0].header
        spectrum = f[0].data

        # Linear wavelength solution from header
        crval1 = header.get('CRVAL1', 6563.0)  # default near Hα
        cdelt1 = header.get('CDELT1', 1.0)
        crpix1 = header.get('CRPIX1', 1.0)

        wavelengths = crval1 + (np.arange(len(spectrum)) + 1 - crpix1) * cdelt1

        return wavelengths, spectrum, header

    except Exception as e:
        print(f"⚠️ Failed to read {file}: {e}")
        return None, None, None

# ----------------------------
# sort_files: wavelength filter + DATE-OBS from header
# ----------------------------
def sort_files(file_paths, h_alpha=6563.0):
    """
    Collect FITS files covering Hα, store observation dates from header.

    Returns:
        valid_files: list of paths
        valid_times: list of astropy Time objects (from DATE-OBS)
    """
    files = glob.glob(file_paths)
    valid_files = []
    valid_times = []

    print(f"Found {len(files)} files in directory.")

    for file in files:
        try:
            wavelengths, spectrum, header = BeSSSpectra(file)
            min = np.min(wavelengths)
            max = np.max(wavelengths)
            peak = np.max(spectrum)
            if wavelengths is None or peak > 1000:
                continue

            # Wavelength coverage filter
            if min > h_alpha or max < h_alpha or np.abs(min - h_alpha) > 1500 or np.abs(max - h_alpha) > 1500:
                continue

            # Get observation date from header
            date_obs = header.get('DATE-OBS', None)
            if date_obs is None:
                print(f"⚠️ {file} has no DATE-OBS")
                continue

            try:
                date_obj = Time(date_obs, format='isot', scale='utc')
            except Exception:
                # Fallback for non-standard DATE-OBS
                date_obj = Time(datetime.strptime(date_obs[:10], "%Y-%m-%d"))

            valid_files.append(file)
            valid_times.append(date_obj)

        except Exception as e:
            print(f"⚠️ Skipping {file} due to error: {e}")
            continue

    # Sort by date
    if valid_files:
        sorted_pairs = sorted(zip(valid_times, valid_files))
        valid_times, valid_files = zip(*sorted_pairs)
    else:
        valid_times, valid_files = [], []

    print(f"{len(valid_files)} files passed Hα filter and have DATE-OBS.")

    return list(valid_files), list(valid_times)

# ----------------------------
# spec_grid: extract spectra around Hα
# ----------------------------
def spec_grid(target, radius, bess_files_sorted, h_alpha=6564.46):
    """
    Extract spectral segments around Hα from 1-D spectra.

    Args:
        target: placeholder, not used here
        radius: number of pixels on either side of target wavelength
        bess_files_sorted: list of FITS file paths
        h_alpha: target rest wavelength

    Returns:
        result: concatenated np.ndarray of segments
    """
    result = np.array([])

    for fname in bess_files_sorted:
        wavelengths, spectrum, header = BeSSSpectra(fname)
        if wavelengths is None:
            continue

        # Find closest pixel to Hα
        index = np.argmin(np.abs(wavelengths - h_alpha))
        if index - radius < 0 or index + radius > len(spectrum):
            continue

        segment = spectrum[index - radius:index + radius]
        result = np.append(result, segment)

    return result

def stack_daily_spectra(spectra, days_from_start, wavelength_corrections, mask):
  unique_days = np.unique(days_from_start)
  stacked_wavelength_corrections = np.zeros(len(unique_days))
  stacked = copy(spectra[0:len(unique_days)])
  for j in range(len(unique_days)):
    d = unique_days[j]
    x = np.zeros(0)
    y = np.zeros(0)
    for i in [range(len(spectra))[i] for i in range(len(spectra)) if mask[i]]:
      if days_from_start[i] == d:
        x = np.append(x, spectra[i]['Wavelength'])
        y = np.append(y, spectra[i]['Flux'])
        stacked_wavelength_corrections[j] = wavelength_corrections[i]
    tab = Table([x,y], names=('Wavelength', 'Flux'))
    stacked[j] = tab.group_by('Wavelength')
  return stacked, stacked_wavelength_corrections

def select_tellurics(folder, molecules, min_wavelength, max_wavelength, min_intensity = 0):
    tellurics = dict([])
    for molecule in molecules:
        telluric_files = glob.glob(f'{folder}*_{molecule}_*.txt')
        for file in telluric_files:
            data = np.loadtxt(file, skiprows=1)
            wavelengths = data[:, 0] * 10000  # Convert from microns to Angstroms
            intensities = data[:, 1]
            for i in range(len(wavelengths)):
                if intensities[i] >= min_intensity and min_wavelength <= wavelengths[i] <= max_wavelength:
                    tellurics[molecule + f'_{wavelengths[i]:.0f}'] = wavelengths[i]
    return tellurics