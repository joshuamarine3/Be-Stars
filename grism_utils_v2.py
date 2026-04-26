#imports and definitions
import os
from xml.parsers.expat import model
import numpy as np
import pandas as pd
from datetime import date

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
from matplotlib.offsetbox import AnchoredText

import astropy.units as u
from astropy.io.fits import getdata
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from astropy.modeling.polynomial import Polynomial1D
from astropy.modeling.fitting import LinearLSQFitter

from scipy import stats
from scipy.ndimage import rotate
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.interpolate import LSQUnivariateSpline
from scipy.signal import medfilt, find_peaks, detrend

from lmfit.models import VoigtModel, ExponentialModel, SplineModel, LinearModel, GaussianModel, PolynomialModel, ConstantModel
from lmfit import report_fit

from astroquery.simbad import Simbad

import astroscrappy

    
############################
# Calibration Spectrum Class
############################
    
class spectrum:
    def __init__(self, grism_image, calib_spectrum, filter, calib_img = False):
        """
        The main class for grism spectrum calibration generation and analysis.

        calib_spectrum contains all the functions required for extracting, deriving calibration for, and visualizing spectra from grism images.

        Parameters
        ----------
        grism_image : str
            location of the grism image you want to work on.

        rot_angle : int
            read from the calibration file; rotation angle of extraction box


        Other Parameters
        ----------------
        TBD

        Raises
        ------
        TBD

        See Also
        --------
        TBD

        Notes
        -----
        TBD

        References
        ----------
        TBD

        Examples
        --------
        TBD
        """

        self.grism_image = grism_image
        self.filter = filter
        self.balmer = {
            "Hε": 3970.1,
            "Hδ": 4101.7,
            "Hγ": 4340.5,
            "Hβ": 4861.4,
            "Hα": 6562.8
        }
        self.curved = False

        if calib_img:
            self.calib_spectrum = None
        else:
            self.calib_spectrum = calib_spectrum
            self.wave_calib = calib_spectrum.wave_calib if calib_spectrum else None
            self.gain_spline = calib_spectrum.gain_spline if calib_spectrum else None

        self.flagged = False

        ''' Open image, extract header information '''
        im, hdr = getdata(grism_image, 1, header=True)
        if filter == 'hrg':
            self.im = np.fliplr(im)
            self.R = 2000
            self.wavelength_range = [6200,7000]
        if filter == 'lrg':
            self.im = im
            self.R = 400
            self.wavelength_range = [3850,7250]

        self.hdr = hdr
        self.object_name  = hdr.get('BLKNAME', None)
        self.obs_date     = hdr['DATE-OBS'].replace('T',' ')
        self.camera       = hdr['INSTRUME'][0:9] # either 'QHYCCD-Ca' or 'ASI Camer'
        self.focus        = hdr.get('FOCPOS', None)
        self.exp_time     = hdr['EXPTIME']
        self.telescope    = hdr['OBSNAME']
        self.imsize_x = hdr['NAXIS1'] ; self.imsize_y = hdr['NAXIS2']
        self.airmass = hdr['AIRMASS']
        self.moonangle = hdr.get('MOONANGL', None)
        self.moonphase = hdr.get('MOONPHAS', None) 
        self.ra = hdr['TELRA']
        self.dec = hdr['TELDEC']
        self.az = hdr['TELAZ']
        self.alt = hdr['TELALT']
        self.cent = hdr.get('CENTERED', None)
        self.winstabl = hdr.get('WINSTABL', None)
        if self.cent is True or self.winstabl is True:
            self.centered = True
        else:
            self.centered = False

        # Create default plot title
        self.title = '%s (%s)\nGrism spectrum: %s %s' % \
        (self.object_name, self.obs_date, self.telescope, self.camera)

    def summary_info(self):
        """
        Prints key parameters from the current instantiation.
        """
        return self.object_name, self.obs_date, self.im, self.rotangle, self.box, self.wave_calib, self.gain_spline

    def remove_hot_pixels(self, readnoise = 3, niter=1, verbose=False):
        """
        Conservative astroscrappy hot-pixel removal for grism images.
        """
        cal_image = np.array(self.im, dtype=float, copy=True)

        mask, cal_image = astroscrappy.detect_cosmics(
            cal_image,
            niter=niter,
            gain = 0.25,
            sigclip=5.0,
            sigfrac=0.1,
            objlim=15.0,
            satlevel = 50000,
            readnoise=readnoise,
            verbose=verbose
        )

        self.hot_pixel_mask = mask
        self.im = cal_image

        if hasattr(self, "hdr") and self.hdr is not None:
            self.hdr.add_history(
                f"Hot pixels removed (conservative spectral-trace mode, niter={niter}, "
                f"readnoise={readnoise}, sigclip=20.0, objlim=15.0)"
            )

        return mask, cal_image

    def fit_trace(self, plot=True, ymin=None, ymax=None, curved=True, show_points=False):

        data = self.im

        # --- define y-range ---
        if self.filter == 'hrg':
            xmin = 1000
            xmax = 3695
            if ymin is None or ymax is None:
                ymin = 1450
                ymax = 1720

        if self.filter == 'lrg':
            if self.camera == 'ASI Camer':
                xmin = 1700
                xmax = 3250
                if ymin is None or ymax is None:
                    ymin = 1720
                    ymax = 1850
            if self.camera == 'QHYCCD-Ca':
                xmin = 2000
                xmax = 3250
                if ymin is None or ymax is None:
                    ymin = 1600
                    ymax = 1900

        yvals = np.argmax(data[ymin:ymax, :], axis=0) + ymin
        xvals = np.arange(len(data[0]))

        self.trace_xvals = xvals
        self.trace_yvals = yvals

        bad_pixels = (yvals < ymin) | (yvals > ymax) | (xvals > xmax) | (xvals < xmin)
        fit_mask = ~bad_pixels

        if curved is True:
            self.curved = True
            # Fit a 2nd-order polynomial
            polymodel = Polynomial1D(degree=2)
            linfitter = LinearLSQFitter()
            fitted_polymodel = linfitter(
                polymodel, xvals[fit_mask], yvals[fit_mask]
            )
        else:
            self.curved = False
            # Fit a 1st-order polynomial (linear fit)
            polymodel = Polynomial1D(degree=1)
            linfitter = LinearLSQFitter()
            fitted_polymodel = linfitter(
                polymodel, xvals[fit_mask], yvals[fit_mask]
            )

        trace_center = fitted_polymodel(xvals)

        if self.filter == 'hrg':
            if self.camera == 'ASI Camer':
                over = 60
                under = 70
            if self.camera == 'QHYCCD-Ca':
                over = 60
                under = 80
            cutouts = np.array(
                [
                    data[int(yval) - under : int(yval) + over, ii]
                    for yval, ii in zip(trace_center, xvals)
                ]
            )

        if self.filter == 'lrg':
            under = 35
            over = 55
            cutouts = np.array(
                [
                    data[int(yval) - under : int(yval) + over, ii]
                    for yval, ii in zip(trace_center, xvals)
                ]
            )

        self.trace_center = trace_center
        self.trace_model = fitted_polymodel
        self.cutouts = cutouts

        subim = cutouts.T
        self.subim = subim

        col_flux = np.sum(subim, axis=0)
        self.trace_x_center = float((xvals[-1]+xvals[0])/2.0)
        self.trace_y_center = float(fitted_polymodel(self.trace_x_center))

        self.flux_x_center = float(np.sum(xvals * col_flux) / np.sum(col_flux))
        self.flux_y_center = float(fitted_polymodel(self.flux_x_center))

        if plot is True:
            plt.figure(figsize=(10, 8))
            ax1 = plt.subplot(2, 1, 1)
            ax1.set_title('Trace Fit + Extraction Region')
            ax1.imshow(
                data[ymin:ymax, :],
                extent=[0, data.shape[1], ymin, ymax],
                origin='lower'
            )
            ax1.set_aspect(8)
            ax1.plot(
                xvals,
                fitted_polymodel(xvals),
                'w',
                linewidth=1,
                label='Flux-Weighted Trace Center'
            )

            if show_points:
                ax1.scatter(
                    xvals[fit_mask],
                    yvals[fit_mask],
                    s=5,
                    color='cyan',
                    alpha=0.7,
                    label='Points Used for Fit'
                )

            ax1.axis((xmin - 800, xmax + 800, ymin, ymax))
            ax1.set_xlabel("X-Pixel")
            ax1.set_ylabel("Y-Pixel")
            ax1.fill_between(
                xvals,
                fitted_polymodel(xvals) - under,
                fitted_polymodel(xvals) + over,
                color="lime",
                alpha=0.2,
                label="Full Extraction Region (Trace + Sky Bkg)",
            )

            ax1.scatter(
                [self.trace_x_center],
                [self.trace_y_center],
                color='red',
                s=40,
                marker='x',
                label=f'Trace Center ({self.trace_x_center:.1f}, {self.trace_y_center:.1f})'
            )

            ax1.legend(loc='best')
            plt.tight_layout()
            plt.show()


        return trace_center, cutouts, subim
    
    
    def trace_geometry(self):
        """
        Return the global detector (x, y) location of the center of the fitted trace,
        where x is the midpoint of the fitted x-range [xmin, xmax].
        """
        if not hasattr(self, "trace_model"):
            raise AttributeError("Run fit_trace() first.")

        x0 = self.trace_x_center
        y0 = self.trace_y_center

        degree = self.trace_model.degree
        coeffs = np.array([getattr(self.trace_model, f'c{i}').value for i in range(degree + 1)])

        dydx = 0.0
        for i in range(1, degree + 1):
            dydx += i * coeffs[i] * x0**(i - 1)

        d2ydx2 = 0.0
        for i in range(2, degree + 1):
            d2ydx2 += i * (i - 1) * coeffs[i] * x0**(i - 2)

        angle_deg = float(np.degrees(np.arctan(dydx)))
        curvature = float(np.abs(d2ydx2) / (1.0 + dydx**2)**1.5)

        self.trace_angle_deg = angle_deg
        self.trace_curvature = curvature

        return {
            "x": float(x0),
            "y": y0,
            "slope": float(dydx),
            "angle_deg": angle_deg,
            "curvature": curvature
        }

    def plot_box(self, vmin=None, vmax=None, cmap='gray', sat_level=50000, fullwell=65500):
        """
        Plot extraction subimage with robust saturation diagnostics overlay.
        """

        subim = self.subim

        fig, ax = plt.subplots(1, figsize=(20, 10))

        # -------------------------
        # Robust ADU / saturation stats (no histogram)
        # -------------------------
        data = np.asarray(subim)

        # ignore NaNs if any
        finite = np.isfinite(data)
        if np.any(finite):
            d = data[finite].astype(float)
        else:
            d = data.ravel().astype(float)

        adu_max = np.max(d)
        p99   = np.percentile(d, 99.0)
        p999  = np.percentile(d, 99.9)
        p9999 = np.percentile(d, 99.99)
        p99999 = np.percentile(d, 99.999)

        n_pix = d.size
        n_sat = int(np.sum(d >= sat_level))
        frac_sat = 100.0 * n_sat / max(n_pix, 1)

        n_full = int(np.sum(d >= fullwell))
        frac_full = 100.0 * n_full / max(n_pix, 1)

        # -------------------------
        # Your existing scaling
        # -------------------------
        vmin_p, vmax_p = np.percentile(d, [15, 99])
        norm = mcolors.LogNorm(vmin=max(vmin_p, 1e-6), vmax=vmax_p * 1.6)

        ax.imshow(
            subim,
            cmap=cmap,
            vmin=None if norm else vmin,
            vmax=None if norm else vmax,
            aspect=12,
            origin='lower',
            norm=norm
        )

        ax.set_title(f"{self.object_name} ({self.obs_date}): Extraction Box")

        # # -------------------------
        # # Draw your regions
        # # -------------------------
        # y_center = len(subim) / 2
        # ax.fill_between([0, self.imsize_x], y_center-y_center*extract_percent/100, y_center+y_center*extract_percent/100,
        #                 color='lime', alpha=0.3, label='Trace Extraction Region')
        # ax.fill_between([0, self.imsize_x], 0, y_center-y_center*extract_percent/100,
        #                 color='skyblue', alpha=0.3, label='Sky Background Region')
        # ax.fill_between([0, self.imsize_x], y_center+y_center*extract_percent/100, 2*y_center,
        #                 color='skyblue', alpha=0.3)

        # -------------------------
        # Draw your regions
        # -------------------------
        ny, nx = subim.shape
        y_center = ny / 2.0
        x = np.arange(nx)

        extract_profile = np.array([self._column_extract_percent(xx) for xx in x])
        half_height = y_center * extract_profile / 100.0

        y_low = y_center - half_height
        y_high = y_center + half_height

        ax.fill_between(
            x, y_low, y_high,
            color='lime', alpha=0.3, label='Trace Extraction Region'
        )
        ax.fill_between(
            x, 0, y_low,
            color='skyblue', alpha=0.3, label='Sky Background Region'
        )
        ax.fill_between(
            x, y_high, ny,
            color='skyblue', alpha=0.3
        )

        # -------------------------
        # Overlay a compact stats box (robust saturation “distribution sense”)
        # -------------------------
        # If you want a single "headline" metric, p99.9 is usually the best.
        warn = "⚠️ Oversaturated ⚠️" if (p999 >= sat_level or frac_sat > 0.05) else ""
        stats_txt = (
            f"Subimage Statistics: {warn}\n"
            f"ADU max count: {adu_max:,.0f}  (99.9%: {p999:,.0f}, 99.999%: {p99999:,.0f})\n"
            f"Counts >={sat_level:,}: {n_sat:,} px ({frac_sat:.2f}%)   "
            f"Counts >={fullwell:,}: {n_full:,} px ({frac_full:.2f}%)"
        )
        
        box_color = "red" if warn else "black"
        face_color = "#460505" if warn else "white"

        at = AnchoredText(
            stats_txt,
            loc="lower left",
            prop=dict(size=11, color=box_color),
            frameon=True
        )
        at.patch.set_facecolor(face_color)
        at.patch.set_edgecolor(box_color)
        ax.add_artist(at)

        ax.legend(loc='upper right')

        ax.set_ylim(0, subim.shape[0])
        ax.set_aspect(25, adjustable='box')

        plt.show()

        if warn == "⚠️ Oversaturated ⚠️":
            raise ValueError(f"Image flagged for oversaturation: {stats_txt}")
        
        return fig
    

    def __calc_channel_signal(self, xpixel):
        """Calculates total counts in specified spectral channel xpixel
        by subtracting background and summing. The spectral signal is assumed
        to be centered vertically in the subimage, but the extraction width
        may vary with x if curved_box=True.
        """

        subim = self.subim
        yvals = subim[:, xpixel]
        yindex = np.arange(len(yvals))

        # column-dependent extraction width
        extract_percent_col = self._column_extract_percent(xpixel)

        percentile = float(len(yindex) / 100.0)

        extbot = 50 - extract_percent_col / 2.0
        exttop = 50 + extract_percent_col / 2.0

        n1 = int(extbot * percentile)
        n2 = int(exttop * percentile)

        # guard against pathological bounds
        n1 = max(0, min(n1, len(yindex) - 1))
        n2 = max(n1 + 1, min(n2, len(yindex)))

        x1 = yindex[0:n1]
        x2 = yindex[n2:]
        y1 = yvals[0:n1]
        y2 = yvals[n2:]

        X = np.concatenate((x1, x2), axis=0)
        Y = np.concatenate((y1, y2), axis=0)

        c = np.polyfit(X, Y, 1)
        p = np.poly1d(c)
        base = p(yindex)

        signal = yvals - base
        signal_max = np.max(signal)
        ymax = np.argmax(signal)

        # sum only inside extraction region, not the whole column
        tot_signal = np.sum(signal[n1:n2])

        skyave = np.mean(base)
        return (ymax, tot_signal, signal_max, skyave)
    
    def plot_spectrum(self, xaxis = 'pixel',yaxis = 'uncal', title='', \
        plot_balmer = False, medavg = 1,grid=True, show = True):

        fig, ax = plt.subplots(1,1,figsize=(16, 6))

        if xaxis == 'pixel':
            x = self.spectrum[0] ; ax.set_xlabel('Pixels')
        if xaxis == 'wavelength':
            x = self.waves ; ax.set_xlabel('Wavelength (Å)')
            xmin, xmax = self.wavelength_range
            ax.set_xlim(xmin,xmax)
            mask = (x > xmin) & (x < xmax)
        if yaxis == 'uncal':
            fig.suptitle(f'{self.object_name} Uncalibrated Spectrum {self.obs_date} ({self.exp_time} s)')
            y = self.spectrum[1] ; ax.set_ylabel('Uncalibrated flux (counts)')
        if yaxis == 'cal':
            fig.suptitle(f'{self.object_name} Calibrated Spectrum ({self.exp_time} s)', fontsize = 16)
            ax.set_title(f'{self.obs_date} [Camera = {self.hdr["INSTRUME"]}]', fontsize = 10)
            y = self.cal_spec   ; ax.set_ylabel(r'Flux  (erg cm$^{-2}$ s$^{-1}$ Angstrom$^{-1}$)')
            ax.set_ylim(np.min(y[mask]) - np.std(y[mask]),np.max(y[mask])*1.2)

        if plot_balmer:
            colors = {"Hε": "purple", "Hδ": "violet",  "Hγ": "blue", "Hβ": "cyan", "Hα": "red"}
            if self.filter == 'lrg':
                for name, wave in self.balmer.items():
                    ax.axvline(
                        x=wave,
                        linestyle='dotted',
                        color=colors.get(name, 'white'),
                        label=f'{name} ({wave} Å)'
                    )

            if self.filter == 'hrg':
                wave = self.balmer["Hα"]
                ax.axvline(
                    x=wave,
                    linestyle='dotted',
                    color=colors["Hα"],
                    label=f"Hα ({wave} Å)"
                )
                ax.axvline(
                    x = 6872.0, 
                    label='Red Telluric Line (6872.0 Å)', 
                    color='darkred', 
                    linestyle='--'
                )
                ax.axvline(
                    x = 6279.0, 
                    label='Red Telluric Line (6279.0 Å)', 
                    color='blue', 
                    linestyle='--'
                )


        # Median average if requested
        y = medfilt(y,kernel_size = medavg)
        ax.plot(x,y,'k-')

        if grid:
            ax.grid()

        ax.legend()

        if show:
            plt.show()

        return fig

    def extract_spectrum(
        self,
        extract_percent=None,
        norm=False,
        show_box=True,
        plot=True,
        curved_box=False,
        center_extract_percent=None,
        curve_power=None
        ):

        subim = self.subim
        
        self.curved_box = curved_box
        self.center_extract_percent = center_extract_percent

        if self.filter == 'hrg':
            rec_curve_power = 1.5
            rec_extract_percent = 90
        if self.filter == 'lrg':
            rec_curve_power = 0.0
            rec_extract_percent = 80

        self.curve_power = curve_power if curve_power else rec_curve_power
        self.extract_percent = extract_percent if extract_percent else rec_extract_percent

        pixels = np.arange(subim.shape[1])

        uncal_amp = []
        for pixel in pixels:
            ymax, signal, signal_max, skyave = self.__calc_channel_signal(pixel)
            uncal_amp.append(signal)

        uncal_amp = np.array(uncal_amp)

        if norm:
            uncal_amp /= np.max(uncal_amp)

        spectrum = np.vstack([pixels, uncal_amp])
        self.spectrum = spectrum

        if show_box:
            self.plot_box(cmap='viridis')
        if plot:
            self.plot_spectrum()

        # -------------------------
        # Robust ADU / saturation stats (no histogram)
        # -------------------------
        data = np.asarray(subim)

        # ignore NaNs if any
        finite = np.isfinite(data)
        if np.any(finite):
            d = data[finite].astype(float)
        else:
            d = data.ravel().astype(float)

        sat_level = 50000
        fullwell = 65500

        adu_max = np.max(d)
        p99   = np.percentile(d, 99.0)
        p999  = np.percentile(d, 99.9)
        p9999 = np.percentile(d, 99.99)
        p99999 = np.percentile(d, 99.999)

        n_pix = d.size
        n_sat = int(np.sum(d >= sat_level))
        frac_sat = 100.0 * n_sat / max(n_pix, 1)

        n_full = int(np.sum(d >= fullwell))
        frac_full = 100.0 * n_full / max(n_pix, 1)

        warn = "⚠️ Oversaturated ⚠️" if (p999 >= sat_level or frac_sat > 0.05) else ""
        stats_txt = (
            f"Subimage Statistics: {warn}\n"
            f"ADU max count: {adu_max:,.0f}  (99.9%: {p999:,.0f}, 99.999%: {p99999:,.0f})\n"
            f"Counts >={sat_level:,}: {n_sat:,} px ({frac_sat:.2f}%)   "
            f"Counts >={fullwell:,}: {n_full:,} px ({frac_full:.2f}%)"
        )
        
        if warn == "⚠️ Oversaturated ⚠️":
            raise ValueError(f"Image flagged for oversaturation: {stats_txt}")

        return spectrum
    
    def _column_extract_percent(self, xpixel):
        """
        Return extraction percent for a given column.
        If curved_box=False, this is constant.
        If curved_box=True, it is smallest at the center and largest at the edges.
        """
        edge_percent = float(self.extract_percent)

        curved_box = getattr(self, "curved_box", False)
        if not curved_box:
            return edge_percent

        center_percent = getattr(self, "center_extract_percent", None)
        if center_percent is None:
            center_percent = 0.4 * edge_percent

        power = float(getattr(self, "curve_power", 2))

        ncols = self.subim.shape[1]
        xmid = 0.5 * (ncols - 1)

        # normalized distance from center: 0 at center, 1 at edges
        if xmid == 0:
            t = 0.0
        else:
            t = np.abs(xpixel - xmid) / xmid

        # smooth curve: minimum at center, maximum at edges
        frac = t**power

        extract_percent = center_percent + (edge_percent - center_percent) * frac
        return np.clip(extract_percent, 1, 100)
    
    def fit_telluric(self, x_guess = None, recursion = None, offset = None, plot = False, debugging = False, manual_override = False):

        x = self.spectrum[0]
        y = self.spectrum[1]

        if manual_override:

            self.telluric_pixel = x_guess

            if self.filter == 'hrg':
                mask = (self.spectrum[0] >= (self.telluric_pixel - 2500)) & (self.spectrum[0] <= 4500)
                self.pixel_mask = self.spectrum[0][mask] - (self.telluric_pixel - 2500)
                self.amp_uncal_mask = self.spectrum[1][mask]
            if self.filter == 'lrg':
                mask = (self.spectrum[0] >= (self.telluric_pixel - 1550)) & (self.spectrum[0] <= 4500)
                self.pixel_mask = self.spectrum[0][mask] - (self.telluric_pixel - 1550)
                self.amp_uncal_mask = self.spectrum[1][mask]

            return self.telluric_pixel


        if recursion is None:
            recursion = 0

        if self.camera == 'ASI Camer':
            if self.filter == 'hrg':
                x_guess = x_guess if x_guess else 3080
                amp1, amp2 = -0.1, -0.1
                mask = (x > x_guess - 150) & (x < x_guess + 150)
            if self.filter == 'lrg':
                x_guess = x_guess if x_guess else 3000
                amp1, amp2 = -0.1, -0.1
                mask = (x > x_guess - 80) & (x < x_guess + 80)
        if self.camera == 'QHYCCD-Ca':
            if self.filter == 'hrg':
                x_guess = x_guess if x_guess else 2970
                amp1, amp2 = -0.1, -0.1
                mask = (x > x_guess - 150) & (x < x_guess + 150)
            if self.filter == 'lrg':
                x_guess = x_guess if x_guess else 2955
                amp1, amp2 = -0.1, -0.1
                mask = (x > x_guess - 60) & (x < x_guess + 60)

        x = x[mask]
        y = y[mask]/np.max(y[mask])

        if offset is None:
            offset = 0
        if recursion == 10:
            # print('Did not detect feature leftward of initial guess')
            offset = 0
        if recursion > 20:
            print("Recursion limit reached, unable to fit feature")
            return

        # Define the left Gaussian (g1) model
        gauss1 = GaussianModel(prefix='g1_')
        pars = gauss1.make_params(center=x_guess + offset, sigma=5)
        pars['g1_amplitude'].set(value=amp1, vary=True, max=0)

        # Define the right Gaussian (g2) model
        gauss2 = GaussianModel(prefix='g2_')
        pars.update(gauss2.make_params(center=x_guess + 30 + offset, sigma=5))
        pars['g2_amplitude'].set(value=amp2, vary=True, max=0)

        # Define the background model
        bkg = LinearModel(prefix='bkg_')
        pars.update(bkg.guess(y, x))

        # Combine the models
        mod = gauss1 + gauss2 + bkg
        init = mod.eval(pars, x=x)
        out = mod.fit(y, pars, x=x)

        # FWHM, fit components, and centroids for each Gaussian
        fwhms = [2 * out.params['g1_sigma'] * np.sqrt(2 * np.log(2)), 2 * out.params['g2_sigma'] * np.sqrt(2 * np.log(2))]
        centroids = [out.params['g1_center'].value, out.params['g2_center'].value]
        amplitudes = [out.params['g1_height'].value, out.params['g2_height'].value]
        comps = out.eval_components(x=x)
        if centroids[0] < centroids[1]:
            fwhm1 = fwhms[0]
            fwhm2 = fwhms[1]
            g1_fit = comps['g1_'] + comps['bkg_']
            g2_fit = comps['g2_'] + comps['bkg_']
            centroid1 = centroids[0]
            centroid2 = centroids[1]
            amp1 = amplitudes[0]
            amp2 = amplitudes[1]
        else:
            fwhm1 = fwhms[1]
            fwhm2 = fwhms[0]
            g1_fit = comps['g2_'] + comps['bkg_']
            g2_fit = comps['g1_'] + comps['bkg_']
            centroid1 = centroids[1]
            centroid2 = centroids[0]
            amp1 = amplitudes[1]
            amp2 = amplitudes[0]

        idx1 = np.argmin(np.abs(x - centroid1))
        idx2 = np.argmin(np.abs(x - centroid2))

        diff = np.abs(out.best_fit[idx1] - y[idx1])
        diff2 = np.abs(centroid2 - centroid1)
        diff3 = out.best_fit[idx2] - out.best_fit[idx1]

        if debugging:

            print('Fit Parameters:')
            print(f'  Left Gaussian FWHM: {fwhm1:.2f} pixels')
            print(f'  Left Gaussian Centroid: x = {centroid1:.2f}')
            print(f' Left Gaussian Amplitude: {amp1:.2f}')
            print(f' Right Gaussian Amplitude: {amp2:.2f}')
            print(f'  Right Gaussian FWHM: {fwhm2:.2f} pixels')
            print(f'  Right Gaussian Centroid: x = {centroid2:.2f}')
            print('')
            print(f'  Left Centroid Fit Value - Left Centroid Data Value [y]: {diff:.4f}')
            print(f'  Centroid Difference [x]: {diff2:.2f}')
            print(f'  Centroid Height Difference (right - left) [y]: {diff3:.4f}')

            # Plotting for debugging
            fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))

            # Left Plot: Data with Fitted Curve
            axes[0].plot(x, y, label="Data", alpha=0.8)
            axes[0].plot(x, out.best_fit, '-', label='Best Fit', color='black')
            axes[0].legend()
            axes[0].set_title("Data with Best Fit")

            # Right Plot: Data with Individual Components Overlayed
            axes[1].plot(x, y, label="Data", alpha=0.5)
            axes[1].plot(x, g1_fit, '--', label='Gaussian 1 + Background', color='green')
            axes[1].plot(x, g2_fit, '--', label='Gaussian 2 + Background', color='blue')
            axes[1].plot(x, comps['bkg_'], '--', label='Background', color='orange')
            axes[1].legend()
            axes[1].set_title("Data with Component Fits Overlayed")

            plt.tight_layout()
            plt.show()

        # diff = np.abs(out.best_fit[int(centroid1 - x[0])] - y[int(centroid1 - x[0])])
        # diff2 = np.abs(centroid2 - centroid1)
        # diff3 = out.best_fit[int(centroid2 - x[0])] - out.best_fit[int(centroid1 - x[0])]

        if self.filter == 'hrg':

            if (centroid1 - x[0]) < 0 or (centroid2 - x[0]) < 0 or (centroid1 - x[0]) > len(x) or (centroid2 - x[0]) > len(x):
                if recursion < 10:
                    # print('')
                    # print(f"Poor fit (centroid out of bounds), refitting at {x_guess + offset - 10}")
                    return self.fit_telluric(x_guess=x_guess, recursion = recursion + 1, offset = offset - 5, plot = plot, debugging = debugging)
                if recursion >= 10:
                    # print('')
                    # print(f"Poor fit (centroid out of bounds), refitting at {x_guess + offset + 10}")
                    return self.fit_telluric(x_guess=x_guess, recursion = recursion + 1, offset = offset + 5, plot = plot, debugging = debugging)

            if fwhm1 < 10 or fwhm1 > 50 or diff > 0.2 or diff2 > 50 or diff3 < 0 or fwhm2 < 50 or fwhm2 > 500:
                if recursion < 10:
                    # print('')
                    # print(f"Poor fit parameters, refitting at {x_guess + offset - 10}")
                    return self.fit_telluric(x_guess=x_guess, recursion = recursion + 1, offset = offset - 5, plot = plot, debugging = debugging)
                if recursion >= 10:
                    # print('')
                    # print(f"Poor fit parameters, refitting at {x_guess + offset + 10}")
                    return self.fit_telluric(x_guess=x_guess, recursion = recursion + 1, offset = offset + 5, plot = plot, debugging = debugging)
            
            if np.abs(amp1) < 0.02 or np.abs(amp2) < 0.02 or np.abs(amp1) > 1 or np.abs(amp2) > 1:
                if recursion < 10:
                    # print('')
                    # print(f"Poor fit amplitudes, refitting at {x_guess + offset - 10}")
                    return self.fit_telluric(x_guess=x_guess, recursion = recursion + 1, offset = offset - 5, plot = plot, debugging = debugging)
                if recursion >= 10:
                    # print('')
                    # print(f"Poor fit amplitudes, refitting at {x_guess + offset + 10}")
                    return self.fit_telluric(x_guess=x_guess, recursion = recursion + 1, offset = offset + 5, plot = plot, debugging = debugging)
            
        if self.filter == 'lrg':
            
            if (centroid1 - x[0]) < 0 or (centroid2 - x[0]) < 0 or (centroid1 - x[0]) > len(x) or (centroid2 - x[0]) > len(x):
                if recursion < 10:
                    # print('')
                    # print(f"Poor fit (centroid out of bounds), refitting at {x_guess + offset - 10}")
                    return self.fit_telluric(x_guess=x_guess, recursion = recursion + 1, offset = offset - 5, plot = plot, debugging = debugging)
                if recursion >= 10:
                    # print('')
                    # print(f"Poor fit (centroid out of bounds), refitting at {x_guess + offset + 10}")
                    return self.fit_telluric(x_guess=x_guess, recursion = recursion + 1, offset = offset + 5, plot = plot, debugging = debugging)

            if fwhm1 < 4 or fwhm1 > 20 or diff > 0.02 or diff2 > 20 or diff3 < -0.04 or fwhm2 < 1 or fwhm2 > 50:
                if recursion < 10:
                    # print('')
                    # print(f"Poor fit parameters, refitting at {x_guess + offset - 10}")
                    return self.fit_telluric(x_guess=x_guess, recursion = recursion + 1, offset = offset - 5, plot = plot, debugging = debugging)
                if recursion >= 10:
                    # print('')
                    # print(f"Poor fit parameters, refitting at {x_guess + offset + 10}")
                    return self.fit_telluric(x_guess=x_guess, recursion = recursion + 1, offset = offset + 5, plot = plot, debugging = debugging)
            
            if np.abs(amp1) < 0.03 or np.abs(amp2) < 0.03 or np.abs(amp1) > 1 or np.abs(amp2) > 1:
                if recursion < 10:
                    # print('')
                    # print(f"Poor fit amplitudes, refitting at {x_guess + offset - 10}")
                    return self.fit_telluric(x_guess=x_guess, recursion = recursion + 1, offset = offset - 5, plot = plot, debugging = debugging)
                if recursion >= 10:
                    # print('')
                    # print(f"Poor fit amplitudes, refitting at {x_guess + offset + 10}")
                    return self.fit_telluric(x_guess=x_guess, recursion = recursion + 1, offset = offset + 5, plot = plot, debugging = debugging)

        if plot:

            # Plotting
            fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))

            # Left Plot: Data with Fitted Curve
            axes[0].plot(x, y, label="Data", alpha=0.8)
            axes[0].plot(x, out.best_fit, '-', label='Best Fit', color='black')
            axes[0].legend()
            axes[0].set_title("Data with Best Fit")

            # Right Plot: Data with Individual Components Overlayed
            axes[1].plot(x, y, label="Data", alpha=0.5)
            axes[1].plot(x, g1_fit, '--', label=f'Gaussian 1 + Bkg (Centroid: x = {out.params['g2_center'].value:.2f})', color='green')
            axes[1].plot(x, g2_fit, '--', label=f'Gaussian 2 + Bkg (Centroid: x = {out.params['g1_center'].value:.2f})', color='blue')
            axes[1].plot(x, comps['bkg_'], '--', label='Background', color='orange')
            axes[1].legend()
            axes[1].set_title("Data with Component Fits Overlayed")

            plt.tight_layout()
            plt.show()

        # Return the previous outputs (wave_fit, fwhm1, fwhm2) along with r_squared_left
        wave_fit = min(out.params['g1_center'].value, out.params['g2_center'].value)
        self.telluric_pixel = wave_fit

        if self.filter == 'hrg':
            mask = (self.spectrum[0] >= (self.telluric_pixel - 2500)) & (self.spectrum[0] <= 4500)
            self.pixel_mask = self.spectrum[0][mask] - (self.telluric_pixel - 2500)
            self.amp_uncal_mask = self.spectrum[1][mask]
        if self.filter == 'lrg':
            mask = (self.spectrum[0] >= (self.telluric_pixel - 1550)) & (self.spectrum[0] <= 4500)
            self.pixel_mask = self.spectrum[0][mask] - (self.telluric_pixel - 1550)
            self.amp_uncal_mask = self.spectrum[1][mask]

        return wave_fit
    

    def wavelength_calibrate(self, plot = False):

        filter = self.filter
        pixel_mask = self.pixel_mask
        wave_coeffs = self.wave_calib
        amp_uncal_mask = self.amp_uncal_mask

        a, b, c, d = wave_coeffs[0], wave_coeffs[1], wave_coeffs[2], wave_coeffs[3]
        waves = a*pixel_mask**3 + b*pixel_mask**2 + c*pixel_mask + d

        if plot:
            plt.figure(figsize = (12,6))

            plt.plot(waves, amp_uncal_mask, color = 'black')
            plt.xlabel('Wavelength (Å)')
            plt.ylabel('Uncalibrated Flux (counts)')
            plt.title('Wavelength Calibrated Spectrum')

            plt.tight_layout()
            plt.show()
        
        self.waves = waves

        return waves
    
    def gain_calibrate(self, plot = False):

        waves = self.waves
        amp_uncal_mask = self.amp_uncal_mask
        gain_spline = self.gain_spline
        gain_curve = gain_spline(waves)

        cal_spec = amp_uncal_mask/gain_curve

        xlow, xhigh = self.wavelength_range
        mask = (waves >= xlow) & (waves <= xhigh)

        norm_gain_curve = gain_curve/np.max(gain_curve[mask])
        norm_cal_spec = cal_spec/np.max(cal_spec[mask])

        if plot:
            plt.figure(figsize = (12,6))

            plt.plot(waves[mask], amp_uncal_mask[mask]/np.max(amp_uncal_mask[mask]), color = 'blue', label = 'Uncalibrated Spectrum')
            plt.plot(waves[mask], norm_gain_curve[mask], color = 'red', label = 'Gain Curve')
            plt.plot(waves[mask], norm_cal_spec[mask], color = 'purple', label = 'Calibrated Spectrum')
            plt.xlabel('Wavelength (Å)')
            plt.ylabel('Normalized Flux')
            plt.title('Flux Calibrated Spectrum')
            
            # plt.xlim(xlow, xhigh)
            plt.ylim(np.min(norm_cal_spec[mask]) - np.std(norm_cal_spec[mask]), np.max(norm_gain_curve[mask]) + np.std(norm_gain_curve[mask]))

            plt.tight_layout()
            plt.legend()
            plt.show()

        self.cal_spec = cal_spec

        return self.cal_spec


    ######################
    # Deriving Calibration
    ######################

    def load_stelib_spectrum(self, folder_path):
        star_name = self.object_name.replace(" ", "").lower()

        for file in os.listdir(folder_path):
            file_lower = file.lower()
            if (star_name in file_lower) and file_lower.endswith(".csv"):
                file_path = os.path.join(folder_path, file)

                df = pd.read_csv(file_path)

                # Try named columns first
                cols_norm = {c.strip().lower(): c for c in df.columns}
                if "wavelength" in cols_norm and "flux" in cols_norm:
                    wcol = cols_norm["wavelength"]
                    fcol = cols_norm["flux"]
                    self.ref_wave = df[wcol].to_numpy()
                    self.ref_flux = df[fcol].to_numpy()
                    return self.ref_wave, self.ref_flux

                # Otherwise fall back to first two columns by position
                if df.shape[1] < 2:
                    raise ValueError(f"{file_path} has <2 columns: {list(df.columns)}")

                self.ref_wave = df.iloc[:, 0].to_numpy()
                self.ref_flux = df.iloc[:, 1].to_numpy()
                return self.ref_wave, self.ref_flux

        raise FileNotFoundError(f"No file found for star: {star_name}")
    
    def derive_wavelength_correction(self):

        star_name = self.object_name
        date_obs = self.hdr['DATE-OBS']
        c = 299792.458 * u.km/u.s

        # Reset to default fields
        Simbad.reset_votable_fields()
        Simbad.add_votable_fields('rvz_radvel')
        result_table = Simbad.query_object(star_name)
        rv = result_table['rvz_radvel'][0]

        self.target_rv = rv * u.km/u.s

        target_star = SkyCoord.from_name(star_name)
        obs_loc = EarthLocation.of_site('Winer')
        t = Time(date_obs, format='isot', scale='utc', location=obs_loc)
        helio_vel = target_star.radial_velocity_correction(obstime=t).to('km/s')

        self.helio_vel = helio_vel

        wavelength_correction = 1 + (self.helio_vel + self.target_rv) / c

        self.wavelength_correction = wavelength_correction

        return self.wavelength_correction

    def wavelength_dependent_gaussian_convolution(self, R = None):
        """
        Convolve a spectrum to constant resolving power R
        using a wavelength-dependent Gaussian kernel.

        Parameters
        ----------
        wave : ndarray
            Wavelength array (Å), monotonic
        flux : ndarray
            Flux array
        R : float
            Target resolving power

        Returns
        -------
        flux_conv : ndarray
            Resolution-matched spectrum
        """

        R = R if R else self.R

        wave = np.asarray(self.ref_wave)
        flux = np.asarray(self.ref_flux)

        # Local dispersion (Å / pix)
        dw = np.gradient(wave)

        flux_conv = np.zeros_like(flux)

        for i in range(len(wave)):
            fwhm_lambda = wave[i] / R
            sigma_lambda = fwhm_lambda / 2.355
            sigma_pix = sigma_lambda / dw[i]

            # Limit kernel size for stability
            sigma_pix = np.clip(sigma_pix, 0.5, 50)

            # Apply local Gaussian
            kernel_radius = int(4 * sigma_pix)
            lo = max(0, i - kernel_radius)
            hi = min(len(flux), i + kernel_radius + 1)

            x = np.arange(lo, hi)
            g = np.exp(-0.5 * ((x - i) / sigma_pix) ** 2)
            g /= g.sum()

            flux_conv[i] = np.sum(flux[lo:hi] * g)

        return flux_conv
    
    # def voigt_centroid(self, x, y, x0, window):
    #     """
    #     Fit a Voigt profile near x0 and return the centroid.

    #     Parameters
    #     ----------
    #     x : array
    #         Pixel or wavelength array
    #     y : array
    #         Flux array (absorption line)
    #     x0 : float
    #         Initial guess (pixel or wavelength)
    #     window : int
    #         Half-width of fit window in x units

    #     Returns
    #     -------
    #     center : float
    #         Fitted centroid
    #     center_err : float or None
    #         1σ uncertainty (if available)
    #     result : lmfit result object
    #     """

    #     idx = np.argmin(np.abs(x - x0))
    #     lo = max(idx - window, 0)
    #     hi = min(idx + window + 1, len(x))

    #     xfit = x[lo:hi]
    #     yfit = y[lo:hi]

    #     # Absorption line → invert for lmfit peak models
    #     yfit_inv = np.max(yfit) - yfit

    #     voigt = VoigtModel(prefix='v_')
    #     cont = LinearModel(prefix='c_')
    #     model = voigt + cont

    #     params = model.make_params()

    #     params['v_center'].set(value=x0, min=xfit.min(), max=xfit.max())
    #     params['v_sigma'].set(value=1.0, min=0.1)
    #     params['v_gamma'].set(value=1.0, min=0.1)
    #     params['v_amplitude'].set(value=np.trapz(yfit_inv, xfit), min=0)

    #     # params['c_c'].set(value=np.median(yfit_inv))
    #     params.update(cont.guess(yfit_inv, x=xfit))

    #     result = model.fit(yfit_inv, params, x=xfit)

    #     center = result.params['v_center'].value
    #     center_err = result.params['v_center'].stderr

    #     return center, center_err

    def voigt_centroid(
        self,
        x,
        y,
        x0,
        window=6,
        window_try=None,
        min_snr=3.0,
        max_center_shift=None,
        return_result=False,
        debug=False
    ):
        """
        Robust centroid of an absorption feature using Voigt + linear continuum.

        Parameters
        ----------
        x, y : array-like
            Spectrum coordinates and flux.
        x0 : float
            Initial guess for centroid.
        window : int
            Default half-window in x units if window_try is None.
        window_try : list[int] or None
            Candidate half-window sizes to try. Best acceptable fit is chosen.
        min_snr : float
            Minimum approximate line SNR required for accepting a Voigt fit.
        max_center_shift : float or None
            Maximum allowed |center - x0|. If None, defaults to ~0.75*window.
        return_result : bool
            If True, also return best lmfit result and diagnostics dict.
        debug : bool
            Print diagnostics.

        Returns
        -------
        center : float
        center_err : float
        result : lmfit result, optional
        info : dict, optional
        """
        from lmfit.models import VoigtModel, LinearModel
        import numpy as np

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if window_try is None:
            window_try = sorted(set([max(3, window-2), window, window+2, window+4]))

        if max_center_shift is None:
            max_center_shift = 0.75 * max(window_try)

        best = None
        best_score = np.inf
        diagnostics = []

        for w in window_try:
            idx = np.argmin(np.abs(x - x0))
            lo = max(idx - w, 0)
            hi = min(idx + w + 1, len(x))

            xfit = x[lo:hi]
            yfit = y[lo:hi]

            if len(xfit) < 7:
                continue
            if not np.all(np.isfinite(xfit)) or not np.all(np.isfinite(yfit)):
                continue

            dx = np.median(np.diff(xfit)) if len(xfit) > 2 else 1.0
            if not np.isfinite(dx) or dx <= 0:
                dx = 1.0

            # approximate line depth / noise using edge pixels as pseudo-continuum
            nedge = max(2, len(yfit) // 5)
            edge_y = np.concatenate([yfit[:nedge], yfit[-nedge:]])
            cont_med = np.median(edge_y)
            noise = np.std(edge_y) if len(edge_y) > 1 else np.std(yfit)
            noise = max(noise, 1e-12)

            line_depth = cont_med - np.min(yfit)   # absorption depth
            approx_snr = line_depth / noise

            # absorption -> positive peak for Voigt
            yinv = np.max(yfit) - yfit

            voigt = VoigtModel(prefix='v_')
            cont = LinearModel(prefix='c_')
            model = voigt + cont

            pars = model.make_params()
            pars['v_center'].set(value=x0, min=xfit.min(), max=xfit.max())
            pars['v_sigma'].set(value=max(1.5 * dx, 0.5 * dx), min=0.25 * dx, max=10 * dx)
            pars['v_gamma'].set(value=max(1.5 * dx, 0.5 * dx), min=0.25 * dx, max=10 * dx)
            pars['v_amplitude'].set(value=max(np.trapz(yinv, xfit), 1e-12), min=0)

            # let continuum be guessed from inverted spectrum
            pars.update(cont.guess(yinv, x=xfit))

            try:
                result = model.fit(
                    yinv,
                    pars,
                    x=xfit,
                    method='least_squares',
                    fit_kws={'loss': 'soft_l1'}
                )
            except Exception:
                continue

            center = result.params['v_center'].value
            center_err = result.params['v_center'].stderr
            sigma = result.params['v_sigma'].value
            gamma = result.params['v_gamma'].value

            best_fit = result.best_fit
            resid = yinv - best_fit
            rms = np.sqrt(np.mean(resid**2))

            # approximate reduced chi-like score
            red_metric = np.mean((resid / noise)**2)

            # fit acceptance
            bad = False
            if not np.isfinite(center):
                bad = True
            if abs(center - x0) > max_center_shift:
                bad = True
            if not np.isfinite(sigma) or not np.isfinite(gamma):
                bad = True
            if sigma < 0.2 * dx or gamma < 0.2 * dx:
                bad = True
            if approx_snr < min_snr:
                bad = True

            diagnostics.append({
                "window": w,
                "center": center,
                "center_err": center_err,
                "snr": approx_snr,
                "rms": rms,
                "red_metric": red_metric,
                "accepted": not bad,
            })

            if bad:
                continue

            # prefer low residual and small center error
            err_term = center_err if (center_err is not None and np.isfinite(center_err)) else 5 * dx
            score = red_metric + 0.2 * (err_term / dx)

            if score < best_score:
                best_score = score
                best = (center, err_term, result, {
                    "window": w,
                    "snr": approx_snr,
                    "rms": rms,
                    "red_metric": red_metric,
                })

        # fallback: weighted centroid around local absorption minimum
        if best is None:
            idx = np.argmin(np.abs(x - x0))
            lo = max(idx - window, 0)
            hi = min(idx + window + 1, len(x))

            xfit = x[lo:hi]
            yfit = y[lo:hi]

            if len(xfit) < 3:
                center = float(x0)
                center_err = np.inf
            else:
                nedge = max(2, len(yfit) // 5)
                edge_y = np.concatenate([yfit[:nedge], yfit[-nedge:]])
                cont = np.median(edge_y)
                weights = np.clip(cont - yfit, 0, None)  # absorption-only weights

                if np.sum(weights) > 0:
                    center = float(np.sum(xfit * weights) / np.sum(weights))
                    center_err = float(np.std(xfit) / max(np.sqrt(np.sum(weights > 0)), 1))
                else:
                    center = float(xfit[np.argmin(yfit)])
                    center_err = np.inf

            if debug:
                print(f"voigt_centroid fallback used at x0={x0:.2f}")

            if return_result:
                return center, center_err, None, {"fallback": True, "trials": diagnostics}
            return center, center_err

        center, center_err, result, info = best

        if debug:
            print(
                f"voigt_centroid x0={x0:.2f} -> center={center:.3f}, "
                f"err={center_err:.3f}, window={info['window']}, "
                f"SNR={info['snr']:.2f}, red_metric={info['red_metric']:.3f}"
            )

        if return_result:
            info["fallback"] = False
            info["trials"] = diagnostics
            return center, center_err, result, info

        return center, center_err
    
    def derive_wavelength_solution(self, show_points = False, plot = False):

        filter = self.filter
        cal_star = self.object_name
        telluric_pixel = self.telluric_pixel
        pixel_mask = self.pixel_mask
        amp_uncal_mask = self.amp_uncal_mask
        xlow, xhigh = self.wavelength_range
        wavelength_correction = self.wavelength_correction

        if filter == 'hrg':
            telluric_x = 2500
            if cal_star in ['hr 718','HR 718', 'hr 4468', 'HR 4468']:
                wavelengths = [6279.0, 6347.11, 6371.37, 6562.819, 6872.0]
                pixel_guesses = [telluric_x - 1292, telluric_x - 1136, telluric_x - 1081, telluric_x - 652, telluric_x]
                wavelength_labels = [r"O$_2$ $\gamma$ band", r"Si II 6347", r"Si II 6371", r"H$\alpha$", r"O$_2$ B band"]
            if cal_star in ['hr 3454', 'HR 3454']:
                wavelengths = [5875.62510, 6279.0, 6347.11, 6562.819, 6678.15174, 6872.0] 
                wavelength_labels = [r"He I 5876", r"O$_2$ $\gamma$ band", r"Si II 6347", r"H$\alpha$", r"He I 6678", r"O$_2$ B band"]
                pixel_guesses = [telluric_x - 2306, telluric_x - 1292, telluric_x - 1136, telluric_x - 652, telluric_x - 405, telluric_x]
            if cal_star in ['hr 4963', 'HR 4963']:
                wavelengths = [5889.95094, 6279.0, 6347.11, 6371.37, 6562.819, 6872.0] 
                wavelength_labels = [r"Na I 5890", r"O$_2$ $\gamma$ band", r"Si II 6347", r"Si II 6371", r"H$\alpha$", r"O$_2$ B band"]
                pixel_guesses = [telluric_x - 2269, telluric_x - 1292, telluric_x - 1136, telluric_x - 1081, telluric_x - 652, telluric_x]
            if cal_star in ['hr 7589', 'HR 7589']:
                wavelengths = [5875.62510, 6279.0, 6560.14160, 6678.15174, 6872.0] 
                wavelength_labels = [r"He I 5876", r"O$_2$ $\gamma$ band", r"He I 6678",r"He I 6560", r"O$_2$ B band"]
                pixel_guesses = [telluric_x - 2306, telluric_x - 1288, telluric_x - 657, telluric_x - 405, telluric_x]
        if filter == 'lrg':
            if self.camera == 'ASI Camer':
                telluric_x = 1550
                if cal_star in ['hr 718', 'HR 718','hr 3454', 'HR 3454', 'hr 4468', 'HR 4468','hr 4963', 'HR 4963']:
                    wavelengths = [3970.0788, 4101.7415, 4340.471, 4861.333, 6562.819, 6872.0, 7607.5]
                    wavelength_labels = [r"H$\epsilon$", r"H$\delta$", r"H$\gamma$", r"H$\beta$", r"H$\alpha$", r"O$_2$ B band", r"O$_2$ A band"]
                    pixel_guesses = [telluric_x - 1312, telluric_x - 1246, telluric_x - 1132, telluric_x - 887, telluric_x - 133, telluric_x, telluric_x + 316]
                    # wavelengths = [3970.0788, 4101.7415, 4861.333, 4921.931036, 6562.819, 6872.0, 7607.5]
                    # wavelength_labels = [r"H$\epsilon$", r"H$\delta$", r"H$\beta$", r"He I 4922", r"H$\alpha$", r"O$_2$ B band", r"O$_2$ A band"]
                    # pixel_guesses = [telluric_x - 1312, telluric_x - 1246, telluric_x - 887, telluric_x - 860, telluric_x - 133, telluric_x, telluric_x + 316]
            if self.camera == 'QHYCCD-Ca':
                telluric_x = 1550
                if cal_star in ['hr 718', 'HR 718','hr 3454', 'HR 3454', 'hr 4468', 'HR 4468','hr 4963', 'HR 4963']:
                    wavelengths = [3970.0788, 4101.7415, 4340.471, 4861.333, 6562.819, 6872.0, 7607.5]
                    wavelength_labels = [r"H$\epsilon$", r"H$\delta$", r"H$\gamma$", r"H$\beta$", r"H$\alpha$", r"O$_2$ B band", r"O$_2$ A band"]
                    pixel_guesses = [telluric_x - 1290, telluric_x - 1225, telluric_x - 1112, telluric_x - 872, telluric_x - 133, telluric_x, telluric_x + 316]
        
        wavelength_centroids = []
        wavelength_errors = []

        for wavelength in wavelengths:
            if wavelength not in [6279.0, 6872.0, 7607.5]:
                wavelength_centroids.append(wavelength*wavelength_correction)
                wavelength_errors.append(0.05)
            else:
                wavelength_centroids.append(wavelength)
                wavelength_errors.append(0.5)

        pixel_centroids = []
        pixel_errors = []

        for pixel_guess in pixel_guesses:
            if pixel_guess == telluric_x:
                pixel_centroids.append(pixel_guess)
                pixel_errors.append(0.5)
            else:
                # if pixel_guess == telluric_x - 652:
                #     center, err = self.voigt_centroid(x = pixel_mask, y = amp_uncal_mask, x0 = pixel_guess, window = 8)
                # else:
                center, err = self.voigt_centroid(x = pixel_mask, y = amp_uncal_mask, x0 = pixel_guess, window = 6)
                pixel_centroids.append(center)
                pixel_errors.append(err)

        if show_points:
            colors = ['violet', 'red', 'green', 'purple', 'magenta', 'blue', 'brown', 'black', 'orange', 'grey']

            plt.figure(figsize=(12, 8))
            plt.plot(pixel_mask, amp_uncal_mask, lw=1)
            plt.xlabel('Pixels', fontsize = 24)
            plt.ylabel('Flux (counts)', fontsize = 24)
            plt.title(f'Uncalibrated Spectrum', fontsize = 24)

            color_index = 0
            for pix_centroid, wave_centroid, wavelength_label in zip(pixel_centroids, wavelength_centroids, wavelength_labels):
                index = np.argmin(np.abs(pixel_mask-pix_centroid))
                plt.scatter(pixel_mask[index], amp_uncal_mask[index], color = colors[color_index], 
                            marker = '*', s=100, alpha = 0.4, label = fr"{wavelength_label}: x={pix_centroid:.1f} → λ={wave_centroid:.2f} Å")
                color_index += 1

            # plt.xlim(0,3000)
            plt.grid(True)
            plt.tight_layout()
            plt.legend(loc = 'upper right')
            plt.show()

        # Pixel --> Wavelength Solution Derivation
        degree = 3

        xdata = np.asarray(pixel_centroids)
        ydata = np.asarray(wavelength_centroids)

        sigma_x = np.asarray(pixel_errors)
        sigma_y = np.asarray(wavelength_errors)

        sigma_x = np.where(sigma_x == None, np.inf, sigma_x).astype(float)
        sigma_y = np.where(sigma_y == None, np.inf, sigma_y).astype(float)

        low = int(np.min(xdata))
        high = int(np.max(xdata))
        xplot = np.linspace(low, high, high - low + 1)

        model = PolynomialModel(degree=degree, prefix='p_')

        params = model.make_params()

        # Sensible initial guesses
        params['p_c0'].set(value=np.median(ydata))  # constant term

        for k in range(1, degree + 1):
            params[f'p_c{k}'].set(value=0.0)

        # Optional: weak regularization for higher orders
        if degree >= 2:
            params['p_c2'].set(value=1e-6)

        result0 = model.fit(ydata, params, x=xdata)

        # print("\n=== Fit Report ===")
        # report_fit(result0)

        # Polynomial derivative evaluated at xdata
        p = result0.params
        dfdx = sum(
            k * p[f'p_c{k}'].value * xdata**(k-1)
            for k in range(1, degree + 1)
        )

        sigma_eff = np.sqrt(
            sigma_y**2 + (dfdx * sigma_x)**2
        )

        line_labels = np.array(wavelength_labels, dtype=object)

        sigma_eff_mod = sigma_eff.copy()

        if self.filter == 'hrg':
            halpha_mask = np.array([
            ("H$\\alpha$" in str(lbl)) or ("Hα" in str(lbl))
            for lbl in line_labels
            ])
            O2_mask = np.array([
                ("O$_2$ $\gamma$ band" in str(lbl))
                for lbl in line_labels
            ])
            if np.any(halpha_mask):
                non_halpha = ~halpha_mask

                if np.any(non_halpha):
                    # current weights of all non-Hα points
                    non_halpha_weights = 1.0 / sigma_eff[non_halpha]

                    # next-highest weighted point among the non-Hα anchors
                    next_highest_weight = np.max(non_halpha_weights)

                    # force Hα to be 2x that weight  ->  half the uncertainty
                    target_halpha_weight = next_highest_weight
                    target_halpha_sigma = 1.0 / target_halpha_weight

                    target_O2_weight = target_halpha_weight * 0.85
                    target_O2_sigma = 1.0 / target_O2_weight

                    sigma_eff_mod[halpha_mask] = target_halpha_sigma
                    sigma_eff_mod[O2_mask] = target_O2_sigma

        if self.filter == 'lrg':
            hbeta_mask = np.array([
            ("H$\\beta$" in str(lbl)) or ("Hβ" in str(lbl))
            for lbl in line_labels
            ])
            hdelta_mask = np.array([
            ("H$\\delta$" in str(lbl)) or ("Hδ" in str(lbl))
            for lbl in line_labels
            ])
            halpha_mask = np.array([
            ("H$\\alpha$" in str(lbl)) or ("Hα" in str(lbl))
            for lbl in line_labels
            ])
            if np.any(hbeta_mask):
                non_hbeta = ~hbeta_mask

                if np.any(non_hbeta):
                    # current weights of all non-Hβ points
                    non_hbeta_weights = 1.0 / sigma_eff[non_hbeta]

                    # next-highest weighted point among the non-Hβ anchors
                    next_highest_weight = np.max(non_hbeta_weights)

                    # force Hβ to be 2x that weight  ->  half the uncertainty
                    target_hbeta_weight = next_highest_weight
                    target_hbeta_sigma = 1.0 / target_hbeta_weight

                    target_hdelta_weight = target_hbeta_weight * 0.8
                    target_hdelta_sigma = 1.0 / target_hdelta_weight

                    target_halpha_weight = target_hbeta_weight * 0.9
                    target_halpha_sigma = 1.0 / target_halpha_weight

                    sigma_eff_mod[hbeta_mask] = target_hbeta_sigma
                    sigma_eff_mod[hdelta_mask] = target_hdelta_sigma
                    sigma_eff_mod[halpha_mask] = target_halpha_sigma

        weights = 1.0 / sigma_eff_mod

        result = model.fit(
            ydata,
            result0.params,
            x=xdata,
            weights=weights,
            scale_covar=False,
            method='least_squares',
        )

        # print("\n=== Fit Report ===")
        # report_fit(result)

        yfit = result.eval(x=xplot)

        resid = ydata - result.eval(x=xdata)
        ss_res = np.sum(resid**2)
        ss_tot = np.sum((ydata - np.mean(ydata))**2)
        r2 = 1 - ss_res / ss_tot
        # print(f"\nR^2 = {r2:.10f}")

        cov = result.covar
        if cov is None:
            print("WARNING: covariance matrix unavailable — uncertainties cannot be computed.")
        else:
            # Build Jacobian dynamically
            J = np.vstack([
                xplot**k for k in range(degree + 1)
            ])  # shape: (degree+1, N)

            var_y = np.einsum('ij,jk,ik->i', J.T, cov, J.T)
            sigma_y = np.sqrt(np.maximum(var_y, 0))

        if plot:

            # weights used in the lmfit fit
            sigma_eff = np.asarray(sigma_eff, dtype=float)
            w = np.asarray(weights, dtype=float)

            # normalize for marker sizing
            w_norm = (w - np.min(w)) / (np.ptp(w) + 1e-12)  # 0..1
            sizes = 30 + 170 * w_norm  # tweak range to taste

            plt.figure(figsize=(11, 8))


            sc = plt.scatter(
                xdata, ydata,
                s=sizes, c=w, cmap='viridis', edgecolor='k', linewidth=0.4,
                label='Anchors (size/color ∝ weight)'
            )
            plt.colorbar(sc, label='Weight = 1/σ_eff')

            plt.plot(xplot, yfit, 'r', label=f'Weighted polynomial (deg={degree})')

            if cov is not None:
                plt.fill_between(xplot, yfit - sigma_y, yfit + sigma_y, alpha=0.25, label='1σ band')

            plt.xlabel('Pixel (x)')
            plt.ylabel('Wavelength (Å)')
            plt.title(f'Pixel → Wavelength Calibration for {cal_star} on {self.obs_date}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

        coeffs = [result.params[f"p_c{k}"].value for k in range(degree, -1, -1)]  # high→low

        self.wave_centroids = wavelength_centroids
        self.pix_centroids = pixel_centroids
        self.wave_r2 = r2
        self.wave_calib = coeffs
        self.pixel_mask = pixel_mask
        self.amp_uncal_mask = amp_uncal_mask

        return self.wave_calib
    
    def match_and_interpolate(self):
        """
        Safely interpolate both observed and reference spectra to a common grid.
        """
        # Ensure all arrays are 1D numpy arrays
        waves_ref = np.array(self.ref_wave).flatten()
        flux_ref = np.array(self.ref_flux).flatten()
        waves_data = np.array(self.waves).flatten()
        flux_data = np.array(self.amp_uncal_mask).flatten()

        flux_ref_conv = self.wavelength_dependent_gaussian_convolution(R = self.R/2.25 if self.filter == 'hrg' else self.R/1.33)

        xlow, xhigh = self.wavelength_range
        mask = (waves_ref >= xlow - 400) & (waves_ref <= xhigh + 400)
        waves_ref_mask = waves_ref[mask]
        flux_ref_mask = flux_ref_conv[mask]

        # Check that arrays match in size before interpolating
        if len(waves_data) != len(flux_data):
            raise ValueError(f"waves_data and flux_data lengths do not match: {len(waves_data)} vs {len(flux_data)}")
        if len(waves_ref_mask) != len(flux_ref_mask):
            raise ValueError(f"waves_ref and flux_ref lengths do not match: {len(waves_ref_mask)} vs {len(flux_ref_mask)}")

        # Determine overlapping wavelength range
        min_wave = max(np.min(waves_data), np.min(waves_ref_mask))
        max_wave = min(np.max(waves_data), np.max(waves_ref_mask))

        if min_wave >= max_wave:
            raise ValueError("No overlapping wavelength region between data and reference.")

        # Define common wavelength grid
        wave_grid = np.linspace(min_wave, max_wave, len(waves_data))

        # Interpolate both spectra onto the same grid
        flux_data_interp = np.interp(wave_grid, waves_data, flux_data)
        flux_ref_interp = np.interp(wave_grid, waves_ref_mask, flux_ref_mask)

        self.wave_grid = wave_grid
        self.flux_data_interp = flux_data_interp
        self.flux_ref_interp = flux_ref_interp

        return self.wave_grid, self.flux_data_interp, self.flux_ref_interp
    
    def derive_gain_calibration(self, plot = False):

        wave_grid = self.wave_grid
        flux_data_interp = self.flux_data_interp
        flux_ref_interp = self.flux_ref_interp
        xlow, xhigh = self.wavelength_range

        gain = flux_data_interp/flux_ref_interp

        if self.filter == 'hrg':
            knots = np.linspace(wave_grid.min()+200, wave_grid.max()-200, 45)
            if self.object_name in ['hr 718','HR 718']:
                exclude_regions = [(6460, 6500), (6555, 6580), (6605, 6640), (6855, 6990)]
            if self.object_name in ['hr 3454', 'HR 3454', 'hr 4963', 'HR 4963', 'hr 7589', 'HR 7589']:
                exclude_regions = [(6550, 6580), (6860, 6915)]
            if self.object_name in ['hr 4468', 'HR 4468']:
                exclude_regions = [(6460, 6500), (6555, 6580), (6855, 6990)]
            if self.object_name in ['hr 4963', 'HR 4963']:
                exclude_regions = [(6445, 6500), (6550, 6580), (6860, 6915)]
        if self.filter == 'lrg':
            knots = np.linspace(wave_grid.min() + 400, 7000, 105)
            if self.object_name in ['hr 718','HR 718']:
                exclude_regions = [(4310, 4360),(4835, 4905),(6460, 6500),(6525, 6595),(6860, 6950)]
            if self.object_name in ['hr 3454', 'HR 3454', 'hr 4468', 'HR 4468', 'hr 4963', 'HR 4963']:
               exclude_regions = [(4310, 4360), (4605, 4755), (4835, 4905), (6525, 6595), (6860, 6925)]

        mask = np.ones_like(wave_grid, dtype=bool)

        for lo, hi in exclude_regions:
            mask &= (wave_grid < lo) | (wave_grid > hi)

        for lo, hi in exclude_regions:
            knots = knots[(knots < lo) | (knots > hi)]

        spline = LSQUnivariateSpline(wave_grid[mask], gain[mask], t=knots, k=3)
        gain_smooth = spline(wave_grid)

        if plot:
            plt.figure(figsize = (12,10))
            plt.plot(wave_grid, gain, color = 'red', label = 'gain curve')
            plt.plot(wave_grid, gain_smooth, color = 'blue', alpha = 0.8, linewidth = 2, label = 'cubic spline fit')
            if self.filter == 'hrg':
                plt.xlim(xlow - 400,xhigh + 400)
            if self.filter == 'lrg':
                plt.xlim(xlow - 200,xhigh)
            # plt.ylim(0,2.5)
            labeled = False
            for lo, hi in exclude_regions:
                if not labeled:
                    plt.axvspan(lo, hi, color='gray', alpha=0.2, label='Masked region')
                    labeled = True
                else:
                    plt.axvspan(lo, hi, color='gray', alpha=0.2)
            plt.xlabel('Wavelength (Angstroms)')
            plt.ylabel('Flux (Arbitrary units)')
            plt.title(f'Gain Curve Cubic Spline Fit ({self.filter})')
            plt.legend()
            plt.show()

        self.gain = gain
        self.gain_spline = spline
        self.gain_smooth = gain_smooth

        return gain, spline, gain_smooth
