#!/usr/bin/env python3
import os, glob, warnings
from collections import defaultdict
import numpy as np
from tqdm import tqdm

from astropy.io import fits
from astropy.time import Time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# USER PATHS
# ---------------------------------------------------------
# datapath = os.path.expanduser('~/Desktop/Code/reduced_data/')
datapath = os.path.expanduser('~/gdrive/Shared drives/MACRO-Be/resources/Data/raw_data/')
outpath  = os.path.expanduser('~/Desktop/Code/')
outpath2 = os.path.expanduser('~/gdrive/Shared drives/MACRO-Be/resources/')

# ---------------------------------------------------------
# Helper: normalize star names
# ---------------------------------------------------------
def normalize_star_name(name):
    """
    Convert variants like:
        '69_Ori', '69ori', '69 ORI', '69_ORI_hrg' -> '69ori'
    """
    name = name.lower()
    for tok in ["_", "-", " "]:
        name = name.replace(tok, "")
    return name

# ---------------------------------------------------------
# Master alias table
# ---------------------------------------------------------
STAR_ALIASES = {
    "pzgem": "hd45314",
    "hd45314": "hd45314",

    "bngem": "hd60848",
    "hd60848": "hd60848",

    "zet_tau": "zeta_tau",
    "zeta_tau": "zeta_tau",

    "gamma_cas": "gamma_cas",
    "gam_cas":  "gamma_cas",

    "HD_17520": "BD+59_553",
    "BD+59_553": "BD+59_553",
}

# Normalize alias keys so lookup works
STAR_ALIASES = {normalize_star_name(k): v for k, v in STAR_ALIASES.items()}

# ---------------------------------------------------------
# 1. Collect FITS files
# ---------------------------------------------------------
patterns = [f"{datapath}/202*/*.fz"]

image_list = []
for pattern in patterns:
    image_list.extend(glob.glob(pattern))

print(f"\nFound {len(image_list)} FITS files.\n")

# ---------------------------------------------------------
# 2. Read timestamps & sort chronologically
# ---------------------------------------------------------
timelist = []
for f in tqdm(image_list, desc="Reading timestamps", unit="file"):
    try:
        hdu = fits.open(f)
        hd = hdu[1].header
        time = Time(hd["DATE-OBS"], format="isot", scale="utc").jd
        timelist.append(time)
    except Exception:
        timelist.append(np.inf)  # skip broken files

astlist = [x for _, x in sorted(zip(timelist, image_list))]

# Save the sorted list (optional)
astlist_outfile = os.path.join(outpath, f"Be.astlist_{os.path.basename(astlist[-1])[:10]}.txt")
with open(astlist_outfile, "w") as f:
    for item in astlist:
        f.write(item + "\n")

# ---------------------------------------------------------
# 3. Build observation count table
# ---------------------------------------------------------
observations = defaultdict(lambda: defaultdict(int))   # star → date → count

for filepath in astlist:
    fname = os.path.basename(filepath)

    if "_hrg_" not in fname.lower():
        continue

    parts = fname.split("_")
    if len(parts) < 5:
        continue

    # Find the 'hrg' index
    hrg_index = None
    for i, p in enumerate(parts):
        if p.lower() == "hrg":
            hrg_index = i
            break
    if hrg_index is None:
        continue

    # Extract raw star name (all parts from 1 up to 'hrg')
    raw_star_name = "_".join(parts[1:hrg_index])
    star_name = normalize_star_name(raw_star_name)

    # Map to canonical name using STAR_ALIASES
    star_name = STAR_ALIASES.get(star_name, star_name)

    # Extract date
    date_token = parts[-1].split("T")[0]
    if len(date_token) != 10:
        continue

    observations[star_name][date_token] += 1

# ---------------------------------------------------------
# 4. Prepare for plotting
# ---------------------------------------------------------
stars = sorted(observations.keys())
dates = sorted({date for s in stars for date in observations[s]})

all_counts = [observations[s][d] for s in stars for d in observations[s]]
min_obs = min(all_counts)
max_obs = max(all_counts)
norm = Normalize(vmin=min_obs, vmax=max_obs)

# ---------------------------------------------------------
# 5. Plotting function
# ---------------------------------------------------------
def plot_star_subset(star_subset, pdf_pages):
    x, y, counts = [], [], []

    for i, star in enumerate(star_subset):
        for j, date in enumerate(dates):
            if date in observations[star]:
                x.append(j)
                y.append(i)
                counts.append(observations[star][date])

    fig, ax = plt.subplots(figsize=(12, 6))
    sc = ax.scatter(x, y, c=counts, cmap="magma", edgecolors="k", s=120, norm=norm)

    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=9)

    ax.set_yticks(range(len(star_subset)))
    ax.set_yticklabels(star_subset)

    ax.set_xlabel("Date of Observation")
    ax.set_ylabel("Star Name")
    ax.set_title("RLMT Hα Grism Observations")

    plt.grid(True, linestyle="--", alpha=0.4)

    cbar = plt.colorbar(sc, label="Observations per Night")
    cbar.set_ticks([min_obs, max_obs])
    cbar.set_ticklabels([str(min_obs), str(max_obs)])

    plt.tight_layout()
    pdf_pages.savefig(fig)
    plt.close(fig)

# ---------------------------------------------------------
# 6. SAVE PDF SAFELY
# ---------------------------------------------------------
pdf_path = os.path.join(outpath, f"Be-star_observations_{dates[-1]}.pdf")
pdf_path2 = os.path.join(outpath2, f"Be-star_observations_{dates[-1]}.pdf")

print("\nGenerating PDF pages...\n")

with PdfPages(pdf_path) as pdf_pages:
    for i in range(0, len(stars), 10):
        subset = stars[i:i+10]
        plot_star_subset(subset, pdf_pages)

with PdfPages(pdf_path2) as pdf_pages:
    for i in range(0, len(stars), 10):
        subset = stars[i:i+10]
        plot_star_subset(subset, pdf_pages)

print(f"\nSaved PDF → {pdf_path}\n")
print(f"\nSaved PDF → {pdf_path2}\n")