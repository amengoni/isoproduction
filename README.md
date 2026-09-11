# calculate_prod.py

`calculate_prod.py` is a specialized Python utility for neutron-induced isotope production yield calculations and Spectrum-Averaged Cross Section (SACS) evaluation at the **CERN n_TOF facility** (covering experimental zones such as **NEAR**, **EAR1**, and **EAR2**). 

It automates target sample material calculations (areal density, stoichiometry, target nucleus counts) and calculates production yields (e.g., $^{10}\text{Be}$, $^{26}\text{Al}$) by delegating energy-dependent spectral integrations to an underlying calculation driver (`calculate_sacs3.py`) via subprocess calls.

---

## Directory Architecture

The script expects a standard repository layout relative to the execution directory:

```text
project_root/
├── scripts/
│   ├── calculate_prod.py       # Main production yield calculator
│   └── calculate_sacs3.py      # SACS spectral integration driver (called via subprocess)
├── data/                       # Directory containing neutron energy flux spectrum datasets
│   ├── NEAR-MCecc              # Default NEAR flux spectrum file
│   ├── NEAR-SKopanos-wmod-20cm # Alternative flux spectrum variant
│   ├── Z21-EAR1                # EAR1 central capture spectrum
│   ├── Z21-EAR1FC              # EAR1 full container spectrum
│   └── Z22-EAR2                # EAR2 neutron spectrum
├── tempfiles/                  # Contains pre-calculated multiple scattering data (Fms)
│   ├── fms_d13.0_m1.5_rpBe10.dat
│   └── fms_d13.0_m1.5_rp004010.dat
└── inputs/                     # Parameter input files
    └── input_NEAR_BurialDating_prod
```

---

## Dependencies & Requirements

* **Python**: `3.6+`
* **Python Packages**: `numpy`
* **Standard Modules**: `os`, `sys`, `argparse`, `subprocess`

Ensure `calculate_sacs3.py` is available in `scripts/` or the local working directory before executing.

---

## Physical & Mathematical Model

### 1. Target Nuclei Count & Areal Density
The total target nuclei count ($N_{\text{target}}$) and target thickness in atoms per barn ($t_{\text{barns}}$) are determined using compound stoichiometry and sample geometry:

$$\text{Molar Mass } (M_{\text{SiO}_2}) = 60.0843 \text{ g/mol}$$

$$N_{\text{target}} = \left(\frac{m_{\text{sample}}}{M_{\text{compound}}}\right) \times N_A \times S_{\text{stoich}}$$

$$t_{\text{barns}} = \left(\frac{N_{\text{compound}}}{\text{Area}_{\text{cm}^2}}\right) \times 10^{-24}$$

### 2. SACS Corrections Progression
The integration driver evaluates cross sections with progressive physical corrections:
* **SACS**: Uncorrected spectrum-averaged cross section ($\sigma$).
* **f-SACS**: Includes filter material self-attenuation ($f_{\text{att}}$).
* **ssf-SACS**: Includes target sample self-shielding ($f_{\text{self}}$).
* **ms-SACS**: Fully corrected cross section including multiple scattering ($F_{\text{ms}}$).

### 3. Fluence & Production Yield
Total pulse fluence ($\Phi_{\text{total}}$) is calculated relative to the standard n_TOF pulse intensity ($7.0 \times 10^{12}$ protons/pulse) scaled by the Beam Interception Factor ($	ext{BIF}$):

$$N_{\text{produced}} = N_{\text{target}} \times (\sigma_{\text{ms-SACS}} \times 10^{-24}) \times \Phi_{\text{total}}$$

---

## Command-Line Usage

```bash
python3 scripts/calculate_prod.py [input_file] [-f FLUX_VARIANT] [--no-ms]
```

### Options & Arguments

| Argument / Option | Type | Description |
| :--- | :--- | :--- |
| `input_file` | Positional | Path to the multi-column input parameters file. |
| `-f`, `--flux-variant` | Optional | Spectrum flux variant suffix (Default: `MCecc`). |
| `--no-ms` | Flag | Disables multiple scattering corrections (forces $F_{\text{ms}} = 1.0$). |

### Examples

1. **Standard Run (Default Flux Variant `MCecc`)**:
   ```bash
   python3 scripts/calculate_prod.py inputs/input_NEAR_BurialDating_prod
   ```

2. **Custom Flux Spectrum Variant**:
   ```bash
   python3 scripts/calculate_prod.py inputs/input_NEAR_BurialDating_prod -f SKopanos-wmod-20cm
   ```

3. **Disable Multiple Scattering Corrections**:
   ```bash
   python3 scripts/calculate_prod.py inputs/input_NEAR_BurialDating_prod --no-ms
   ```

---

## Input File Format

The input parameter file is a space-separated ASCII file (lines starting with `#` are treated as comments). Each row defines a target sample/irradiation setup with 12 mandatory columns:

```text
#  1           2        3        4        5         6         7       8        9        10         11     12
#area fthick[mm]   sample  mass[g] diam[mm] thick[mm]  material iso2act  isoprod   tirr[s]   nprotons     BIF
NEAR         0.0  quartz1      1.5     13.0      4.26      SiO2    Si28     Al26    491770  7.980e+17     1.0   
NEAR         0.0  quartz2      1.5     13.0      4.26      SiO2     O16     Be10    491770  7.980e+17     1.0   
NEAR         0.0  quartz1      2.5     13.0      4.26      SiO2    Si28     Al26    491770  7.980e+17     1.0   
NEAR         0.0  quartz2      2.5     13.0      4.26      SiO2     O16     Be10    491770  7.980e+17     1.0   

```

### Column Definitions

| Col | Field | Type | Description |
| :-: | :--- | :--- | :--- |
| **1** | `area` | String | Experimental area (`NEAR`, `EAR1CC`, `EAR1FC`, `EAR2`) |
| **2** | `fthick_mm` | Float | Filter thickness [mm] |
| **3** | `sample` | String | Sample identification label |
| **4** | `mass_g` | Float | Target sample mass [g] |
| **5** | `diam_mm` | Float | Sample diameter [mm] |
| **6** | `thick_mm` | Float | Sample thickness [mm] |
| **7** | `material` | String | Chemical compound formula (e.g., `SiO2`) |
| **8** | `iso2act` | String | Target activation isotope (e.g., `O16`, `Si28`) |
| **9** | `isoprod` | String | Reaction product notation (`004010`, `rpBe10`, or `Be10`) |
| **10** | `tirr_s` | Float | Irradiation duration [s] |
| **11** | `nprotons` | Float | Total integrated protons on target |
| **12** | `bif` | Float | Beam Interception Factor (BIF) |

---

## Output Description

Output is printed directly to `STDOUT` and automatically written to `prod_summary.out` in the working directory:

```text
# Input Table           : inputs/input_NEAR_BurialDating_prod
# Resolved Spectrum(s)  : NEAR-MCecc
# SACS Driver           : scripts/calculate_sacs3.py via Subprocess Execution
# Corrections Applied   : Self-attenuation (f_att) & Multiple Scattering (Fms)
# Flux Variant Option   : MCecc
#1        2          3       4        5          6         7          8            9            10           11           12               13              14      
#area     sample     target  product  fthick[mm] mass[g]   thick[mm]  SACS[b]      f-SACS[b]    ssf-SACS[b]  ms-SACS[b]   n_total[n/cm2]   N_produced      BIF     
#################################################################################################################################################################
NEAR     SAMP_A     O16     Be10     0.00       1.5000    2.0000     1.240e-04    1.240e-04    1.215e-04    1.285e-04    8.928e+14        1.854e+09       1.00    
NEAR     SAMP_B     Si28    Al26     0.00       2.0000    2.5000     3.120e-04    3.120e-04    3.050e-04    3.210e-04    8.482e+14        5.412e+09       0.95    

# Summary table saved to 'prod_summary.out'.
```

---

## Troubleshooting

* **`[FATAL ERROR] Spectrum file error... FILE NOT FOUND`**:
  Verify that the requested spectrum file (e.g., `data/NEAR-MCecc`) exists inside the `data/` folder.
* **`Could not locate 'calculate_sacs3.py'`**:
  Ensure `calculate_sacs3.py` is present in `scripts/` or the execution directory.
* **Missing Multiple Scattering Files**:
  If `tempfiles/fms_*.dat` files are not found, the code defaults to `Fms = 1.0` or allows execution with `--no-ms`.
