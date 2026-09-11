#!/usr/bin/env python3

import os
import sys
import argparse
import subprocess
import numpy as np

# ==========================================
# Constants & Reference Data
# ==========================================
DATA_DIR = "data"
TEMP_DIR = "tempfiles"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

AVOGADRO = 6.02214076e23       # atoms/mol
PROTONS_PER_STANDARD_PULSE = 7.0e12  # Standard n_TOF pulse intensity
RHO_SIO2 = 2.65                # Density of SiO2 [g/cm3] for sanity check

MOLAR_MASSES = {
    "SiO2": 60.0843,
    "Si28": 27.9769,
    "O16": 15.9949,
}

STOICHIOMETRY = {
    ("SiO2", "Si28"): 1.0,
    ("SiO2", "O16"): 2.0,
}

ELEMENT_SYMBOLS = {
    1: "H",   2: "He",  3: "Li",  4: "Be",  5: "B",   6: "C",   7: "N",   8: "O",   9: "F",  10: "Ne",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",  16: "S",  17: "Cl", 18: "Ar", 19: "K",  20: "Ca",
    21: "Sc", 22: "Ti", 23: "V",  24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn",
    31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr", 37: "Rb", 38: "Sr", 39: "Y",  40: "Zr",
    41: "Nb", 42: "Mo", 43: "Tc", 44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In", 50: "Sn",
    51: "Sb", 52: "Te", 53: "I",  54: "Xe", 55: "Cs", 56: "Ba", 57: "La", 58: "Ce", 59: "Pr", 60: "Nd",
    61: "Pm", 62: "Sm", 63: "Eu", 64: "Gd", 65: "Tb", 66: "Dy", 67: "Ho", 68: "Er", 69: "Tm", 70: "Yb",
    71: "Lu", 72: "Hf", 73: "Ta", 74: "W",  75: "Re", 76: "Os", 77: "Ir", 78: "Pt", 79: "Au", 80: "Hg",
    81: "Tl", 82: "Pb", 83: "Bi", 84: "Po", 85: "At", 86: "Rn", 87: "Fr", 88: "Ra", 89: "Ac", 90: "Th",
    91: "Pa", 92: "U",  93: "Np", 94: "Pu", 95: "Am", 96: "Cm"
}

USAGE_TEXT = """
==============================================================================
 Usage Block - calculate_prod.py
==============================================================================
Description:
  Calculates isotope production (e.g., Be10, Al26) and SACS metrics 
  (SACS, f-SACS, ssf-SACS, ms-SACS) by delegating spectral integration to 
  calculate_sacs3.py via subprocess calls.

Usage:
  python3 scripts/calculate_prod.py [input_file] [-f FLUX_VARIANT] [--no-ms]

Arguments:
  input_file          : Path to input parameters file (e.g., inputs/input_NEAR_BurialDating_prod)
  -f, --flux-variant  : Flux spectrum variant (default: MCecc)
  --no-ms             : Disable multiple scattering correction (forces Fms = 1.0)

Examples:
  python3 scripts/calculate_prod.py inputs/input_NEAR_BurialDating_prod -f SKopanos-wmod-20cm
  python3 scripts/calculate_prod.py inputs/input_NEAR_BurialDating_prod --no-ms
  python3 scripts/calculate_prod.py inputs/input_NEAR_BurialDating_prod -f SKopanos-wmod-20cm --no-ms
==============================================================================
"""

def parse_isoprod_name(isoprod_str):
    """Parses ZZZAAA string (e.g. '004010') or isotope string into element notation without hyphens (e.g. 'Be10')."""
    clean_str = str(isoprod_str).lower().replace("rp", "").replace(".rip", "").strip()
    if len(clean_str) == 6 and clean_str.isdigit():
        z = int(clean_str[:3])
        a = int(clean_str[3:])
        symbol = ELEMENT_SYMBOLS.get(z, f"Z{z}")
        return f"{symbol}{a}"
    return str(isoprod_str).replace("-", "").strip()

def resolve_spectrum_file(area, flux_variant="MCecc"):
    """Resolves neutron spectrum filename from area string and validates existence."""
    area_clean = str(area).strip()
    if area_clean == "NEAR":
        spectrum_file = f"NEAR-{flux_variant}"
    elif area_clean == "EAR1CC":
        spectrum_file = "Z21-EAR1"
    elif area_clean in ["EAR1FC", "EAR1"]:
        spectrum_file = "Z21-EAR1FC"
    elif area_clean == "EAR2":
        spectrum_file = "Z22-EAR2"
    else:
        spectrum_file = f"{area_clean}-{flux_variant}"

    target_path = os.path.join(DATA_DIR, spectrum_file)
    if not os.path.exists(target_path):
        sys.stderr.write(
            f"\n[FATAL ERROR] Spectrum file error for area '{area_clean}' with variant '{flux_variant}':\n"
            f"  Expected file path : '{target_path}'\n"
            f"  Status            : FILE NOT FOUND or INCOMPATIBLE VARIANT\n"
            f"Execution terminated.\n\n"
        )
        sys.exit(1)
    return spectrum_file

def resolve_sacs3_script():
    """Locates calculate_sacs3.py in scripts/ folder or local execution paths."""
    candidate_paths = [
        os.path.join(os.getcwd(), "scripts", "calculate_sacs3.py"),
        os.path.join(SCRIPT_DIR, "calculate_sacs3.py"),
        os.path.join(SCRIPT_DIR, "scripts", "calculate_sacs3.py"),
        "scripts/calculate_sacs3.py",
        "calculate_sacs3.py"
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Could not locate 'calculate_sacs3.py' in 'scripts/' or working directory.")

def get_fms_filepath(diam_mm, mass_g, isoprod_raw, no_ms=False):
    """Locates Fms file path from tempfiles/ using updated rp<Isotope> notation."""
    if no_ms:
        return "NONE"

    # Ensure 'rp' prefix is present for isotope string (e.g., Be10 -> rpBe10, 004010 -> rp004010)
    iso_str = isoprod_raw if isoprod_raw.startswith("rp") else f"rp{isoprod_raw}"

    # Primary expected filename format: fms_d13.0_m1.5_rpBe10.dat
    formatted_filename = f"fms_d{diam_mm:.1f}_m{mass_g:.1f}_{iso_str}.dat"
    fms_path = os.path.join(TEMP_DIR, formatted_filename)

    if not os.path.exists(fms_path):
        # Fallback 1: Integer diameter format
        alt_filename = f"fms_d{int(diam_mm)}_m{mass_g:.1f}_{iso_str}.dat"
        fms_path_alt = os.path.join(TEMP_DIR, alt_filename)
        if os.path.exists(fms_path_alt):
            fms_path = fms_path_alt
        else:
            # Fallback 2: Check without 'rp' prefix (legacy support)
            legacy_filename = f"fms_d{diam_mm:.1f}_m{mass_g:.1f}_{isoprod_raw}.dat"
            fms_path_legacy = os.path.join(TEMP_DIR, legacy_filename)
            if os.path.exists(fms_path_legacy):
                fms_path = fms_path_legacy

    return fms_path if os.path.exists(fms_path) else "NONE"

def parse_input_file(filepath):
    entries = []
    with open(filepath, 'r') as f:
        for line in f:
            line_str = line.strip()
            if not line_str or line_str.startswith('#'):
                continue
            parts = line_str.split()
            if len(parts) >= 12:
                entry = {
                    "area": parts[0],
                    "fthick_mm": float(parts[1]),
                    "sample": parts[2],
                    "mass_g": float(parts[3]),
                    "diam_mm": float(parts[4]),
                    "thick_mm": float(parts[5]),
                    "material": parts[6],
                    "iso2act": parts[7],
                    "isoprod": parts[8],
                    "tirr_s": float(parts[9]),
                    "nprotons": float(parts[10]),
                    "bif": float(parts[11])  # Beam Interception Factor
                }
                entries.append(entry)
    return entries

def call_calculate_sacs3(nspectrum, iso, react, fthick_mm, sample_mat, 
                         sample_thick_atoms_per_barn, bpd=100.0, bin_mode=0, fms_file="NONE"):
    """
    Executes scripts/calculate_sacs3.py as a subprocess.
    """
    sacs3_script = resolve_sacs3_script()

    cmd = [
        sys.executable, sacs3_script,
        nspectrum,                               # 1. nspectrum
        str(iso),                                # 2. iso
        str(react),                              # 3. react
        str(fthick_mm),                          # 4. fthick_mm
        str(sample_mat),                         # 5. sample_mat
        f"{sample_thick_atoms_per_barn:.6e}",    # 6. sample_thick_atoms_per_barn
        str(int(bpd)),                           # 7. bpd
        str(bin_mode),                           # 8. bin_mode (0 = Master Grid)
        "1",                                     # 9. extrap_mode
        "1/E",                                   # 10. interp_flag
        fms_file                                 # 11. fms_file
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(f"[ERROR] calculate_sacs3.py failed:\n{proc.stderr}\n")
        raise RuntimeError("Subprocess execution failed.")

    for line in proc.stdout.splitlines():
        if line.startswith("line:"):
            parts = line.split()
            return {
                "nn": float(parts[3]),
                "fnn": float(parts[4]),
                "SACS": float(parts[5]),
                "f-SACS": float(parts[6]),
                "MB-SACS": float(parts[7]),
                "ssf-SACS": float(parts[8]),
                "ms-SACS": float(parts[9])
            }

    sys.stderr.write(f"[ERROR] Could not parse 'line:' from calculate_sacs3.py output:\n{proc.stdout}\n")
    raise ValueError("Invalid output format from calculate_sacs3.py")

def calculate_production_for_entry(entry, flux_variant="MCecc", no_ms=False):
    material = entry["material"]
    iso2act = entry["iso2act"]
    isoprod_raw = entry["isoprod"]
    isoprod_name = parse_isoprod_name(isoprod_raw)
    
    mass_g = entry["mass_g"]
    diam_mm = entry["diam_mm"]
    thick_mm = entry["thick_mm"]
    fthick_mm = entry["fthick_mm"]
    area = entry["area"]
    nprotons = entry["nprotons"]
    bif = entry["bif"]

    # 1. Target Nuclei Count & Areal Density
    molar_mass_compound = MOLAR_MASSES.get(material, 60.0843)
    stoich_factor = STOICHIOMETRY.get((material, iso2act), 1.0)
    n_target_nuclei = (mass_g / molar_mass_compound) * AVOGADRO * stoich_factor

    diam_cm = diam_mm * 0.1
    sample_area_cm2 = np.pi * (0.5 * diam_cm) ** 2
    n_compound_molecules = (mass_g / molar_mass_compound) * AVOGADRO
    compound_atoms_per_barn = (n_compound_molecules / sample_area_cm2) * 1e-24

    # 2. Spectrum Resolution
    spectrum_file = resolve_spectrum_file(area, flux_variant)

    # 3. Locate Fms file
    fms_path = get_fms_filepath(diam_mm, mass_g, isoprod_raw, no_ms=no_ms)

    # 4. Target mapping for calculate_sacs3.py
    iso_param = isoprod_raw if isoprod_raw.startswith("rp") else (f"rp{isoprod_raw}" if isoprod_raw.isdigit() else isoprod_raw)
    react_param = "rip"

    sacs_res = call_calculate_sacs3(
        nspectrum=spectrum_file,
        iso=iso_param,
        react=react_param,
        fthick_mm=fthick_mm,
        sample_mat=material,
        sample_thick_atoms_per_barn=compound_atoms_per_barn,
        bpd=100.0,
        bin_mode=0,
        fms_file=fms_path
    )

    # 5. Fluence Normalization
    equivalent_pulses = nprotons / PROTONS_PER_STANDARD_PULSE
    if "Z21" in spectrum_file or "EAR1" in spectrum_file or "Z22" in spectrum_file or "EAR2" in spectrum_file:
        total_fluence = (sacs_res["nn"] / sample_area_cm2) * equivalent_pulses * bif
    else:
        total_fluence = sacs_res["nn"] * equivalent_pulses * bif

    # Production calculated with final fully-corrected ms-SACS cross section
    total_atoms = n_target_nuclei * (sacs_res["ms-SACS"] * 1e-24) * total_fluence

    return {
        "area": area,
        "sample": entry["sample"],
        "target": iso2act,
        "isoprod": isoprod_name,
        "fthick_mm": fthick_mm,
        "mass_g": mass_g,
        "thick_mm": thick_mm,
        "sacs_b": sacs_res["SACS"],
        "f_sacs_b": sacs_res["f-SACS"],
        "ssf_sacs_b": sacs_res["ssf-SACS"],
        "ms_sacs_b": sacs_res["ms-SACS"],
        "n_target_nuclei": n_target_nuclei,
        "total_fluence": total_fluence,
        "total_atoms": total_atoms,
        "bif": bif,
        "spec_used": spectrum_file
    }

def main():
    if len(sys.argv) == 1:
        print(USAGE_TEXT)
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Calculate isotope production yield and SACS metrics with formatted output.",
        epilog=USAGE_TEXT,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input_file", nargs="?", default=None, help="Path to input parameter file")
    parser.add_argument("-f", "--flux-variant", default="MCecc", help="Flux spectrum variant (default: MCecc)")
    parser.add_argument("--no-ms", action="store_true", help="Disable multiple scattering correction (forces Fms = 1.0)")

    args = parser.parse_args()

    if not args.input_file or not os.path.exists(args.input_file):
        if args.input_file:
            print(f"[Error] Specified input parameter file '{args.input_file}' was not found.\n")
        print(USAGE_TEXT)
        sys.exit(1)

    entries = parse_input_file(args.input_file)
    if not entries:
        print(f"[Error] No valid data entries parsed from '{args.input_file}'.")
        sys.exit(1)

    resolved_spectrums = sorted(list(set(
        resolve_spectrum_file(entry["area"], args.flux_variant) for entry in entries
    )))
    spec_str = ", ".join(resolved_spectrums)

    ms_status = "Disabled via --no-ms (Fms = 1.0)" if args.no_ms else "Self-attenuation (f_att) & Multiple Scattering (Fms)"

    # Metadata Header construction
    meta_headers = [
        f"# Input Table           : {args.input_file}",
        f"# Resolved Spectrum(s)  : {spec_str}",
        f"# SACS Driver           : scripts/calculate_sacs3.py via Subprocess Execution",
        f"# Corrections Applied   : {ms_status}",
        f"# Flux Variant Option   : {args.flux_variant}"
    ]

    # Column Headers: area as 1st column, BIF (Beam Interception Factor) as 14th/last column
    header_fmt1 = f"#{'1':<8} {'2':<10} {'3':<7} {'4':<8} {'5':<10} {'6':<9} {'7':<10} {'8':<12} {'9':<12} {'10':<12} {'11':<12} {'12':<16} {'13':<15} {'14':<8}"
    header_fmt2 = f"#{'area':<7} {'sample':<10} {'target':<7} {'product':<8} {'fthick[mm]':<10} {'mass[g]':<9} {'thick[mm]':<10} {'SACS[b]':<12} {'f-SACS[b]':<12} {'ssf-SACS[b]':<12} {'ms-SACS[b]':<12} {'n_total[n/cm2]':<16} {'N_produced':<15} {'BIF':<8}"
    divider_line = "#" * len(header_fmt2)

    table_headers = f"{header_fmt1}\n{header_fmt2}\n{divider_line}"
    output_lines = meta_headers + [table_headers]

    for entry in entries:
        try:
            res = calculate_production_for_entry(entry, flux_variant=args.flux_variant, no_ms=args.no_ms)
            line_str = (
                f"{res['area']:<8} {res['sample']:<10} {res['target']:<7} {res['isoprod']:<8} "
                f"{res['fthick_mm']:<10.2f} {res['mass_g']:<9.4f} {res['thick_mm']:<10.4f} "
                f"{res['sacs_b']:<12.3e} {res['f_sacs_b']:<12.3e} {res['ssf_sacs_b']:<12.3e} {res['ms_sacs_b']:<12.3e} "
                f"{res['total_fluence']:<16.3e} {res['total_atoms']:<15.3e} {res['bif']:<8.2f}"
            )
            output_lines.append(line_str)
        except Exception as e:
            print(f"[Error] Failed processing sample {entry.get('sample')}: {e}")

    # Print to STDOUT
    print("\n".join(output_lines))

    # Save output file
    output_filename = "prod_summary.out"
    with open(output_filename, "w") as out_file:
        out_file.write("\n".join(output_lines) + "\n")

    print(f"\n# Summary table saved to '{output_filename}'.")

if __name__ == "__main__":
    main()
