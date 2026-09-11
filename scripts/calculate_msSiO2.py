#!/usr/bin/env python3
"""
calculate_msSiO2.py — 3D Monte Carlo Script for Compound Targets (SiO2 Quartz)
Supports single runs or reading directly from multi-column parameter files.
"""

import os
import sys
import argparse
import numpy as np
from numba import njit

# ==========================================
# 1. Parameter Reader & Table Parser
# ==========================================

def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="3D Monte Carlo Multiple Scattering Correction Driver for SiO2 (calculate_msSiO2.py)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  python3 calculate_msSiO2.py inputs/input_BurialDating_prod
  python3 calculate_msSiO2.py --scat-file data/SiO2.ntot --capt-file data/Be10.rip --isoprod Be10 --mass-g 1.5 --diam-mm 13.0
"""
    )
    parser.add_argument("params_file", nargs="?", default=None, help="Path to input table or config file")
    parser.add_argument("--scat-file", type=str, help="Path to total compound cross section file (SiO2.ntot)")
    parser.add_argument("--capt-file", type=str, help="Path to reaction cross section file (*.rip)")
    parser.add_argument("--isoprod", type=str, help="Product isotope code (e.g., Be10 or Al26)")
    parser.add_argument("--target-err", type=float, help="Target relative precision (e.g., 0.005 for 0.5%%)")
    parser.add_argument("--min-histories", type=int, help="Minimum MC histories per energy point")
    parser.add_argument("--max-histories", type=int, help="Maximum MC histories cutoff")
    parser.add_argument("--batch-size", type=int, help="Batch size per convergence loop")
    parser.add_argument("--diam-mm", type=float, help="Sample diameter in mm")
    parser.add_argument("--thickness-mm", type=float, help="Sample thickness in mm")
    parser.add_argument("--mass-g", type=float, help="Sample mass in grams")
    parser.add_argument("--density-g-cm3", type=float, help="Sample density in g/cm3")
    parser.add_argument("--molar-mass", type=float, help="Sample molar mass in g/mol")
    
    if len(sys.argv) == 1 and not os.path.exists("params.txt"):
        parser.print_help()
        sys.exit(0)

    return parser.parse_args()

def parse_input_table(filepath):
    """Parses whitespace-delimited multi-column input table files."""
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
                    "isoprod": parts[8],  # e.g., Be10 or Al26
                    "tirr_s": float(parts[9]),
                    "nprotons": float(parts[10]),
                    "bif": float(parts[11])
                }
                entries.append(entry)
    return entries

def resolve_capture_file(isoprod):
    """Resolves cross section file path for production reaction."""
    clean_iso = str(isoprod).replace("-", "").strip()
    
    candidate_paths = [
        os.path.join("data", f"{clean_iso}.rip"),
        os.path.join("data", f"rp{clean_iso}.rip")
    ]
    
    for path in candidate_paths:
        if os.path.exists(path):
            return path
            
    # Default fallback
    return os.path.join("data", f"{clean_iso}.rip")

def get_fms_output_filename(diam_mm, mass_g, isoprod):
    os.makedirs("tempfiles", exist_ok=True)
    diam_str = f"{float(diam_mm):.1f}"
    mass_str = f"{float(mass_g):.1f}"
    
    clean_isoprod = str(isoprod).replace("-", "").strip()
    iso_str = clean_isoprod if clean_isoprod.startswith("rp") else f"rp{clean_isoprod}"
    
    return f"tempfiles/fms_d{diam_str}_m{mass_str}_{iso_str}.dat"

# ==========================================
# 2. Pointwise ASCII Cross-Section Loader
# ==========================================

def load_reconstructed_xs(filepath):
    possible_paths = [
        filepath,
        os.path.join("data", os.path.basename(filepath)),
        os.path.join("data", os.path.basename(filepath).lower()),
        os.path.join("data", os.path.basename(filepath).upper())
    ]
    resolved_path = None
    for p in possible_paths:
        if os.path.exists(p):
            resolved_path = p
            break
    if not resolved_path:
        raise FileNotFoundError(f"Cross-section file '{filepath}' not found.")
    data = np.loadtxt(resolved_path, comments=('#', '//', 'C', 'c'), usecols=(0, 1), unpack=True)
    return np.ascontiguousarray(data[0], dtype=np.float64), np.ascontiguousarray(data[1], dtype=np.float64)

def build_unified_master_grid(grids_list):
    merged = np.concatenate(grids_list)
    unique_grid = np.unique(merged)
    return np.ascontiguousarray(sorted(unique_grid), dtype=np.float64)

# ==========================================
# 3. NUMBA JIT FULL 3D TRACKING KERNEL
# ==========================================

@njit(fastmath=True)
def get_xs_numba(E_eV, grid_nel, vals_nel, grid_target, vals_target):
    sig_nel    = np.interp(E_eV, grid_nel, vals_nel) if (E_eV >= grid_nel[0] and E_eV <= grid_nel[-1]) else 0.0
    sig_target = np.interp(E_eV, grid_target, vals_target) if (E_eV >= grid_target[0] and E_eV <= grid_target[-1]) else 0.0
    sig_tot    = sig_nel + sig_target
    return sig_tot, sig_nel, sig_target

@njit(fastmath=True)
def sample_isotropic_direction():
    phi = 2.0 * np.pi * np.random.random()
    w = 2.0 * np.random.random() - 1.0
    sin_th = np.sqrt(max(0.0, 1.0 - w**2))
    u = sin_th * np.cos(phi)
    v = sin_th * np.sin(phi)
    return u, v, w

@njit(fastmath=True)
def run_history_batch_3d_numba(E_inc, radius, thickness, N_atoms,
                                grid_nel, vals_nel, grid_target, vals_target,
                                batch_size=10000):
    batch_prim = np.zeros(batch_size, dtype=np.float64)
    batch_mult = np.zeros(batch_size, dtype=np.float64)
    AWR_SiO2 = 20.0  
    MAX_STACK = 256

    for i in range(batch_size):
        stack = np.zeros((MAX_STACK, 9), dtype=np.float64)
        sig_tot_inc, sig_nel_inc, sig_target_inc = get_xs_numba(
            E_inc, grid_nel, vals_nel, grid_target, vals_target
        )
        if sig_tot_inc <= 0.0:
            continue

        macro_tot_inc = sig_tot_inc * 1e-24 * N_atoms
        r_entry = radius * np.sqrt(np.random.random())
        th_entry = 2.0 * np.pi * np.random.random()
        x0 = r_entry * np.cos(th_entry)
        y0 = r_entry * np.sin(th_entry)
        z0 = 0.0
        u0, v0, w0 = 0.0, 0.0, 1.0

        P_interact = 1.0 - np.exp(-macro_tot_inc * thickness)
        if P_interact <= 0.0:
            continue

        batch_prim[i] = P_interact * (sig_target_inc / sig_tot_inc)
        dist_first = -np.log(1.0 - np.random.random() * P_interact) / macro_tot_inc
        
        stack[0, 0] = x0; stack[0, 1] = y0; stack[0, 2] = z0 + dist_first
        stack[0, 3] = u0; stack[0, 4] = v0; stack[0, 5] = w0
        stack[0, 6] = E_inc
        stack[0, 7] = 0.0; stack[0, 8] = 1.0
        stack_ptr = 1

        while stack_ptr > 0:
            stack_ptr -= 1
            x = stack[stack_ptr, 0]; y = stack[stack_ptr, 1]; z = stack[stack_ptr, 2]
            u = stack[stack_ptr, 3]; v = stack[stack_ptr, 4]; w = stack[stack_ptr, 5]
            E = stack[stack_ptr, 6]
            gen = int(stack[stack_ptr, 7])
            weight = stack[stack_ptr, 8]

            sig_tot, sig_nel, sig_target = get_xs_numba(
                E, grid_nel, vals_nel, grid_target, vals_target
            )
            if sig_tot <= 0.0:
                continue

            if gen > 0:
                macro_tot = sig_tot * 1e-24 * N_atoms
                dist = -np.log(np.random.random()) / macro_tot
                x += u * dist; y += v * dist; z += w * dist
                if (z < 0.0 or z > thickness) or (x**2 + y**2 > radius**2):
                    continue

            r_sample = np.random.random() * sig_tot
            if r_sample < sig_nel:
                alpha = ((AWR_SiO2 - 1.0) / (AWR_SiO2 + 1.0))**2
                mu_cm = 2.0 * np.random.random() - 1.0
                E_scattered = E * 0.5 * ((1.0 + alpha) + (1.0 - alpha) * mu_cm)
                
                if stack_ptr < MAX_STACK - 1:
                    us, vs, ws = sample_isotropic_direction()
                    stack[stack_ptr, 0] = x; stack[stack_ptr, 1] = y; stack[stack_ptr, 2] = z
                    stack[stack_ptr, 3] = us; stack[stack_ptr, 4] = vs; stack[stack_ptr, 5] = ws
                    stack[stack_ptr, 6] = E_scattered; stack[stack_ptr, 7] = gen + 1; stack[stack_ptr, 8] = weight
                    stack_ptr += 1
            else:
                if gen > 0:
                    batch_mult[i] += weight * P_interact

    return batch_prim, batch_mult

def run_simulation_adaptive_3d(E_inc, radius, thickness, N_atoms,
                               grid_nel, vals_nel, grid_target, vals_target,
                               target_rel_err=0.005, min_hist=10000, max_hist=100000, batch_size=10000):
    hist_primary = np.zeros(max_hist, dtype=np.float64)
    hist_multiple = np.zeros(max_hist, dtype=np.float64)
    current_histories = 0

    while current_histories < max_hist:
        next_hist = current_histories + batch_size
        b_prim, b_mult = run_history_batch_3d_numba(
            E_inc, radius, thickness, N_atoms,
            grid_nel, vals_nel, grid_target, vals_target,
            batch_size
        )
        hist_primary[current_histories:next_hist] = b_prim
        hist_multiple[current_histories:next_hist] = b_mult
        current_histories = next_hist

        if current_histories >= min_hist:
            X = hist_primary[:current_histories]
            Y_tot = X + hist_multiple[:current_histories]
            mean_X = np.mean(X)
            mean_Ytot = np.mean(Y_tot)
            
            if mean_X < 1e-12:
                return 0.0, 0.0, 1.0, 0.0, current_histories

            f_ms = mean_Ytot / mean_X if mean_X > 0 else 1.0
            if np.sum(hist_multiple[:current_histories]) == 0:
                return mean_X, 0.0, 1.0, 0.0, current_histories

            var_X = np.var(X, ddof=1)
            var_Ytot = np.var(Y_tot, ddof=1)
            cov_X_Ytot = np.cov(X, Y_tot)[0, 1]

            var_mean_X = var_X / current_histories
            var_mean_Ytot = var_Ytot / current_histories
            cov_mean_X_Ytot = cov_X_Ytot / current_histories

            rel_variance = (var_mean_Ytot / (mean_Ytot**2) + 
                            var_mean_X / (mean_X**2) - 
                            2.0 * cov_mean_X_Ytot / (mean_X * mean_Ytot))

            f_ms_uncertainty = f_ms * np.sqrt(np.maximum(0.0, rel_variance))
            rel_err = f_ms_uncertainty / f_ms if f_ms > 0 else 0.0

            if rel_err <= target_rel_err:
                break

    mean_X = np.mean(hist_primary[:current_histories])
    mean_Ytot = np.mean(hist_primary[:current_histories] + hist_multiple[:current_histories])
    f_ms = mean_Ytot / mean_X if mean_X > 0 else 1.0
    return mean_X, np.mean(hist_multiple[:current_histories]), f_ms, f_ms_uncertainty, current_histories

def run_mc_for_config(scat_file, capt_file, isoprod, mass_g, diam_mm, thick_mm,
                      density_g_cm3=2.65, molar_mass=60.0843, target_err=0.005,
                      min_histories=10000, max_histories=100000, batch_size=10000, random_seed=42):
    
    grid_nel, vals_nel = load_reconstructed_xs(scat_file)
    grid_target, vals_target = load_reconstructed_xs(capt_file)
    master_energy_grid = build_unified_master_grid([grid_nel, grid_target])

    thickness_cm = thick_mm * 0.1
    radius_cm = (diam_mm * 0.1) / 2.0
    N_atoms = (density_g_cm3 * 6.02214076e23) / molar_mass

    output_filename = get_fms_output_filename(diam_mm, mass_g, isoprod)
    np.random.seed(random_seed)

    # Warmup JIT
    _ = run_history_batch_3d_numba(
        4.906, radius_cm, thickness_cm, N_atoms,
        grid_nel, vals_nel, grid_target, vals_target,
        batch_size=100
    )

    with open(output_filename, 'w') as f_out:
        f_out.write(f"#target material  : SiO2 Quartz\n")
        f_out.write(f"#total xs file    : {scat_file}\n")
        f_out.write(f"#prod xs file     : {capt_file}\n")
        f_out.write(f"#sample mass  [g] : {mass_g:.4f}\n")
        f_out.write(f"#sample diam [mm] : {diam_mm:.1f}\n")
        f_out.write(f"#target precision : {target_err*100:.2f}%\n")
        f_out.write("#" * 105 + "\n")
        f_out.write(f"{'#Energy (eV)':<18}{'Weighted Prim':<16}{'Weighted Multi':<16}{'F_ms':<12}{'Uncertainty':<14}{'Rel Err':<10}{'Histories':<10}\n")
        f_out.write("#" * 105 + "\n")

        for i in range(len(master_energy_grid)):
            E_inc = master_energy_grid[i]
            prim, mult, f_ms, f_ms_err, histories = run_simulation_adaptive_3d(
                E_inc=E_inc, radius=radius_cm, thickness=thickness_cm, N_atoms=N_atoms,
                grid_nel=grid_nel, vals_nel=vals_nel, grid_target=grid_target, vals_target=vals_target,
                target_rel_err=target_err, min_hist=min_histories, max_hist=max_histories, batch_size=batch_size
            )
            rel_err_pct = (f_ms_err / f_ms) * 100 if f_ms > 0 else 0.0
            line = f"{E_inc:<18.6e}{prim*1000:<16.2f}{mult*1000:<16.2f}{f_ms:<12.5f}{f_ms_err:<14.5f}{rel_err_pct:<6.2f}%   {histories:<10}\n"
            f_out.write(line)

    print(f"    [3D MC OK] Saved results ({len(master_energy_grid)} points) to '{output_filename}'")

# ==========================================
# 5. Main Execution Script
# ==========================================

if __name__ == "__main__":
    cli_args = parse_cli_args()
    
    # Check if input file is a multi-column table
    if cli_args.params_file and os.path.exists(cli_args.params_file):
        try:
            table_entries = parse_input_table(cli_args.params_file)
        except Exception:
            table_entries = []
        
        if table_entries:
            print(f"Parsed {len(table_entries)} target configurations from table '{cli_args.params_file}'\n" + "="*75)
            for idx, entry in enumerate(table_entries, 1):
                material = entry["material"]
                isoprod = entry["isoprod"]
                scat_file = os.path.join("data", f"{material}.ntot")
                capt_file = resolve_capture_file(isoprod)
                
                print(f"[{idx}] Running Monte Carlo for sample '{entry['sample']}' | Target: {entry['iso2act']} -> Product: {isoprod}")
                run_mc_for_config(
                    scat_file=scat_file,
                    capt_file=capt_file,
                    isoprod=isoprod,
                    mass_g=entry["mass_g"],
                    diam_mm=entry["diam_mm"],
                    thick_mm=entry["thick_mm"]
                )
            sys.exit(0)

    # Fallback to single config execution
    isoprod_c = cli_args.isoprod if cli_args.isoprod else "Be10"
    scat_f = cli_args.scat_file if cli_args.scat_file else "data/SiO2.ntot"
    capt_f = cli_args.capt_file if cli_args.capt_file else resolve_capture_file(isoprod_c)
    mass = cli_args.mass_g if cli_args.mass_g else 1.5
    diam = cli_args.diam_mm if cli_args.diam_mm else 13.0
    thick = cli_args.thickness_mm if cli_args.thickness_mm else 5.0

    run_mc_for_config(
        scat_file=scat_f,
        capt_file=capt_f,
        isoprod=isoprod_c,
        mass_g=mass,
        diam_mm=diam,
        thick_mm=thick
    )
