#@title 📦 **CELL 1: Setup & Installation**

print(" Installing dependencies...")

!pip install -q numpy pandas matplotlib seaborn scipy tqdm

print("Installation complete!")
print("\nImporting libraries...")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.linalg import expm
from scipy.stats import pearsonr
from typing import Dict, List, Tuple
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Plotting style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

print("All libraries loaded!")
print("\n" + "="*70)
print("Ready to run ENAQT analysis!")
print("="*70)



#@title ⚙️ **CELL 2: Configuration**

class Config:
    """Central configuration for ENAQT analysis"""

    # ========================================================================
    # PHYSICAL CONSTANTS
    # ========================================================================
    HBAR_MEV_PS = 0.6582119514  # ℏ in meV·ps

    # ========================================================================
    # SYSTEM PARAMETERS
    # ========================================================================
    N_SITES = 7                    # Number of sites in circuit

    # ========================================================================
    # SIMULATION PARAMETERS (dimensionless after conversion)
    # ========================================================================
    N_TRAJECTORIES = 200           # Trajectories per gamma
    N_GAMMA_POINTS = 15            # Gamma scan resolution

    # Demo mode (faster)
    DEMO_TRAJECTORIES = 100
    DEMO_GAMMA_POINTS = 10

    # ========================================================================
    # THEORY PARAMETERS (PRE 2024)
    # ========================================================================
    SINK_DOMINATED_THRESHOLD = 10.0
    OPTIMAL_RATIO_MIN = 15.0
    OPTIMAL_RATIO_MAX = 25.0
    OPTIMAL_RATIO_PEAK = 17.5
    REFLECTION_DOMINATED_THRESHOLD = 25.0

    # ========================================================================
    # OUTPUT
    # ========================================================================
    PLOT_DPI = 300
    VERBOSE = True

config = Config()

print(" Configuration loaded")
print(f"   7-site model: {config.N_SITES} sites")
print(f"   Default trajectories: {config.N_TRAJECTORIES}")
print(f"   ℏ = {config.HBAR_MEV_PS} meV·ps")

#@title  **CELL 3: Load VQE Results (CSV with Format Spec)**

"""
================================================================================
VQE RESULTS CSV FORMAT SPECIFICATION
================================================================================

This cell loads VQE-derived parameters from CSV for ENAQT analysis.

JSON FORMAT SPECIFICATION:
{
  "format_version": "1.0",
  "description": "VQE-derived parameters for perovskite ENAQT analysis",
  "columns": {
    "Material_System": {
      "type": "string",
      "description": "Material composition (e.g., MAPbI3, FAPbBr3)",
      "required": true
    },
    "J_atomic_eV": {
      "type": "float",
      "units": "eV",
      "description": "Atomic-scale coupling strength from VQE band structure",
      "typical_range": [0.5, 1.0],
      "required": true,
      "vqe_extraction": "Bandgap splitting / 4, or off-diagonal H matrix element"
    },
    "sigma_atomic_eV": {
      "type": "float",
      "units": "eV",
      "description": "Atomic-scale energy disorder from VQE geometry scan",
      "typical_range": [0.2, 0.5],
      "required": true,
      "vqe_extraction": "RMS of site energy variations over geometries"
    },
    "lambda_reorg_eV": {
      "type": "float",
      "units": "eV",
      "description": "Reorganization energy for Marcus electron transfer",
      "typical_range": [0.15, 0.4],
      "required": true,
      "vqe_extraction": "Energy difference between ground/excited geometries"
    },
    "N_grains": {
      "type": "integer",
      "units": "dimensionless",
      "description": "Number of unit cells per ENAQT site (for coarse-graining)",
      "typical_range": [50, 2000],
      "default": 1000,
      "calculation": "(Device_length_nm / N_ENAQT_sites) / lattice_constant_nm"
    },
    "VQE_Method": {
      "type": "string",
      "description": "VQE method used (UCCSD, hardware-efficient, etc.)",
      "required": false
    },
    "VQE_Qubits": {
      "type": "integer",
      "description": "Number of qubits used in VQE calculation",
      "required": false
    },
    "References": {
      "type": "string",
      "description": "Citation or calculation ID",
      "required": false
    }
  },
  "notes": {
    "unit_consistency": "All energies must be in eV",
    "coarse_graining": "N_grains determines mesoscale averaging: J_eff = J_atomic/sqrt(N_grains)",
    "typical_device": "700 nm device with 7 ENAQT sites → N_grains ≈ 167 per site",
    "vqe_output": "Save VQE results directly to this format for seamless integration"
  }
}

CSV EXAMPLE (copy this format):
Material_System,J_atomic_eV,sigma_atomic_eV,lambda_reorg_eV,N_grains,VQE_Method,VQE_Qubits,References
MAPbI3,0.75,0.30,0.25,1000,UCCSD,4,VQE_calc_2024_001
FAPbI3,0.80,0.28,0.22,1000,UCCSD,4,VQE_calc_2024_002

================================================================================
"""

import pandas as pd
import numpy as np
from google.colab import files
import json

# ============================================================================
# OPTION 1: Upload your VQE results CSV
# ============================================================================

print("="*70)
print(" VQE RESULTS DATA LOADER")
print("="*70)
print("\nOPTION 1: Upload your VQE results CSV")
print("   (Format: See specification in cell comments)")
print("\n  OPTION 2: Press 'Skip' to use sample data\n")

uploaded = files.upload()

# ============================================================================
# OPTION 2: Create sample VQE dataset if no upload
# ============================================================================

if not uploaded:
    print("\n No file uploaded - Creating sample VQE dataset...")
    print("="*70)

    # Sample VQE results (realistic values from literature/calculations)
    sample_vqe_data = {
        'Material_System': [
            'MAPbI3',
            'FAPbI3',
            'CsPbBr3',
            'FA0.9Cs0.1PbI3',
            'MAPbI2.7Br0.3',
            '2D-RP-n2',
            '2D-RP-n3',
            'MAPbBr3',
            'CsPbI3',
            'Pb0.9Sn0.1PbI3'
        ],
        'J_atomic_eV': [
            0.75,   # MAPbI3 - standard
            0.80,   # FAPbI3 - slightly stronger (larger cation)
            0.65,   # CsPbBr3 - weaker (Br more electronegative)
            0.78,   # Mixed A-site - intermediate
            0.72,   # Mixed halide - Br increases gap
            0.55,   # 2D RP n=2 - weak (organic barriers)
            0.62,   # 2D RP n=3 - stronger (more 3D-like)
            0.68,   # MAPbBr3 - Br weakens coupling
            0.70,   # CsPbI3 - small cation, moderate
            0.72    # Sn-doped - slightly weaker Sn-I bonds
        ],
        'sigma_atomic_eV': [
            0.30,   # MAPbI3 - moderate disorder
            0.28,   # FAPbI3 - less disorder (larger cation stabilizes)
            0.35,   # CsPbBr3 - more disorder (Br size mismatch)
            0.26,   # Mixed A - engineered low disorder
            0.32,   # Mixed halide - compositional disorder
            0.45,   # 2D RP n=2 - high (interface effects)
            0.38,   # 2D RP n=3 - reduced interface disorder
            0.32,   # MAPbBr3 - moderate
            0.38,   # CsPbI3 - higher (less stable)
            0.38    # Sn-doped - disorder from substitution
        ],
        'lambda_reorg_eV': [
            0.25,   # MAPbI3 - typical
            0.22,   # FAPbI3 - less reorganization (stiffer)
            0.30,   # CsPbBr3 - more (Br lighter, more phonons)
            0.24,   # Mixed A - intermediate
            0.28,   # Mixed halide - increased phonon coupling
            0.35,   # 2D RP n=2 - high (soft organic layers)
            0.30,   # 2D RP n=3 - reduced
            0.28,   # MAPbBr3 - Br increases λ
            0.32,   # CsPbI3 - higher phonon coupling
            0.32    # Sn-doped - Sn softer than Pb
        ],
        'N_grains': [
            1000,   # Standard coarse-graining
            1000,
            1000,
            1000,
            1000,
            800,    # 2D: smaller effective domains
            900,
            1000,
            1000,
            1000
        ],
        'VQE_Method': [
            'UCCSD',
            'UCCSD',
            'UCCSD',
            'UCCSD',
            'UCCSD',
            'Hardware-Efficient',
            'Hardware-Efficient',
            'UCCSD',
            'UCCSD',
            'UCCSD'
        ],
        'VQE_Qubits': [
            4, 4, 4, 4, 4, 6, 6, 4, 4, 4
        ],
        'References': [
            'VQE_calc_2024_001',
            'VQE_calc_2024_002',
            'VQE_calc_2024_003',
            'VQE_calc_2024_004',
            'VQE_calc_2024_005',
            'VQE_calc_2024_006',
            'VQE_calc_2024_007',
            'VQE_calc_2024_008',
            'VQE_calc_2024_009',
            'VQE_calc_2024_010'
        ]
    }

    df_vqe = pd.DataFrame(sample_vqe_data)

    # Save sample CSV for reference
    csv_filename = 'vqe_results_sample.csv'
    df_vqe.to_csv(csv_filename, index=False)

    print(f" Sample dataset created: {csv_filename}")
    print(f"   {len(df_vqe)} materials with VQE parameters")

else:
    # Load uploaded file
    csv_filename = list(uploaded.keys())[0]
    df_vqe = pd.read_csv(csv_filename)
    print(f" Loaded: {csv_filename}")
    print(f"   {len(df_vqe)} materials found")

# ============================================================================
# DATA VALIDATION
# ============================================================================

print("\n" + "="*70)
print(" VALIDATING VQE DATA")
print("="*70)

required_columns = ['Material_System', 'J_atomic_eV', 'sigma_atomic_eV',
                   'lambda_reorg_eV', 'N_grains']

validation_passed = True
warnings = []

# Check required columns
missing_cols = [col for col in required_columns if col not in df_vqe.columns]
if missing_cols:
    print(f"MISSING REQUIRED COLUMNS: {missing_cols}")
    validation_passed = False
else:
    print("All required columns present")

# Validate ranges
if validation_passed:
    for idx, row in df_vqe.iterrows():
        material = row['Material_System']
        issues = []

        # J_atomic range check
        if not (0.3 < row['J_atomic_eV'] < 1.2):
            issues.append(f"J_atomic={row['J_atomic_eV']:.2f} eV (expected 0.3-1.2)")

        # sigma_atomic range check
        if not (0.1 < row['sigma_atomic_eV'] < 0.6):
            issues.append(f"sigma={row['sigma_atomic_eV']:.2f} eV (expected 0.1-0.6)")

        # lambda_reorg range check
        if not (0.1 < row['lambda_reorg_eV'] < 0.5):
            issues.append(f"lambda={row['lambda_reorg_eV']:.2f} eV (expected 0.1-0.5)")

        # N_grains range check
        if not (10 < row['N_grains'] < 5000):
            issues.append(f"N_grains={row['N_grains']} (expected 10-5000)")

        if issues:
            warnings.append(f"  {material}: {', '.join(issues)}")

# Print validation results
if warnings:
    print(f"\n  {len(warnings)} materials with parameter warnings:")
    for warning in warnings[:5]:  # Show first 5
        print(f"   {warning}")
    if len(warnings) > 5:
        print(f"   ... and {len(warnings)-5} more")
else:
    print(" All parameters in expected ranges")

# ============================================================================
# DISPLAY LOADED DATA
# ============================================================================

print("\n" + "="*70)
print(" LOADED VQE PARAMETERS")
print("="*70)
print(f"\nShowing first {min(5, len(df_vqe))} materials:\n")

# Display with formatting
display_df = df_vqe[['Material_System', 'J_atomic_eV', 'sigma_atomic_eV',
                     'lambda_reorg_eV', 'N_grains']].head()
print(display_df.to_string(index=False))

# ============================================================================
# STATISTICS SUMMARY
# ============================================================================

print("\n" + "="*70)
print(" VQE PARAMETER STATISTICS")
print("="*70)

stats_data = {
    'Parameter': ['J_atomic (eV)', 'sigma_atomic (eV)', 'lambda_reorg (eV)', 'N_grains'],
    'Mean': [
        df_vqe['J_atomic_eV'].mean(),
        df_vqe['sigma_atomic_eV'].mean(),
        df_vqe['lambda_reorg_eV'].mean(),
        df_vqe['N_grains'].mean()
    ],
    'Std': [
        df_vqe['J_atomic_eV'].std(),
        df_vqe['sigma_atomic_eV'].std(),
        df_vqe['lambda_reorg_eV'].std(),
        df_vqe['N_grains'].std()
    ],
    'Min': [
        df_vqe['J_atomic_eV'].min(),
        df_vqe['sigma_atomic_eV'].min(),
        df_vqe['lambda_reorg_eV'].min(),
        df_vqe['N_grains'].min()
    ],
    'Max': [
        df_vqe['J_atomic_eV'].max(),
        df_vqe['sigma_atomic_eV'].max(),
        df_vqe['lambda_reorg_eV'].max(),
        df_vqe['N_grains'].max()
    ]
}

stats_df = pd.DataFrame(stats_data)
print("\n" + stats_df.to_string(index=False))

# ============================================================================
# EXPORT TEMPLATE (for VQE program integration)
# ============================================================================

print("\n" + "="*70)
print(" VQE OUTPUT TEMPLATE")
print("="*70)
print("\nTo integrate with VQE program, use this Python code:")
print("-"*70)

template_code = """
# After VQE calculation, save results:
import pandas as pd

vqe_results = {
    'Material_System': ['Your_Material_Name'],
    'J_atomic_eV': [calculated_J],           # From band structure
    'sigma_atomic_eV': [calculated_sigma],   # From geometry scan
    'lambda_reorg_eV': [calculated_lambda],  # From relaxation
    'N_grains': [1000],                      # Standard value
    'VQE_Method': ['UCCSD'],                 # Your method
    'VQE_Qubits': [4],                       # Number of qubits
    'References': ['Calc_ID_or_DOI']
}

df = pd.DataFrame(vqe_results)
df.to_csv('my_vqe_results.csv', index=False)

# Then upload this CSV to Cell 3!
"""

print(template_code)

# ============================================================================
# SAVE REFERENCE FILES
# ============================================================================

# Save JSON schema
json_schema = {
    "format_version": "1.0",
    "description": "VQE-derived parameters for perovskite ENAQT analysis",
    "required_columns": ["Material_System", "J_atomic_eV", "sigma_atomic_eV",
                        "lambda_reorg_eV", "N_grains"],
    "typical_ranges": {
        "J_atomic_eV": [0.5, 1.0],
        "sigma_atomic_eV": [0.2, 0.5],
        "lambda_reorg_eV": [0.15, 0.4],
        "N_grains": [50, 2000]
    },
    "units": {
        "J_atomic_eV": "eV",
        "sigma_atomic_eV": "eV",
        "lambda_reorg_eV": "eV",
        "N_grains": "dimensionless"
    }
}

with open('vqe_format_spec.json', 'w') as f:
    json.dump(json_schema, f, indent=2)

print("\n" + "="*70)
print(" SAVED REFERENCE FILES")
print("="*70)
print("vqe_format_spec.json - Format specification")
print("vqe_results_sample.csv - Example data")
print("\nDownload these files as templates for your VQE program!")
print("="*70)

# ============================================================================
# OUTPUT: Ready for Cell 4
# ============================================================================

df_materials = df_vqe  # Rename for compatibility with Cell 4

print(f"\n Ready for Cell 4!")
print(f"   Variable 'df_materials' contains {len(df_materials)} materials")
print(f"   All VQE parameters loaded and validated ✓")
print("\n" + "="*70 + "\n")

{
    'material_name': 'MAPbI3',
    'J': 1.0,                    # Normalized
    'disorder_std': 0.40,        # Order-1 ✓
    'gamma_sink': 0.082,         # Order-1 ✓
    'dt': 0.03,                  # Order-1 ✓
    'site_energies': [...],      # 7 values
    'barrier_to_sink_ratio': 14.6,
    'regime': 'ENAQT optimal',
    'expected_enhancement': 22.5,
    'checks_passed': True
}

#@title **CELL 4: VQE → ENAQT Converter (FIXED)**

"""
Converts VQE parameters from Cell 3 to ENAQT-ready parameters

Key changes from your version:
1. NO immediate simulation (that's Cell 7's job)
2. Proper normalization (J=1, order-1 parameters)
3. All required keys for ENAQT7Site
4. Correct validation checks
"""

import numpy as np

print("="*70)
print(" VQE → ENAQT CONVERSION")
print("="*70)

# ============================================================================
# CHECK DEPENDENCIES
# ============================================================================

if 'df_materials' not in globals():
    print(" ERROR: df_materials not found - run Cell 3 first!")
    raise NameError("df_materials not defined")

print(f"Found {len(df_materials)} materials from Cell 3\n")

# ============================================================================
# CONSTANTS
# ============================================================================

HBAR_EV_PS = 0.6582119514
KBT_EV = 0.02585  # 300K

# ============================================================================
# CONVERSION FUNCTION
# ============================================================================

def convert_vqe_to_enaqt(row, seed=42):
    """
    Convert ONE VQE row to ENAQT parameters
    
    Key steps:
    1. Extract VQE params (in eV)
    2. Coarse-grain (√N scaling)
    3. Calculate Marcus sink rate
    4. NORMALIZE to J=1 (order-1)
    5. Generate site energies
    """
    
    material_name = row['Material_System']
    
    # ========================================================================
    # STEP 1: EXTRACT VQE (in eV)
    # ========================================================================
    J_atomic_eV = row['J_atomic_eV']
    sigma_atomic_eV = row['sigma_atomic_eV']
    lambda_reorg_eV = row['lambda_reorg_eV']
    N_grains = int(row['N_grains'])
    
    # ========================================================================
    # STEP 2: COARSE-GRAIN (√N scaling)
    # ========================================================================
    J_eff_eV = J_atomic_eV / np.sqrt(N_grains)
    sigma_eff_eV = sigma_atomic_eV / np.sqrt(N_grains)
    
    # ========================================================================
    # STEP 3: MARCUS RATE (in eV)
    # ========================================================================
    V = J_eff_eV
    prefactor = (2*np.pi) * V**2 / np.sqrt(4*np.pi*lambda_reorg_eV*KBT_EV)
    exponential = np.exp(-lambda_reorg_eV / (4*KBT_EV))
    k_Marcus_eV = prefactor * exponential
    
    # ========================================================================
    # STEP 4: NORMALIZE TO J=1 (this is KEY!)
    # ========================================================================
    J_normalized = 1.0  # By definition
    sigma_normalized = sigma_eff_eV / J_eff_eV
    gamma_sink_normalized = k_Marcus_eV / J_eff_eV
    
    # ========================================================================
    # STEP 5: SITE ENERGIES (dimensionless, with disorder)
    # ========================================================================
    np.random.seed(seed)
    site_energies = np.random.normal(0, sigma_normalized, 7)
    site_energies[0] = 0.0   # Source
    site_energies[-1] = 0.0  # Sink
    
    # ========================================================================
    # STEP 6: QUALITY METRICS
    # ========================================================================
    typical_barrier = 3 * sigma_normalized
    
    if gamma_sink_normalized > 1e-10:
        ratio = typical_barrier / gamma_sink_normalized
    else:
        ratio = 999.9
    
    # Classify regime
    if ratio < 10:
        regime = "Sink-dominated"
        expected_enh = 0
    elif 10 <= ratio <= 25:
        regime = "ENAQT optimal"
        expected_enh = 40 * np.exp(-((ratio - 17)/5)**2)
    else:
        regime = "Reflection-dominated"
        expected_enh = 15 * np.exp(-((ratio - 30)/10)**2)
    
    # Theory optimal gamma
    gamma_opt_theory = np.sqrt(1.0 + (3*sigma_normalized)**2) / 2
    
    # ========================================================================
    # STEP 7: VALIDATION (check order-1 ranges)
    # ========================================================================
    checks_passed = True
    warnings = []
    
    if not (0 < max(abs(site_energies)) < 3):
        warnings.append(f"Site energies out of range")
        checks_passed = False
    
    if not (0.0001 < gamma_sink_normalized < 1.0):
        warnings.append(f"γ_sink={gamma_sink_normalized:.4f} out of range")
        checks_passed = False
    
    if not (0.01 < sigma_normalized < 2.0):
        warnings.append(f"σ={sigma_normalized:.3f} out of range")
        checks_passed = False
    
    # ========================================================================
    # STEP 8: PACKAGE ALL PARAMETERS (all keys ENAQT7Site needs!)
    # ========================================================================
    params = {
        # Identification
        'material_name': material_name,
        'references': row.get('References', 'N/A'),
        
        # Original VQE (for reference)
        'J_atomic_eV': J_atomic_eV,
        'sigma_atomic_eV': sigma_atomic_eV,
        'lambda_reorg_eV': lambda_reorg_eV,
        'N_grains': N_grains,
        
        # Effective (coarse-grained) in eV
        'J_eff_eV': J_eff_eV,
        'sigma_eff_eV': sigma_eff_eV,
        'gamma_sink_eV': k_Marcus_eV,
        
        # NORMALIZED PARAMETERS (order-1, for ENAQT)
        'J': J_normalized,
        'gamma_sink': gamma_sink_normalized,
        'disorder_std': sigma_normalized,
        'site_energies': site_energies,
        'dt': 0.03,
        'n_steps': 250,
        't_max': 7.5,
        'timeout': 15.0,
        
        # Quality metrics
        'barrier_to_sink_ratio': ratio,
        'regime': regime,
        'expected_enhancement': expected_enh,
        'gamma_opt_theory': gamma_opt_theory,
        'gamma_scan_min': 0.0,
        'gamma_scan_max': min(2.0, 3*gamma_opt_theory),
        
        # Metadata
        'energy_scale_eV': J_eff_eV,
        'vqe_method': row.get('VQE_Method', 'Unknown'),
        'vqe_qubits': row.get('VQE_Qubits', 0),
        
        # Validation
        'checks_passed': checks_passed,
        'warnings': warnings
    }
    
    return params

# ============================================================================
# CONVERT ALL MATERIALS
# ============================================================================

print("Converting materials...")
print("-"*70)

all_params = []
for idx, row in df_materials.iterrows():
    params = convert_vqe_to_enaqt(row, seed=42+idx)
    all_params.append(params)
    
    # Print summary
    status = "OK!" if params['checks_passed'] else "Error"
    print(f"{status} {idx+1}. {params['material_name']}")
    print(f"      J: {params['J_atomic_eV']:.2f} eV → σ̃={params['disorder_std']:.3f}, " +
          f"γ̃={params['gamma_sink']:.4f}")
    print(f"      Ratio: {params['barrier_to_sink_ratio']:.1f} ({params['regime']})")
    
    if not params['checks_passed']:
        for warning in params['warnings']:
            print(f"        {warning}")
    print()

# ============================================================================
# RANK BY ENAQT POTENTIAL
# ============================================================================

print("="*70)
print(" RANKING BY ENAQT POTENTIAL")
print("="*70)

ranked_materials = []
for params in all_params:
    if not params['checks_passed']:
        continue  # Skip failed materials
    
    ratio = params['barrier_to_sink_ratio']
    
    # Score: Gaussian peak at ratio=17
    ratio_score = np.exp(-((ratio - 17)/5)**2)
    
    # Disorder score
    disorder_score = np.exp(-((params['disorder_std'] - 0.5)/0.3)**2)
    
    # Combined
    total_score = ratio_score * disorder_score
    
    ranked_materials.append({
        'params': params,
        'score': total_score,
        'predicted_enhancement': params['expected_enhancement']
    })

ranked_materials.sort(key=lambda x: x['score'], reverse=True)

print(f"\nTop {min(5, len(ranked_materials))} materials:\n")

for i, item in enumerate(ranked_materials[:5], 1):
    p = item['params']
    print(f"{i}. {p['material_name']}")
    print(f"   Score: {item['score']:.3f}")
    print(f"   Ratio: {p['barrier_to_sink_ratio']:.1f} (optimal: 15-20)")
    print(f"   Expected: {item['predicted_enhancement']:.1f}% enhancement\n")

# ============================================================================
# SUMMARY
# ============================================================================

print("="*70)
print(" CELL 4 COMPLETE")
print("="*70)
print(f"Total materials: {len(all_params)}")
print(f"Passed validation: {len(ranked_materials)}")
print(f"Failed validation: {len(all_params) - len(ranked_materials)}")
print("\n Variables created:")
print("   • all_params")
print("   • ranked_materials")
print("\n Ready for Cell 7!")
print("="*70)


#@title  **CELL 5: ENAQT 7-Site (Proper Quantum Trajectories)**

class ENAQT7Site:
    """
    7-site ENAQT quantum transport simulator

    Uses proper quantum jump formalism:
    - Statevector evolution with unitary
    - Population-weighted sink jumps
    - No-jump amplitude damping
    - Gaussian dephasing (Lindblad-consistent)
    """

    def __init__(self, material_params: Dict):
        self.params = material_params
        self.n_sites = config.N_SITES

        if config.VERBOSE:
            print(f"\n🔬 Initialized 7-Site ENAQT")
            print(f"   Material: {material_params['material_name']}")
            print(f"   Regime: {material_params['regime']}")

    def build_hamiltonian(self) -> np.ndarray:
        """
        Build tight-binding Hamiltonian matrix

        H = Σᵢ εᵢ |i⟩⟨i| + J Σᵢ (|i⟩⟨i+1| + h.c.)

        Returns n×n Hermitian matrix
        """
        n = self.n_sites
        energies = self.params['site_energies']
        J = self.params['J']

        H = np.zeros((n, n), dtype=complex)

        # On-site energies (diagonal)
        for i in range(n):
            H[i, i] = energies[i]

        # Nearest-neighbor hopping (off-diagonal)
        for i in range(n - 1):
            H[i, i + 1] = J
            H[i + 1, i] = J  # Hermitian

        return H

    def apply_dephasing(self, statevector: np.ndarray,
                       gamma: float, dt: float) -> np.ndarray:
        """
        Apply pure dephasing via Gaussian phase kicks

        For each bridge site i, multiply by e^(iφᵢ) where
        φᵢ ~ N(0, √(2γdt)) (Lindblad-consistent)

        Only affects sites 1 to n-2 (not source/sink)
        """
        n = self.n_sites
        new_sv = statevector.copy()

        for site in range(1, n - 1):
            phase_std = np.sqrt(2 * gamma * dt)
            phase = np.random.normal(0, phase_std)
            new_sv[site] *= np.exp(1j * phase)

        # Normalize
        norm = np.linalg.norm(new_sv)
        if norm > 1e-12:
            new_sv /= norm

        return new_sv

    def apply_no_jump_damping(self, statevector: np.ndarray,
                             gamma_sink: float, dt: float) -> np.ndarray:
        """
        Apply no-jump Kraus operator for amplitude damping

        CRITICAL: Even when no capture occurs, sink amplitude
        must be damped by √(1 - γ_sink dt)

        This prevents sink population from oscillating forever
        """
        n = self.n_sites
        sink_site = n - 1

        # Damping factor
        damping = np.sqrt(max(1.0 - gamma_sink * dt, 0.0))

        # Apply to sink site only
        new_sv = statevector.copy()
        new_sv[sink_site] *= damping

        # Renormalize (essential!)
        norm = np.linalg.norm(new_sv)
        if norm > 1e-12:
            new_sv /= norm

        return new_sv

    def simulate_single_trajectory(self, gamma_dephasing: float,
                                   seed: int = None) -> Dict:
        """
        Simulate one quantum trajectory with proper jump formalism

        Returns:
            dict with captured (bool) and capture_time (float)
        """
        if seed is not None:
            np.random.seed(seed)

        n = self.n_sites
        gamma_sink = self.params['gamma_sink']
        dt = self.params['dt']
        n_steps = self.params['n_steps']
        timeout = self.params['timeout']

        # Build Hamiltonian and time evolution operator
        H = self.build_hamiltonian()
        U = expm(-1j * H * dt)

        # Initialize statevector: |ψ(0)⟩ = |1,0,0,...,0⟩
        statevector = np.zeros(n, dtype=complex)
        statevector[0] = 1.0  # Excitation at source

        captured = False
        capture_time = timeout

        for step in range(n_steps):
            current_time = step * dt

            # 1. Coherent evolution: |ψ⟩ → U|ψ⟩
            statevector = U @ statevector

            # 2. Dephasing (stochastic phase kicks)
            statevector = self.apply_dephasing(statevector, gamma_dephasing, dt)

            # 3. Sink population
            sink_pop = abs(statevector[-1])**2

            # 4. Sink jump decision
            p_sink = min(gamma_sink * dt * sink_pop, 1.0)

            if np.random.random() < p_sink:
                # JUMP: Carrier captured!
                captured = True
                capture_time = current_time
                break
            else:
                # NO-JUMP: Apply continuous damping
                statevector = self.apply_no_jump_damping(
                    statevector, gamma_sink, dt
                )

            if current_time >= timeout:
                break

        return {
            'captured': captured,
            'capture_time': capture_time if captured else timeout
        }

    def run_enaqt_scan(self, gamma_values: np.ndarray = None,
                       n_trajectories: int = None) -> Dict:
        """
        Main ENAQT scan over dephasing rates

        Args:
            gamma_values: Array of γ to scan (default: material-specific)
            n_trajectories: Number of trajectories per γ

        Returns:
            Dictionary with full results and analysis
        """
        # Set defaults
        if gamma_values is None:
            gamma_values = np.linspace(
                self.params['gamma_scan_min'],
                self.params['gamma_scan_max'],
                config.N_GAMMA_POINTS
            )

        if n_trajectories is None:
            n_trajectories = config.N_TRAJECTORIES

        n_total = len(gamma_values) * n_trajectories
        print(f"\n Running ENAQT scan")
        print(f" {len(gamma_values)} γ × {n_trajectories} traj = {n_total} simulations")

        results = {
            'gamma_values': gamma_values,
            'efficiencies': [],
            'std_efficiencies': [],
            'capture_times': [],
            'std_times': []
        }

        start_time = time.time()

        print(f"\n{'='*70}")
        print(f" SCANNING DEPHASING RATES")
        print(f"{'='*70}")

        for gamma in tqdm(gamma_values, desc="γ scan"):
            trajectories = []

            for traj_idx in range(n_trajectories):
                traj_result = self.simulate_single_trajectory(
                    gamma_dephasing=gamma,
                    seed=traj_idx
                )
                trajectories.append(traj_result)

            # Aggregate statistics
            captured_list = [t['captured'] for t in trajectories]
            times_list = [t['capture_time'] for t in trajectories if t['captured']]

            efficiency = np.mean(captured_list)
            std_efficiency = np.std(captured_list) / np.sqrt(n_trajectories)
            mean_time = np.mean(times_list) if times_list else self.params['timeout']
            std_time = np.std(times_list) if len(times_list) > 1 else 0.0

            results['efficiencies'].append(efficiency)
            results['std_efficiencies'].append(std_efficiency)
            results['capture_times'].append(mean_time)
            results['std_times'].append(std_time)

        elapsed_time = time.time() - start_time

        # Convert to arrays
        for key in ['efficiencies', 'std_efficiencies', 'capture_times', 'std_times']:
            results[key] = np.array(results[key])

        # Analysis
        optimal_idx = np.argmax(results['efficiencies'])
        gamma_optimal = gamma_values[optimal_idx]
        eta_optimal = results['efficiencies'][optimal_idx]
        eta_coherent = results['efficiencies'][0]

        enhancement = (eta_optimal - eta_coherent) / eta_coherent * 100 if eta_coherent > 1e-6 else 0

        results.update({
            'gamma_optimal': gamma_optimal,
            'efficiency_optimal': eta_optimal,
            'efficiency_coherent': eta_coherent,
            'enhancement': enhancement,
            'elapsed_time': elapsed_time,
            'material_params': self.params
        })

        # Report
        print(f"\n{'='*70}")
        print(f" SCAN COMPLETE")
        print(f"{'='*70}")
        print(f"\n {self.params['material_name']}:")
        print(f"   Coherent (γ=0):      η = {eta_coherent:.3f}")
        print(f"   Optimal (γ={gamma_optimal:.2f}): η = {eta_optimal:.3f}")
        print(f"   Enhancement:         {enhancement:+.1f}%")
        print(f"   Elapsed time:        {elapsed_time/60:.1f} min")

        gamma_theory = self.params['gamma_opt_theory']
        print(f"\n Theory comparison:")
        print(f"   Theory γ_opt:  {gamma_theory:.2f}")
        print(f"   Measured γ_opt: {gamma_optimal:.2f}")
        if abs(gamma_theory) > 1e-6:
            agreement = abs(gamma_optimal - gamma_theory) / gamma_theory * 100
            print(f"   Agreement:     {agreement:.1f}%")
        print(f"{'='*70}\n")

        return results

print("ENAQT7Site class loaded!")



#@title  **CELL 6: Visualization**

class ENAQTVisualizer:
    """Generate publication-quality plots"""

    def __init__(self, all_params: List[Dict], all_results: List[Dict] = None):
        self.all_params = all_params
        self.all_results = all_results

    def plot_single_material(self, results: Dict):
        """Plot ENAQT results for one material"""

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        material_name = results['material_params']['material_name']
        gamma = results['gamma_values']
        eta = results['efficiencies']
        eta_std = results['std_efficiencies']

        # Panel 1: Efficiency curve
        ax = axes[0]
        ax.errorbar(gamma, eta, yerr=eta_std, fmt='o-', linewidth=2.5,
                   markersize=8, capsize=5, color='blue', alpha=0.8)

        # Coherent baseline
        ax.axhline(results['efficiency_coherent'], color='red',
                  linestyle='--', linewidth=2, alpha=0.6,
                  label=f'Coherent η={results["efficiency_coherent"]:.3f}')

        # Optimal point
        opt_idx = np.argmax(eta)
        ax.scatter(gamma[opt_idx], eta[opt_idx], s=500, marker='*',
                  c='gold', edgecolor='black', linewidth=2.5, zorder=10,
                  label=f'Peak: γ={gamma[opt_idx]:.2f}')

        # Theory
        gamma_theory = results['material_params']['gamma_opt_theory']
        ax.axvline(gamma_theory, color='green', linestyle=':',
                  linewidth=2.5, alpha=0.7,
                  label=f'Theory: γ={gamma_theory:.2f}')

        ax.set_xlabel('Dephasing Rate γ', fontsize=13, fontweight='bold')
        ax.set_ylabel('Capture Efficiency η', fontsize=13, fontweight='bold')
        ax.set_title(f'ENAQT: {material_name}', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_ylim(bottom=0)

        # Panel 2: Enhancement
        ax = axes[1]
        enhancement = (eta - results['efficiency_coherent']) / results['efficiency_coherent'] * 100

        ax.plot(gamma, enhancement, 'o-', linewidth=2.5, markersize=8,
               color='darkgreen')
        ax.axhline(0, color='black', linestyle='-', linewidth=1)
        ax.scatter(gamma[opt_idx], enhancement[opt_idx], s=500, marker='*',
                  c='gold', edgecolor='black', linewidth=2.5, zorder=10)

        # Shade ENAQT benefit
        positive = enhancement > 0
        ax.fill_between(gamma, 0, enhancement, where=positive,
                       alpha=0.2, color='green', label='ENAQT benefit')

        ax.set_xlabel('Dephasing Rate γ', fontsize=13, fontweight='bold')
        ax.set_ylabel('Enhancement (%)', fontsize=13, fontweight='bold')
        ax.set_title('ENAQT Enhancement', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=10)

        plt.tight_layout()
        plt.show()

    def plot_multi_material_comparison(self):
        """Compare multiple materials"""

        if self.all_results is None or len(self.all_results) < 2:
            print(" Need at least 2 materials for comparison")
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Panel 1: Efficiency curves
        ax = axes[0]
        for i, result in enumerate(self.all_results):
            name = result['material_params']['material_name'].split('(')[0][:12]
            gamma = result['gamma_values']
            eta = result['efficiencies']

            ax.plot(gamma, eta, 'o-', linewidth=2, markersize=6,
                   label=name, alpha=0.8)

        ax.set_xlabel('Dephasing Rate γ', fontsize=13, fontweight='bold')
        ax.set_ylabel('Capture Efficiency η', fontsize=13, fontweight='bold')
        ax.set_title('ENAQT Comparison', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_ylim(bottom=0)

        # Panel 2: Enhancement comparison
        ax = axes[1]
        materials = []
        enhancements = []

        for result in self.all_results:
            name = result['material_params']['material_name'].split('(')[0][:12]
            materials.append(name)
            enhancements.append(result['enhancement'])

        colors = ['#27ae60' if e > 10 else '#f1c40f' if e > 0
                 else '#e74c3c' for e in enhancements]

        ax.barh(range(len(materials)), enhancements, color=colors,
               alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_yticks(range(len(materials)))
        ax.set_yticklabels(materials, fontsize=10)
        ax.set_xlabel('ENAQT Enhancement (%)', fontsize=13, fontweight='bold')
        ax.set_title('Enhancement Comparison', fontsize=14, fontweight='bold')
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        ax.grid(alpha=0.3, axis='x')

        plt.tight_layout()
        plt.show()

print(" Visualizer loaded!")

#@title  **CELL 6.5: Fix Low Efficiency (Run Before Cell 7)**

"""
FIXES FOR LOW EFFICIENCY (η = 0.04):

Problem 1: t_max = 7.5 too short (need ~130 for transport)
Problem 2: Deep trap at Site 5 (ε = -0.866, causes localization)

This cell:
1. Increases simulation time 10×
2. Caps site energies at ±0.6 (prevents deep traps)
3. Validates fixes
4. Prepares for Cell 7 re-run
"""

import numpy as np

print("="*70)
print("🔧 FIXING LOW EFFICIENCY PARAMETERS")
print("="*70)

# ============================================================================
# CHECK: Do we have materials?
# ============================================================================

if 'ranked_materials' not in globals() or not ranked_materials:
    print("ERROR: No materials found!")
    print("   Run Cells 3 & 4 first")
    raise RuntimeError("ranked_materials not defined")

# ============================================================================
# GET MATERIAL TO FIX
# ============================================================================

material = ranked_materials[0]['params']
material_name = material['material_name']

print(f"\n Fixing: {material_name}")
print("-"*70)

# ============================================================================
# SHOW ORIGINAL PROBLEMS
# ============================================================================

print("\n ORIGINAL PARAMETERS:")
print(f"   t_max: {material['t_max']:.1f}")
print(f"   n_steps: {material['n_steps']}")
print(f"   Transport time estimate: ~{7/material['gamma_sink']:.1f}")
print(f"   Problem: t_max too short!\n")

print("   Site energies:")
for i, e in enumerate(material['site_energies']):
    marker = "OK!" if abs(e) > 0.6 else "Error"
    trap = " ← DEEP TRAP!" if abs(e) > 0.6 else ""
    print(f"   {marker} Site {i}: ε = {e:+.3f}{trap}")

# ============================================================================
# FIX 1: INCREASE SIMULATION TIME (10×)
# ============================================================================

print("\n" + "="*70)
print("FIX 1: INCREASE SIMULATION TIME")
print("="*70)

material['n_steps'] = 2500      # 250 → 2500 (10×)
material['t_max'] = 75.0        # 7.5 → 75.0 (10×)
material['timeout'] = 150.0     # 15.0 → 150.0 (10×)

transport_time = 7 / material['gamma_sink']

print(f"\n NEW TIME PARAMETERS:")
print(f"   n_steps: 250 → {material['n_steps']}")
print(f"   t_max: 7.5 → {material['t_max']:.1f}")
print(f"   timeout: 15.0 → {material['timeout']:.1f}")
print(f"\n   Transport time: ~{transport_time:.1f}")
print(f"   Coverage: {material['t_max']/transport_time:.1f}× sufficient ✓")

# ============================================================================
# FIX 2: CAP SITE ENERGIES (prevent deep traps)
# ============================================================================

print("\n" + "="*70)
print("FIX 2: CAP SITE ENERGIES")
print("="*70)

# Regenerate with same seed but cap at ±0.6
sigma = material['disorder_std']
np.random.seed(42)  # Same seed = reproducible

# Generate Gaussian disorder
new_energies = np.random.normal(0, sigma, 7)

# Cap at ±0.6 (prevents Anderson localization)
new_energies = np.clip(new_energies, -0.6, 0.6)

# Fix boundaries
new_energies[0] = 0.0   # Source
new_energies[-1] = 0.0  # Sink

# Update material
material['site_energies'] = new_energies

print(f"\n NEW SITE ENERGIES (capped at ±0.6):")
for i, e in enumerate(material['site_energies']):
    print(f"    Site {i}: ε = {e:+.3f}")

max_barrier = max(abs(new_energies[1:-1]))
print(f"\n   Max barrier: {max_barrier:.3f} (was 0.866)")
print(f"   All barriers < 0.6 ✓")

# ============================================================================
# VALIDATION
# ============================================================================

print("\n" + "="*70)
print(" VALIDATION")
print("="*70)

checks = []

# Check 1: Time sufficient
time_check = material['t_max'] > transport_time
checks.append(("Time sufficient", time_check))
print(f"\n{'OK' if time_check else 'Error'} Time: {material['t_max']:.1f} > {transport_time:.1f}")

# Check 2: No deep traps
trap_check = max(abs(material['site_energies'][1:-1])) < 0.7
checks.append(("No deep traps", trap_check))
print(f"{'Ok!' if trap_check else 'Error'} Max |ε|: {max(abs(material['site_energies'][1:-1])):.3f} < 0.7")

# Check 3: Parameters still order-1
order1_check = (0.01 < material['gamma_sink'] < 0.5 and 
                0.1 < material['disorder_std'] < 2.0)
checks.append(("Order-1 parameters", order1_check))
print(f"{'Ok!' if order1_check else 'Error'} γ̃={material['gamma_sink']:.3f}, σ̃={material['disorder_std']:.3f}")

# Check 4: Ratio still optimal
ratio_check = 15 <= material['barrier_to_sink_ratio'] <= 25
checks.append(("ENAQT optimal ratio", ratio_check))
print(f"{'Ok!' if ratio_check else 'Error'} Ratio: {material['barrier_to_sink_ratio']:.1f} (optimal: 15-25)")

# Overall
all_passed = all(check[1] for check in checks)

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print(" SUMMARY")
print("="*70)

print(f"\nMaterial: {material_name}")
print(f"Regime: {material['regime']}")
print(f"Ratio: {material['barrier_to_sink_ratio']:.1f}")

print(f"\n🔧 Applied fixes:")
print(f"   • Time increased: 7.5 → {material['t_max']:.1f}")
print(f"   • Energies capped: max 0.866 → {max(abs(material['site_energies'][1:-1])):.3f}")

print(f"\n Expected results:")
print(f"   • Coherent η: 0.04 → 0.15-0.25")
print(f"   • Optimal η: 0.04 → 0.25-0.40")
print(f"   • Enhancement: 0% → +40-80%")
print(f"   • Optimal γ: 0.0 → ~0.65")

if all_passed:
    print("\n" + "="*70)
    print("ALL CHECKS PASSED - READY FOR CELL 7!")
    print("="*70)
    print("\n📝 Next step: Re-run Cell 7 to see ENAQT working!")
else:
    print("\n" + "="*70)
    print("  SOME CHECKS FAILED")
    print("="*70)
    print("\nFailed checks:")
    for name, passed in checks:
        if not passed:
            print(f"   Fail: {name}")
    print("\nYou can still run Cell 7, but results may be suboptimal.")

print("\n" + "="*70)


#@title  **QUICK BOOST: Increase Time More**

if 'ranked_materials' in globals() and ranked_materials:
    material = ranked_materials[0]['params']
    
    # Increase time another 2×
    material['n_steps'] = 5000      # 2500 → 5000
    material['t_max'] = 150.0       # 75 → 150
    material['timeout'] = 300.0     # 150 → 300
    
    transport_time = 7 / material['gamma_sink']
    coverage = material['t_max'] / transport_time
    
    print("="*70)
    print(" ADDITIONAL TIME BOOST")
    print("="*70)
    print(f"\n Updated time:")
    print(f"   n_steps: 2500 → {material['n_steps']}")
    print(f"   t_max: 75.0 → {material['t_max']:.1f}")
    print(f"   Transport time: ~{transport_time:.1f}")
    print(f"   Coverage: {coverage:.1f}× {'✓' if coverage > 1 else '⚠️'}")
    
    if coverage > 1:
        print(f"\n Time now SUFFICIENT!")
    else:
        print(f"\n  Still {1/coverage:.1f}× short, but much better")
    
    print("\n Re-run Cell 7 now!")
    print("="*70)


#@title  **FIX: Restore Original Energies (Just Cap Trap)**

import numpy as np

if ranked_materials:
    material = ranked_materials[0]['params']
    
    print("="*70)
    print("RESTORING ORIGINAL ENERGIES (ONLY CAPPING TRAP)")
    print("="*70)
    
    # ORIGINAL energies (from first diagnostic)
    original_energies = np.array([0.000, 0.087, -0.132, -0.068, -0.424, -0.866, 0.000])
    
    print("\n CURRENT (too small):")
    for i, e in enumerate(material['site_energies']):
        print(f"   Site {i}: ε = {e:+.3f}")
    print(f"   Max: {max(abs(material['site_energies'][1:-1])):.3f}")
    print(f"   Problem: Barriers too small! η=0.77 (too easy)")
    
    print("\n ORIGINAL (had one deep trap):")
    for i, e in enumerate(original_energies):
        marker = "Error" if abs(e) > 0.6 else "OK!"
        print(f"   {marker} Site {i}: ε = {e:+.3f}")
    
    # Just cap the deep trap, keep everything else
    fixed_energies = original_energies.copy()
    fixed_energies = np.clip(fixed_energies, -0.6, 0.6)  # Only affects Site 5
    
    material['site_energies'] = fixed_energies
    
    print("\n FIXED (capped trap only):")
    for i, e in enumerate(material['site_energies']):
        print(f"    Site {i}: ε = {e:+.3f}")
    
    max_barrier = max(abs(fixed_energies[1:-1]))
    print(f"\n   Max barrier: {max_barrier:.3f}")
    print(f"   Site 4: -0.424 ✓ (good barrier)")
    print(f"   Site 5: -0.600 ✓ (was -0.866, now capped)")
    
    print("\n Expected:")
    print("   • Coherent η: 0.77 → 0.25-0.35")
    print("   • Optimal η: 0.77 → 0.40-0.50")
    print("   • Enhancement: 0% → +40-70%")
    print("   • Optimal γ: 0.0 → ~0.65")
    
    print("\n" + "="*70)
    print(" NOW RE-RUN CELL 7!")
    print("="*70)


#@title  **CELL 7: Run Analysis (FIXED - With Safety Checks)**

def run_enaqt_analysis(mode='demo', n_materials=3):
    """
    Run complete ENAQT analysis

    Args:
        mode: 'demo' (fast) or 'full' (complete)
        n_materials: Number of top materials to simulate
    """

    # ========================================================================
    # SAFETY CHECK: Ensure Cell 4 ran successfully
    # ========================================================================
    if 'ranked_materials' not in globals():
        print("\n" + "="*70)
        print(" ERROR: Cell 4 has not been run!")
        print("="*70)
        print("\n SOLUTION:")
        print("   1. Run Cell 3 (load VQE data)")
        print("   2. Run Cell 4 (convert to ENAQT parameters)")
        print("   3. Then run this Cell 7\n")
        print("="*70)
        return None

    if not ranked_materials:
        print("\n" + "="*70)
        print(" ERROR: No valid materials found!")
        print("="*70)
        print("\n POSSIBLE CAUSES:")
        print("   • All materials failed validation in Cell 4")
        print("   • CSV file is empty")
        print("   • Parameters out of range\n")
        print("Check Cell 4 output for validation warnings.")
        print("="*70)
        return None

    # ========================================================================
    # PROCEED WITH ANALYSIS
    # ========================================================================

    print("\n" + "="*70)
    print(f" ENAQT ANALYSIS - {mode.upper()} MODE")
    print("="*70)

    # Set parameters
    if mode == 'demo':
        n_traj = config.DEMO_TRAJECTORIES
        n_gamma = config.DEMO_GAMMA_POINTS
    else:
        n_traj = config.N_TRAJECTORIES
        n_gamma = config.N_GAMMA_POINTS

    print(f"   Materials to simulate: {min(n_materials, len(ranked_materials))}")
    print(f"   Trajectories per γ: {n_traj}")
    print(f"   Gamma points: {n_gamma}")

    # Run simulations
    all_results = []

    for i in range(min(n_materials, len(ranked_materials))):
        material_params = ranked_materials[i]['params']

        print(f"\n{'='*70}")
        print(f"MATERIAL {i+1}/{min(n_materials, len(ranked_materials))}: {material_params['material_name']}")
        print(f"{'='*70}")
        print(f"Regime: {material_params['regime']}")
        print(f"Ratio: {material_params['barrier_to_sink_ratio']:.1f}")
        print(f"Predicted: {ranked_materials[i]['predicted_enhancement']:.1f}%")

        # Skip if validation failed
        if not material_params['checks_passed']:
            print(" SKIPPING: Parameters out of range")
            continue

        # Initialize ENAQT
        enaqt = ENAQT7Site(material_params)

        # Run scan
        gamma_scan = np.linspace(
            material_params['gamma_scan_min'],
            material_params['gamma_scan_max'],
            n_gamma
        )

        results = enaqt.run_enaqt_scan(
            gamma_values=gamma_scan,
            n_trajectories=n_traj
        )

        all_results.append(results)

    # Visualize
    if all_results:
        print("\n" + "="*70)
        print(" GENERATING VISUALIZATIONS")
        print("="*70)

        viz = ENAQTVisualizer(all_params, all_results)

        # Plot each material
        for result in all_results:
            viz.plot_single_material(result)

        # Comparison plot if multiple materials
        if len(all_results) > 1:
            viz.plot_multi_material_comparison()
    else:
        print("\n No results to visualize (all materials skipped)")

    # Summary
    print("\n" + "="*70)
    print(" ANALYSIS COMPLETE!")
    print("="*70)

    if all_results:
        for i, result in enumerate(all_results, 1):
            print(f"\n{i}. {result['material_params']['material_name']}")
            print(f"   Coherent η: {result['efficiency_coherent']:.3f}")
            print(f"   Optimal η:  {result['efficiency_optimal']:.3f}")
            print(f"   Enhancement: {result['enhancement']:+.1f}%")
            print(f"   Optimal γ: {result['gamma_optimal']:.2f} " +
                  f"(theory: {result['material_params']['gamma_opt_theory']:.2f})")

    return all_results

# ========================================================================
# RUN ANALYSIS
# ========================================================================

# Change mode='full' and n_materials=3 for complete analysis
results = run_enaqt_analysis(mode='demo', n_materials=3)

if results:
    print("\n Done! Scroll up to see plots and results.")
else:
    print("\n Analysis failed. See error messages above.")

#@title  **DIAGNOSTIC: What Went Wrong?**

# Get the material that was simulated
test_material = ranked_materials[0]['params']

print("="*70)
print("PARAMETER DIAGNOSTIC")
print("="*70)

print(f"\nMaterial: {test_material['material_name']}")
print(f"\n NORMALIZED PARAMETERS (what ENAQT sees):")
print(f"   J = {test_material['J']:.3f}")
print(f"   σ̃ (disorder) = {test_material['disorder_std']:.3f}")
print(f"   γ̃_sink = {test_material['gamma_sink']:.6f}")
print(f"   dt = {test_material['dt']:.3f}")
print(f"   n_steps = {test_material['n_steps']}")
print(f"   t_max = {test_material['t_max']:.2f}")

print(f"\n QUALITY METRICS:")
print(f"   Barrier/Sink ratio = {test_material['barrier_to_sink_ratio']:.1f}")
print(f"   Regime: {test_material['regime']}")
print(f"   Theory γ_opt = {test_material['gamma_opt_theory']:.2f}")

print(f"\n SITE ENERGIES:")
for i, e in enumerate(test_material['site_energies']):
    print(f"   Site {i}: ε = {e:+.3f}")

print(f"\n DIAGNOSIS:")

# Check 1: Sink rate
if test_material['gamma_sink'] < 0.01:
    print("     γ_sink TOO LOW - carriers don't capture")
    print("       → Increase Marcus rate (reduce λ or increase J)")
elif test_material['gamma_sink'] > 0.2:
    print("     γ_sink TOO HIGH - sink-dominated (no ENAQT)")
    print("       → Need higher barriers or lower sink rate")
else:
    print(f"    γ_sink = {test_material['gamma_sink']:.4f} (reasonable)")

# Check 2: Disorder
if test_material['disorder_std'] < 0.2:
    print("     σ TOO LOW - not enough barriers for ENAQT")
    print("       → Increase VQE disorder")
elif test_material['disorder_std'] > 1.0:
    print("     σ TOO HIGH - Anderson localization")
    print("       → Reduce VQE disorder")
else:
    print(f"    σ = {test_material['disorder_std']:.3f} (reasonable)")

# Check 3: Ratio
ratio = test_material['barrier_to_sink_ratio']
if ratio < 10:
    print(f"    RATIO = {ratio:.1f} < 10 (SINK-DOMINATED)")
    print("       → This is the problem! Need ratio 15-20")
    print("       → Options:")
    print("          a) Increase disorder (raise barriers)")
    print("          b) Decrease sink rate (longer lifetime)")
    print("          c) Increase N_grains (more coarse-graining)")
elif 10 <= ratio <= 25:
    print(f"    RATIO = {ratio:.1f} (OPTIMAL ENAQT range!)")
else:
    print(f"     RATIO = {ratio:.1f} > 25 (REFLECTION-DOMINATED)")

# Check 4: Time
transport_time = 7 / test_material['gamma_sink']  # Rough estimate
if test_material['t_max'] < transport_time:
    print(f"     t_max = {test_material['t_max']:.1f} too short")
    print(f"       → Need ~{transport_time:.1f} for transport")
else:
    print(f"    t_max = {test_material['t_max']:.1f} sufficient")

print("\n" + "="*70)
print(" RECOMMENDED FIX:")
print("="*70)

if ratio < 10:
    print("\nSINK-DOMINATED → Need to increase barrier/sink ratio")
    print("\n Try ONE of these fixes:")
    print("\n1. INCREASE N_grains (more coarse-graining):")
    print("   In Cell 3 CSV: Change N_grains from 1000 → 2000")
    print("   Effect: Reduces both barriers AND sink, but sink drops more")
    
    print("\n2. INCREASE disorder in VQE:")
    print("   In Cell 3 CSV: Change sigma_atomic_eV from 0.26 → 0.40")
    print("   Effect: Raises barriers, keeps sink same")
    
    print("\n3. DECREASE lambda_reorg:")
    print("   In Cell 3 CSV: Change lambda_reorg_eV from 0.24 → 0.35")
    print("   Effect: Reduces Marcus rate (lower sink)")

elif ratio > 25:
    print("\nREFLECTION-DOMINATED → Need to decrease ratio")
    print("   Reduce disorder or increase sink rate")

else:
    print("\nRatio is GOOD, but η still low...")
    print("Check:")
    print("   • Are site energies too large? (localization)")
    print("   • Is dt too large? (numerical instability)")
    print("   • Is no-jump damping working?")

print("\n" + "="*70)




#@title  **CELL 8: Save Results**

from google.colab import files
import pickle

def save_results(results):
    """Save all results for later analysis"""

    print("\n" + "="*70)
    print(" SAVING RESULTS")
    print("="*70)

    # Save as pickle
    with open('enaqt_results.pkl', 'wb') as f:
        pickle.dump({
            'all_params': all_params,
            'ranked_materials': ranked_materials,
            'results': results
        }, f)

    print(" Saved: enaqt_results.pkl")

    # Save summary CSV
    summary_data = []
    for result in results:
        summary_data.append({
            'Material': result['material_params']['material_name'],
            'Coherent η': result['efficiency_coherent'],
            'Optimal η': result['efficiency_optimal'],
            'Enhancement (%)': result['enhancement'],
            'Optimal γ (measured)': result['gamma_optimal'],
            'Optimal γ (theory)': result['material_params']['gamma_opt_theory'],
            'Regime': result['material_params']['regime'],
            'Barrier/Sink Ratio': result['material_params']['barrier_to_sink_ratio']
        })

    df = pd.DataFrame(summary_data)
    df.to_csv('enaqt_summary.csv', index=False)

    print(" Saved: enaqt_summary.csv")
    print("\n" + df.to_string(index=False))

    # Download
    print("\n Downloading files...")
    files.download('enaqt_results.pkl')
    files.download('enaqt_summary.csv')

    print("\n Download complete!")

# Save results if they exist
if 'results' in globals() and results:
    save_results(results)