import torch
import requests
import numpy as np
from mp_api.client import MPRester
import nvalchemiops.neighborlist as nl
import nvalchemi-toolkit-ops # Nvidia Ops Kit

# 1. Setup API Keys and Endpoints
MP_API_KEY = "your_materials_project_key"
NIM_ENDPOINT = "http://localhost:8000/v1/relax"


def get_steel_base():
    """Fetches a standard Austenite (304-like) structure from Materials Project."""
    with MPRester(MP_API_KEY) as mpr:
        # Searching for Fe-Cr-Ni (304 Stainless base)
        docs = mpr.summary.search(elements=["Fe", "Cr", "Ni"], num_elements=3)
        if not docs:
            raise ValueError("No structures found for Fe-Cr-Ni")
        return docs[0].structure


def calculate_attenuation(material_structure, energy_kev=50):
    """
    Calculates X-ray attenuation based on Z and density.
    Formula: mu/rho ~ Z^4 / E^3
    """
    # Mass attenuation coefficients (simplified placeholders or use XrayLib)
    z_map = {"Fe": 26, "Cr": 24, "Ni": 28, "Hf": 72}
    atoms = [site.specie.symbol for site in material_structure]
    avg_z = np.mean([z_map.get(a, 26) for a in atoms])

    density = material_structure.density  # g/cm^3
    # Approximation of mass attenuation coefficient
    mass_attenuation = (avg_z ** 4) / (energy_kev ** 3) * 0.01
    linear_attenuation = mass_attenuation * density
    return linear_attenuation


# 2. Structure Modification (Doping Hafnium)
structure = get_steel_base()
# Replace a percentage of Fe with Hf (Example: 10%)
for i in range(0, len(structure), 10):
    structure.replace(i, "Hf")

# 3. Utilize Nvidia Alchemi Ops Kit (Neighbor List for Local Analysis)
device = "cuda" if torch.cuda.is_available() else "cpu"
positions = torch.tensor(structure.cart_coords, device=device, dtype=torch.float32)
cell = torch.tensor(structure.lattice.matrix, device=device, dtype=torch.float32)
# Fast neighbor list construction for stability checks
indices, offsets = nl.neighbor_list(positions, cell, cutoff=5.0)

# 4. Connect to Nvidia NIM for Geometry Relaxation
# This step ensures the Hf percentage is physically stable
payload = {
    "structure": structure.as_dict(),
    "model": "mace-mp-0",
    "fmax": 0.05
}
# response = requests.post(NIM_ENDPOINT, json=payload) # Uncomment when NIM is running
# relaxed_structure = response.json()['relaxed_structure']

# 5. Output Results
attn_value = calculate_attenuation(structure)
print(f"Hafnium-Doped Steel Density: {structure.density:.2f} g/cm3")
print(f"Predicted Linear Attenuation at 50keV: {attn_value:.4f} cm^-1")