import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, repo_root)

from benchmarks.freefall_comparison.mpm99_materials import MATERIALS, mpm99_properties


def run_benchmark(module, arch, output_directory, export_vtk, export_every, steps=None, particle_count=None, material="liquid"):
    command = [
        sys.executable,
        "-m",
        module,
        "--arch",
        arch,
        "--output",
        output_directory,
        "--export-every",
        str(export_every),
        "--material",
        material,
    ]
    if steps is not None:
        command.extend(["--steps", str(steps)])
    if particle_count is not None:
        command.extend(["--particles", str(particle_count)])
    if not export_vtk:
        command.append("--no-export")
    environment = os.environ.copy()
    environment.setdefault("TI_LOG_LEVEL", "error")
    label = module.rsplit(".", 2)[-2]
    if particle_count is not None:
        label += f" ({particle_count} particles)"
    print(f"Starting {label}", flush=True)
    wall_start = time.perf_counter()
    process = subprocess.Popen(command, cwd=repo_root, env=environment, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    while True:
        try:
            stdout, stderr = process.communicate(timeout=10)
            break
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - wall_start
            print(f"  {label} is running; elapsed {elapsed:.1f} s", flush=True)
    process_wall_seconds = time.perf_counter() - wall_start
    if process.returncode != 0:
        raise RuntimeError(
            f"{' '.join(command)} failed with code {process.returncode}\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )
    result = None
    for line in reversed(stdout.splitlines()):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "timing" in candidate:
            result = candidate
            break
    if result is None:
        raise RuntimeError(f"benchmark did not emit a JSON result:\n{stdout}")
    result["timing"]["process_wall_seconds"] = process_wall_seconds
    print(
        f"Finished {label}: simulation {result['timing']['simulation_seconds']:.2f} s, "
        f"{result['timing']['steps_per_second']:.1f} steps/s",
        flush=True,
    )
    return result


def comparison_against(legacy, adaptive):
    legacy_sim = legacy["timing"]["simulation_seconds"]
    adaptive_sim = adaptive["timing"]["simulation_seconds"]
    legacy_final = legacy["final"]
    adaptive_final = adaptive["final"]
    center_difference = np.asarray(adaptive_final["center_of_mass"]) - np.asarray(legacy_final["center_of_mass"])
    return {
        "performance": {
            "legacy_simulation_seconds": legacy_sim,
            "adaptive_simulation_seconds": adaptive_sim,
            "legacy_jit_warmup_seconds": legacy["timing"]["jit_warmup_seconds"],
            "adaptive_jit_warmup_seconds": adaptive["timing"]["jit_warmup_seconds"],
            "legacy_steps_per_second": legacy["timing"]["steps_per_second"],
            "adaptive_steps_per_second": adaptive["timing"]["steps_per_second"],
            "adaptive_to_legacy_time_ratio": adaptive_sim / legacy_sim,
            "legacy_speedup_over_adaptive": adaptive_sim / legacy_sim,
            "legacy_export_seconds": legacy["timing"]["export_seconds"],
            "adaptive_export_seconds": adaptive["timing"]["export_seconds"],
            "legacy_process_wall_seconds": legacy["timing"]["process_wall_seconds"],
            "adaptive_process_wall_seconds": adaptive["timing"]["process_wall_seconds"],
        },
        "particle_population": {
            "legacy_initial": legacy["initial"]["particles"],
            "legacy_final": legacy_final["particles"],
            "adaptive_initial": adaptive["initial"]["particles"],
            "adaptive_peak": adaptive["peak_particles"],
            "adaptive_final": adaptive_final["particles"],
            "adaptive_peak_by_level": adaptive["peak_particles_by_level"],
            "adaptive_final_by_level": adaptive_final["particles_by_level"],
        },
        "final_state": {
            "legacy": legacy_final,
            "adaptive": adaptive_final,
            "center_of_mass_difference_adaptive_minus_legacy": center_difference.tolist(),
            "center_of_mass_difference_norm": float(np.linalg.norm(center_difference)),
            "top_height_difference": adaptive_final["height_range"][1] - legacy_final["height_range"][1],
            "kinetic_energy_difference": adaptive_final["kinetic_energy"] - legacy_final["kinetic_energy"],
            "max_speed_difference": adaptive_final["max_speed"] - legacy_final["max_speed"],
            "max_pressure_difference": adaptive_final["max_pressure"] - legacy_final["max_pressure"],
        },
    }


def build_comparison(legacy_100, legacy_1500, adaptive, material="liquid"):
    material_props = mpm99_properties(material)
    return {
        "controlled_parameters": {
            "physical_time": adaptive["time"],
            "steps": adaptive["steps"],
            "dt": adaptive["dt"],
            "adaptive_base_dx": adaptive["base_dx"],
            "adaptive_base_grid_cells": adaptive["grid_cells"],
            "legacy_100_dx": legacy_100["base_dx"],
            "legacy_100_grid_cells": legacy_100["grid_cells"],
            "legacy_1500_dx": legacy_1500["base_dx"],
            "legacy_1500_grid_cells": legacy_1500["grid_cells"],
            "gravity": [0.0, -9.81],
            "grid_damping": 1.0,
            "material": material,
            "material_properties": material_props,
            "legacy_particle_populations": [100, 1500],
            "adaptive_initial_particles": adaptive["initial"]["particles"],
        },
        "method_differences": {
            "legacy": {
                "solver": legacy_100["solver"],
                "precision": legacy_100["precision"],
                "refinement": legacy_100["refinement"],
                "boundary_method": legacy_100["boundary_method"],
                "constitutive_update": legacy_100["constitutive_update"],
                "material_properties": legacy_100["liquid_properties"],
            },
            "adaptive": {
                "solver": adaptive["solver"],
                "precision": adaptive["precision"],
                "refinement": adaptive["refinement"],
                "boundary_method": adaptive["boundary_method"],
                "constitutive_update": adaptive["constitutive_update"],
                "material_properties": adaptive["liquid_properties"],
            },
        },
        "material_property_consistency_check": {
            "legacy_100_matches_shared": legacy_100["liquid_properties"] == material_props,
            "legacy_1500_matches_shared": legacy_1500["liquid_properties"] == material_props,
            "adaptive_matches_shared": adaptive["liquid_properties"] == material_props,
        },
        "legacy_100_vs_adaptive": comparison_against(legacy_100, adaptive),
        "legacy_1500_vs_adaptive": comparison_against(legacy_1500, adaptive),
        "interpretation": (
            f"All runs use the same float64 MPM transfer, the 99_bench '{material}' constitutive update, "
            "weak penalty boundary, domain, timestep, material properties, gravity, and physical duration. "
            "Legacy-100 uses the adaptive base grid and matches its initial population. Legacy-1500 uses a "
            "uniform grid at the adaptive finest spacing and approximates the adaptive peak population for "
            "the complete run. The comparison measures fixed coarse, fixed fine, and gradient-adaptive "
            "discretizations."
        ),
        "legacy_100_result": legacy_100,
        "legacy_1500_result": legacy_1500,
        "adaptive_result": adaptive,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--material", choices=MATERIALS, default="liquid")
    vtk_group = parser.add_mutually_exclusive_group()
    vtk_group.add_argument("--with-vtk", dest="with_vtk", action="store_true")
    vtk_group.add_argument("--no-vtk", dest="with_vtk", action="store_false")
    parser.set_defaults(with_vtk=True)
    parser.add_argument("--export-every", type=int, default=500)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    output_directory = args.output or os.path.join(os.path.dirname(__file__), "output")
    legacy_100_output = os.path.join(output_directory, "freefall_old_100")
    legacy_1500_output = os.path.join(output_directory, "freefall_old_1500")
    adaptive_output = os.path.join(output_directory, "freefall")
    for directory in (legacy_100_output, legacy_1500_output, adaptive_output):
        os.makedirs(directory, exist_ok=True)
    print(
        f"Running three freefall benchmarks on {args.arch} with material '{args.material}': "
        f"legacy 100, legacy 1500, adaptive",
        flush=True,
    )
    if args.with_vtk:
        print(f"VTK export enabled every {args.export_every} steps", flush=True)
    else:
        print("VTK export disabled for solver timing", flush=True)
    legacy_100 = run_benchmark(
        "benchmarks.freefall_old.run_experiment",
        args.arch,
        legacy_100_output,
        args.with_vtk,
        args.export_every,
        args.steps,
        100,
        args.material,
    )
    legacy_1500 = run_benchmark(
        "benchmarks.freefall_old.run_experiment",
        args.arch,
        legacy_1500_output,
        args.with_vtk,
        args.export_every,
        args.steps,
        1500,
        args.material,
    )
    adaptive = run_benchmark(
        "benchmarks.freefall.run_experiment",
        args.arch,
        adaptive_output,
        args.with_vtk,
        args.export_every,
        args.steps,
        None,
        args.material,
    )
    for legacy in (legacy_100, legacy_1500):
        if legacy["steps"] != adaptive["steps"] or legacy["dt"] != adaptive["dt"]:
            raise AssertionError("comparison benchmarks did not use matching steps and timestep")
    comparison = build_comparison(legacy_100, legacy_1500, adaptive, args.material)
    output_path = os.path.join(output_directory, "comparison_results.json")
    with open(output_path, "w", encoding="utf-8") as output:
        json.dump(comparison, output, indent=2)
    print(json.dumps(comparison, sort_keys=True))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
