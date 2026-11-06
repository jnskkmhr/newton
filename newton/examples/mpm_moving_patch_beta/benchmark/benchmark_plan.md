Now, let's try benchmark with
* fixed grid 
* sparse grid
* moving patch (mpm_moving_patch_beta)

## What are questions to figure out from benchmark?

1. Does moving patch scale to large terrain size? 
-> Demonstrate our approach for large scale terrain simulation
* Define operation space like 3mx3m to 10mx10m with 1m increments bin
* Moving patch grid only defines active MPM grid with default parameters and you keep moving grid around robot moving in this operating space. 
* Btw, I think there is no doubt that moving patch works better. 

2. Does moving patch have some overhead comapred to without moving patch?
-> Demonstrate that our approach does not slow down computation time for small scale environment too
* Define small scale environment like 1mx1mx0.5/2mx2mx0.5/4mx4mx0.5m
* Simulate with dense, sparse, moving patch 
* See if moving patch method has some computation overhead due to this

## Environment 
* Example box 
* Example rover
* Example G1

## Metrics 
* Metrics is speed (wall clock time/physics step), VRAM comsumption
* Measure mean +- standard deviation
* See newton/examples/mpm_moving_patch_alpha/dynamic_sparse/02_terrain_scaling.png
* Legend style also should be like this figure

## Figure style 
* See newton/examples/mpm_moving_patch_beta/benchmark_plan/cable_admm_proxy_scaling_report.pdf 
* You are supposed to show this in report, so make figure very professional and do not write unncessary texts
* Make figure paper quality, format, style etc


## Writing

The text below is an earlier benchmark draft retained as a writing reference. Current beta measurements and conclusions are in `report.tex`; the old numerical values and no-history description do not describe this run.
 
Base text is here. Make sentence concise and do not intoroduce any words/ideas that you have not introduced. Define it and then talk about it. Do not talk about tiny details like this setting is only for smoke test (author know that).

---

Rover navigation demo using moving terrain patch.
 

How it works 
Simulated particles and visible particles
We separate MPM grid into background and visible regions, and both of these areas are simulated in physics. The visible region follows the object continuously while particles inside background grid are reset to default state once it leaves moving terrain patch.

moving_patch_regions_confluence.png
Particle recycling at the moving boundary
The check runs before every coupled physics step, using each particle's current position. Only particles outside moving patch are reset. A particle outside the visible region but still inside moving patch is not reset. Recycling also handles particles that escape while the moving patch is stationary. Recycling wraps the current horizontal coordinates into the window and restores the particle's original jittered reference height. It resets velocity, APIC velocity gradient, and Cauchy stress to their resting values; elastic and rendering deformation gradients return to identity, and the plastic determinant returns to its default undeformed value. Particles inside the window preserve all of these attributes. Recycled particles represent fresh terrain rather than restoring previously disturbed terrain at that location.

moving_patch_recycling_confluence.png
Per-step execution
The object remains dynamic and receives forces from the sand through two-way coupling. The ground plane remains stationary. Rendering does not change physical particle radii, mass, or material state, and does not freeze background particles. The particle allocation is reused across patch shifts. The hidden buffer normally keeps recycling away from the visible interaction region, but abrupt motion can expose freshly reset particles. This implementation assumes homogeneous terrain and does not preserve tracks outside the window. Long-distance stability and force continuity require further validation.

 

Computation benchmark
Motivation
We evaluate moving-patch terrain as a way to simulate robot–ground interaction over large environments while keeping computational cost manageable. The prototype scene contains a dynamic box moving through homogeneous MPM sand with two-way rigid-body coupling. A shared force controller tracks a straight reference trajectory while allowing the box to respond to sand forces. We compare fixed grids, dynamically allocated sparse grids, and a moving patch using matched material and sampling settings. A simulation image here should show the box, sand, and moving-patch boundary. Measurements exclude rendering and report mean physics-step time ± standard deviation, together with GPU memory usage.

Results
The first result shows that sparse allocation improves efficiency at larger terrain sizes, although fixed grids are faster for the smallest tested bed. At 6 × 6 m, sparse reduces mean step time from 66.93 to 37.36 ms and steady VRAM from 10.13 to 5.99 GiB. However, sparse allocation does not make cost independent of terrain size: its step time and memory usage increase as the simulated sand volume grows.

01_grid_comparison.png
Fig1: Single physics step time and VRAM consumption with MPM’s fixed and sparse grid
 

 

The second result illustrates the benefit of maintaining a fixed-size moving patch. The 3 × 3 m patch measured 18.52 ms per step and approximately 3.48 GiB of steady VRAM. Its simulated particle population remains bounded as the patch follows the object, whereas full-terrain fixed and sparse simulations grow with terrain size. The horizontal patch curve repeats this single measurement as a constant-cost reference; it does not represent separate long-distance traversal experiments.

02_terrain_scaling.png
Fig2: Single physics step time and VRAM consumption with MPM’s fixed and sparse grid and moving patch terrain. We evaluate results among three different terrain sizes: 3mx3m, 6mx6m, and 10mx10m fields.
 

 

The third result shows modest additional cost relative to fixed grids at equal terrain size. With a moving reference, moving-patch simulation takes 14.14 ms per step versus 11.55 ms for fixed and 19.45 ms for dynamic sparse, with steady VRAM of approximately 2.18, 1.99, and 1.24 GiB, respectively. Moving patch is therefore faster than dynamic sparse here, but uses more memory. Moving patch handling and particle recycling likely contribute to the added cost over fixed grids, although these measurements show only small computational overhead. The results support moving patches for large homogeneous environments when terrain history outside the patch can be discarded; longer traversals remain necessary to verify sustained performance.

03_equal_size_comparison.png
Fig3: Single physics step and VRAM consumption when simulating 2mx2mx0.5m terrain.
 