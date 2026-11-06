# Benchmark setup validation

These files are not read by the plotting script. Only `../raw/` contributes to the report.

- `pilot/`: short initial checks of all scene types and methods.
- `preliminary_high_drop/`: excluded box measurements with the interactive example's 2 m starting height.
- `settled_box_pilot/`: excluded trial that changed body pose but not the free-joint coordinates; forward kinematics restored the original height.
- `low_box_validation/`: the final box setup changes both body and free-joint coordinates to 0.8 m. Its successful 150-warm-up/150-measured-step sparse 2 m run was copied to the main raw dataset as repeat 0.

The final benchmark asserts the box starts at 0.8 m. Interactive scene files were not modified.
