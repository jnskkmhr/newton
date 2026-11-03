Prevent mesh contact generation from hanging on CUDA by synchronizing threads before resetting the shared edge-progress counter between shape passes.
