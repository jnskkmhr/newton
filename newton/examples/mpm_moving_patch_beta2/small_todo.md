## Immediate improvement 
* Consider limiting particle ACTIVE to only portion that is touching object + margin 
* Make particles outside INACTIVE + assign kinematic collider (mass = 0?)

## Bit more future 
* Figure out any low level math in warp / implicit mpm solver side in newton that can be improved to speed up mpm solve like this commit: https://github.com/NVIDIA/warp/commit/85e3eea2f7bfb48649d35e848144990efd0e0a3f#diff-be4ab6d076baf8037e6c58418e22415166814642c01f23c12a5d17d51e21b521


## Coupler investigation 
* Figure out why proxy coupler and impulse exchange does not provide the same converged penetration 
    * for example, proxy coupler (/home/jkamohara/isaac/tmp/newton/newton/examples/mpm_mujoco/example_mpm_g1_jump.py) and manual impulse exchange (/home/jkamohara/isaac/tmp/newton/newton/examples/mpm_mujoco/example_mpm_g1_jump_legacy.py). Why equalibrium sinkage looks so different??? 
