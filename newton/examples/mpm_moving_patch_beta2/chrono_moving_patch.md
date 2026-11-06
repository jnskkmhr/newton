Right now, I am working on improving compute speed and memory usage of moving patch terrain. 
Please follow andrej kartphathy's 12 skills for code readability, consistency, and simplificity: https://github.com/twj515895394/andrej-karpathy-skills-12/blob/main/README.md

First, read existing implementation I have tried so far 
* mpm_moving_patch_alpha: super early stage 
* mpm_moving_patch_alpha: alpha version 
* mpm_moving_patch_beta: beta version
* (I want) mpm_moving_patch_beta2: version inspired by Chrono's moving patch terrain implemented in Project Chrono.

Chrono moving patch terrain is implemented in vehicle model. 
Looking at CRM (SPH based continuum model), can you explain what is difference between our approach and their approach? 

https://api.projectchrono.org/vehicle_terrain_crm_api_.html
The source-based comparison and proposed beta2 work are in
[Chrono CRM and Newton moving terrain](chrono_comparison.md).
