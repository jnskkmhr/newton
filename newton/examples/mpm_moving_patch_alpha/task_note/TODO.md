* Moving patch 
    * Properly handles particle reset 
    * Handles boundary condition 
    * Pregenerate rough geometry (background mesh) and keeps moving patch on this surface 
        * So we can apply moving patch to rough terrain 
        * Intuitively, moving small terrain testbed on background rough terrain mesh? 

* Scene 
    * Try to see if making scene smaller is possible while keeping visual artifacts less disturbing 
    * Try with simple wheeled robot or even box 
    * Try with 
        * slope and wheeled robot 
        * multi material terrain (hard ground to very soft one)

* Evaluation metrics 
    * GRF/deformation compared to real hardware data (manipulator)
    * Set ground super stiff and compare with rigid ground results (see if MPM can simulate reasonable)
    * Policy performance 