Now trying to benchmark MPM with 
* fixed grid 
* sparse grid 
* sparse grid with adaptive rendering 
* moving patch 

What do you think is the good benchmark setup? I am thinking about the following evaluation criteria and metrics: speed/physics step and average VRAM comsumption. 
Feel free to add another questions and metrics. 

1. Does moving patch scale to large terrain size? 
Metrics: speed, VRAM comsumption
* Define operation space say, 10m x 10m terrain field 
* Use fixed, sparse, moving patch grid
* Moving patch grid only defines active MPM grid say 3mx3mx0.5m and you keep moving grid around robot moving in this operating space. 
* Btw, I think there is no doubt that moving patch works better. 

2. Does moving patch have some overhead comapred to without moving patch?
Metrics: speed, VRAM comsumption
* Define small scale environment like 2mx2mx0.5m
* Simulate with dense, sparse, moving patch 
* See if moving patch method has some computation overhead due to moving patch implementation 