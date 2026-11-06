> Design response: [Persistent local MPM implementation plan](Implicit_MPM_2.0_design.md).

# Toward scalable large scene simulation with MPM 

## Motivation 
Sparse grid means building grid around particles, NOT around particles with active collision.
It is effective for rough terrain like wave terrain as it has many empty space and fixed grid allocates grid to empty space too.
There won't be much compute/memory overhead difference between fixed and sparse on flat terrain as fixed grid efficiently covers entire sand.

Then, I feel like it does not really solve anything about unbounded terrain.
Because you need to consider physics for all particles and this is super expensive and does not scale to terrain size.

So, I need the some kind of logic to restrict MPM to only focus on terrain around intruding bodies (manipulator/robot leg etc).
And in my understanding, this is not possible with current MPM implementation.
My current moving terrain patch implementation is kind of easy way to achieve this.
But, it would be great if we do more proper implementation that also cache already disturbed particles.

## Level of details adjustment
Persistent local simulation: activate nearby terrain, preserve inactive terrain state, and handle transitions and boundary forces. This requires additional implementation.
* Only simulate terrain close to robot (can be controlled by parameter like active MPM grid size). Meaning build grid around some areas around robot, you can define size of this local terrain. Just focus grid construction within this local terrain and you can still use sparse grid probably
* You also have choice of rendering particles in this local terrain while rest of particles not simulated are cached somewhere (wondering what's good way to scale)
* Also question is when you later parallelize simulation and spawn thousands of robot, do you share this non-simulated terrain among robots? Otherwise, it is expensive to store particle states for each robot's background terrain. I do not have good answer


Here is list of implementation we need to do ? 
### Restricting MPM calculation to local terrain around robot 
* You can burrow some ideas from moving terrain patch implementation already 
* But, you may need to implement MPM wrapper that changes P2G behavior that takes particle position and target body position. At this moment, we might just consider single rigid object/articulation now for this pipeline. Thinking about this for multiple independently moving object is difficult.

### Caching non-simulated particles in background terrain 
* Decide if this terrain should be shared among thousands of robots like you only spawn single terrain mesh for thousands of robots in IsaacLab
* In that case, how do you update particles from multiple robots back and forth unknown. We can prototype with single robot in newton though, but we should have this in mind.

### Implement rendering function 
* Implement function that only returns particles inside local terrain. By default, you only render this. 
* Optionally return particles int he background terrain for debug

## Other random thoughts: Adaptive scaling grid resolution 
From genersis world, there is adaptive resolution MPM published. 
This still requires simulating particles far away from robot, but with more coarse grid.
https://arxiv.org/pdf/2411.12711