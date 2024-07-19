**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* (TODO) PrimedErwin
  * (TODO) [LinkedIn](), [personal website](), [twitter](), etc.
* Tested on: (TODO) Windows 11, AMD Ryzen Threadripper 3990X @ 2.90GHz 128GB, RTX 2070 8192MB (CWC 257 Lab)

### (TODO: Your README)

Include screenshots, analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)

Here are the results of this project. Pay attention that this computer is equipped with vs2019 and CUDA 11.6, while the previous one is vs2022 and CUDA 12.5, which led to compilation failure. Maybe this project is not capable with CUDA 12.5 (many deprecated functions are used in code).

## Implement

Implementation is based on blockSize=128, boidsCount=5000, visualize=1.
### Part 1 Naive Boids
This decomposes force into three, and each force need to iterate N\*N times to calculate all the parameters between the node itself and other nodes within the distance, so this part runs slow with time complexity O(N^2).

On RTX2070, it's about 632 fps.

### Part 2.1 better flocking-Uniform Grid
The whole scene has a limited length of 200, and rule1,2,3 also have a limited distance max out 5. So for every boid, we don't need to check all the boids in the scene, all we need to do is check the boids near it. 
Based on this, uniform grid is proposed. Divide the scene into grids with the length of 10, doubled for max rule distance. Then, for a boid inside the grid, we need to check 2\*2\*2 = 8 grids. We call these grid a grid cell. But unfortunately, this number is not always 8, it could be 27(3\*3\*3).
So generally, for a boid in a grid, we check the grids with coords \[-1, 1] based on the original grid. That means, check all the boids within the grid cell instead of all the boids in the scene, which would make the calculation faster.(If boids gathers around this would slow down, but still faster than brute force)
The following I will give a brief of the algorithm.
- For each frame, check which grid the boid is in. To get this, we need position of thisBoid, 1/gridWidth, gridMinimum. Now, assume a boid with pos.x=-99, it should be at the corner of axis x, close to the origin point. First add gridMinimum to pos.x, makes pos.x = -99+100 = 1. Then, multiply it with 1/gridWidth, we get 1\*1\\10 = 0. So this boid locates at grid.x = 0.
- Grids need to be numbered. Still the same boid, assume its grid pos(0,0,0), we need to covert grid(0,0,0) into a unique index. Index it in coord with 3 dimensions, we need gridResolution, which represents how many grids the scene has along a lane. Here the number is 20. The index of current grid is 0+0\*20+0\*20\*20 = 0.
- Now each boid has a grid index. We know which grid the boid is in. Put them into a table like GridIndices\[index]. Assume the same boid, it's the second boid in \*pos, so its index is 1. So GridIndices\[1] = 0. The boid with index 1 is in grid 0.
- GridIndices is in a mess. The value of it can be 1,0,3,5,0..... We need to make it ordered so that we can check all the boids in the grid we need at one time.
- ArrayIndices stores the info of current index boid. In the beginning, ArrayIndices\[0] = 0, which means boid 0's info is stored in pos\[0].
- Sort GridIndices and ArrayIndices at the same time, GridIndices is the key. thrust::sort_by_key would do this. After thrust::sort_by_key, GridIndices looks like 0,0,0,0,1,1,1,2,2,..... and ArrayIndices looks like 1,4,9,11,0,6,.... Look at this, now we know a boid in grid 0 stores its pos and vel info in pos\[1], vel\[1].
- Check different grids. If a boid is in grid 1, we need to check grid 0 and grid 2 (and other grids nearby), we need to know where grid 0 starts and ends in ArrayIndices, so that we can get the correct number from pos and vel to calculate. gridCellStartIndices and gridCellEndIndices would help. StartIndices\[1] is 4, means grid 1 starts at index 4, which corresponds to ArrayIndices\[4] = 0. The boid of pos\[0] is in grid 1. EndIndices\[1] is 7, means grid 1 ends at index 7.
- For a boid(choose it randomly), we calculate its current position of grid, number the grid, sort GridIndices and ArrayIndices, fill gridCellStartIndices and gridCellEndIndices by checking GridIndices, calculate new pos and vel, update, ping-pong temp values.
One more hint, set StartIndices and EndIndices to -1 at the beginning, so you can jump over empty grids.

On RTX2070, it's about 867 fps.

### Part 2.3 better flocking-Coherent Grid
In Part 2.1, we reduced the number of boids that need to be calculated. But pay attention, ArrayIndices like 1,4,9,11,0,6.... is in a mess. That means when a thread read number from gmem, 128byte(4 float) for a warp, thread 0 needs index 1, thread 1 needs index 4. Wait... nobody wants index 2 and index 3? So the warp has to read again, for index 9 and 11 for thread 2 and 3. This wastes a lot of bandwidth.
We need a way to make memory ordered.
Reorder pos and vel, makes its index from 0,1,2,3,4,.... to 1,4,9,11,0,..... Then we can read them just with one time. Bandwidth up up.
So two temp array is needed for reordering, and slightly change with the program.
After reordering with the assistance of ArrayIndices, we don't need ArrayIndices anymore.

On RTX2070, it's about 1005.3 fps.