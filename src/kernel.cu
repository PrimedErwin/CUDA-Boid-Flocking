#define GLM_FORCE_CUDA
#define __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include "utilityCore.hpp"
#include "kernel.h"

// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char* msg, int line = -1) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		if (line >= 0) {
			fprintf(stderr, "Line %d: ", line);
		}
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

// LOOK-1.2 Parameters for the boids algorithm.
// These worked well in our reference implementation.
#define rule1Distance 5.0f
#define rule2Distance 3.0f
#define rule3Distance 5.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 1.0f

/*! Size of the starting area in simulation space. */
#define scene_scale 100.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

// LOOK-1.2 - These buffers are here to hold all your boid information.
// These get allocated for you in Boids::initSimulation.
// Consider why you would need two velocity buffers in a simulation where each
// boid cares about its neighbors' velocities.
// These are called ping-pong buffers.
glm::vec3* dev_pos;
glm::vec3* dev_vel1;
glm::vec3* dev_vel2;

// LOOK-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.

// For efficient sorting and the uniform grid. These should always be parallel.
int* dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
int* dev_particleGridIndices; // What grid cell is this particle in?

// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int* dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int* dev_gridCellEndIndices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.
glm::vec3* dev_coherent_pos;
glm::vec3* dev_coherent_vel2;
// LOOK-2.1 - Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
	a = (a + 0x7ed55d16) + (a << 12);
	a = (a ^ 0xc761c23c) ^ (a >> 19);
	a = (a + 0x165667b1) + (a << 5);
	a = (a + 0xd3a2646c) ^ (a << 9);
	a = (a + 0xfd7046c5) + (a << 3);
	a = (a ^ 0xb55a4f09) ^ (a >> 16);
	return a;
}

/**
* LOOK-1.2 - this is a typical helper function for a CUDA kernel.
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
	thrust::default_random_engine rng(hash((int)(index * time)));
	thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

	return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

/**
* LOOK-1.2 - This is a basic CUDA kernel.
* CUDA kernel for generating boids with a specified mass randomly around the star.
*/
__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3* arr, float scale) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N) {
		glm::vec3 rand = generateRandomVec3(time, index);
		arr[index].x = scale * rand.x;
		arr[index].y = scale * rand.y;
		arr[index].z = scale * rand.z;
	}
}

/**
* Initialize memory, update some globals
*/
void Boids::initSimulation(int N) {
	numObjects = N;
	dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

	// LOOK-1.2 - This is basic CUDA memory management and error checking.
	// Don't forget to cudaFree in  Boids::endSimulation.
	cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

	cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

	cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");
	// LOOK-1.2 - This is a typical CUDA kernel invocation.
	kernGenerateRandomPosArray << <fullBlocksPerGrid, blockSize >> > (1, numObjects,
		dev_pos, scene_scale);
	checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

	// LOOK-2.1 computing grid params
	gridCellWidth = 2.0f * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
	int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
	gridSideCount = 2 * halfSideCount;

	gridCellCount = gridSideCount * gridSideCount * gridSideCount;
	gridInverseCellWidth = 1.0f / gridCellWidth;
	float halfGridWidth = gridCellWidth * halfSideCount;
	gridMinimum.x -= halfGridWidth;
	gridMinimum.y -= halfGridWidth;
	gridMinimum.z -= halfGridWidth;

	// TODO-2.1 TODO-2.3 - Allocate additional buffers here.

	cudaMallocAsync(&dev_particleArrayIndices, N * sizeof(int), cudaStreamPerThread);
	checkCUDAErrorWithLine("cudaMallocAsync dev_particleArrayIndices failed!");

	cudaMallocAsync(&dev_particleGridIndices, N * sizeof(int), cudaStreamPerThread);
	checkCUDAErrorWithLine("cudaMallocAsync dev_particleGridIndices failed!");
	//Actually start and end indices only need gridCellCount * sizeof(int) bytes
	cudaMallocAsync(&dev_gridCellStartIndices, gridCellCount * sizeof(int), cudaStreamPerThread);
	checkCUDAErrorWithLine("cudaMallocAsync dev_gridCellStartIndices failed!");

	cudaMallocAsync(&dev_gridCellEndIndices, gridCellCount * sizeof(int), cudaStreamPerThread);
	checkCUDAErrorWithLine("cudaMallocAsync dev_gridCellEndIndices failed!");

	cudaMallocAsync(&dev_coherent_pos, N * sizeof(glm::vec3), cudaStreamPerThread);
	checkCUDAErrorWithLine("cudaMallocAsync dev_coherent_pos failed!");

	cudaMallocAsync(&dev_coherent_vel2, N * sizeof(glm::vec3), cudaStreamPerThread);
	checkCUDAErrorWithLine("cudaMallocAsync dev_coherent_vel2 failed!");


	cudaDeviceSynchronize();
}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3* pos, float* vbo, float s_scale) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	float c_scale = -1.0f / s_scale;

	if (index < N) {
		vbo[4 * index + 0] = pos[index].x * c_scale;
		vbo[4 * index + 1] = pos[index].y * c_scale;
		vbo[4 * index + 2] = pos[index].z * c_scale;
		vbo[4 * index + 3] = 1.0f;
	}
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3* vel, float* vbo, float s_scale) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if (index < N) {
		vbo[4 * index + 0] = vel[index].x + 0.3f;
		vbo[4 * index + 1] = vel[index].y + 0.3f;
		vbo[4 * index + 2] = vel[index].z + 0.3f;
		vbo[4 * index + 3] = 1.0f;
	}
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void Boids::copyBoidsToVBO(float* vbodptr_positions, float* vbodptr_velocities) {
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

	kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_pos, vbodptr_positions, scene_scale);
	kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_vel1, vbodptr_velocities, scene_scale);

	checkCUDAErrorWithLine("copyBoidsToVBO failed!");

	cudaDeviceSynchronize();
}


/******************
* stepSimulation *
******************/

/**
* LOOK-1.2 You can use this as a helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
//This function calculates delta of velocity according to the rules
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3* pos, const glm::vec3* vel) {
	// Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
	// Rule 2: boids try to stay a distance d away from each other
	// Rule 3: boids try to match the speed of surrounding boids
	  //x, y, z represents delta of velocity in 3 dimensions
	float neighbor_count = 0.f;
	float neighbor_count3 = 0.f;
	glm::vec3 center(0.0f, 0.0f, 0.0f);
	glm::vec3 seperate(0.0f, 0.0f, 0.0f);
	glm::vec3 cohesion(0.0f, 0.0f, 0.0f);
	glm::vec3 thisBoidp = pos[iSelf];
	glm::vec3 thisBoidv = vel[iSelf];
	glm::vec3 return_vel(0.0f, 0.0f, 0.0f);
	auto sqrt_distance = [](float x, float y, float z) {
		return __fsqrt_rn(x * x + y * y + z * z);
	};
	//iter all other boids
	for (int i = 0; i < N; i++)
	{
		if (i == iSelf) continue;
		float distance = sqrt_distance(thisBoidp.x - pos[i].x, thisBoidp.y - pos[i].y,
			thisBoidp.z - pos[i].z);
		if (distance < rule1Distance)
		{
			//Rule 1
			center.x += pos[i].x;
			center.y += pos[i].y;
			center.z += pos[i].z;
			neighbor_count += 1.0f;
		}
		if (distance < rule2Distance)
		{
			//Rule 2
			seperate.x -= pos[i].x - thisBoidp.x;
			seperate.y -= pos[i].y - thisBoidp.y;
			seperate.z -= pos[i].z - thisBoidp.z;
		}
		if (distance < rule3Distance)
		{
			//Rule 3
			cohesion.x += vel[i].x;
			cohesion.y += vel[i].y;
			cohesion.z += vel[i].z;
			neighbor_count3 += 1.0f;
		}
	}
	if (neighbor_count)
	{
		center.x /= neighbor_count;
		center.y /= neighbor_count;
		center.z /= neighbor_count;
		center.x = (center.x - thisBoidp.x) * rule1Scale;
		center.y = (center.y - thisBoidp.y) * rule1Scale;
		center.z = (center.z - thisBoidp.z) * rule1Scale;
	}
	seperate.x *= rule2Scale;
	seperate.y *= rule2Scale;
	seperate.z *= rule2Scale;
	if (neighbor_count3)
	{
		cohesion.x /= neighbor_count3;
		cohesion.y /= neighbor_count3;
		cohesion.z /= neighbor_count3;
		cohesion.x *= rule3Scale;
		cohesion.y *= rule3Scale;
		cohesion.z *= rule3Scale;
	}
	return_vel.x += center.x + seperate.x + cohesion.x;
	return_vel.y += center.y + seperate.y + cohesion.y;
	return_vel.z += center.z + seperate.z + cohesion.z;
	return return_vel;
}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3* pos,
	glm::vec3* vel1, glm::vec3* vel2) {
	int iSelf = threadIdx.x + blockDim.x * blockIdx.x;
	if (iSelf >= N) return;
	// Compute a new velocity based on pos and vel1
		  //delta of velocity for each boid
	glm::vec3 return_vel;

	return_vel = vel1[iSelf] + computeVelocityChange(N, iSelf, pos, vel1);
	// Clamp the speed
	if (glm::length(return_vel) > maxSpeed)
	{
		return_vel = glm::normalize(return_vel) * maxSpeed;
	}
	// Record the new velocity into vel2. Question: why NOT vel1?
	 //Answer: ping-pong velocity
	vel2[iSelf] = return_vel;

}

/**
* LOOK-1.2 Since this is pretty trivial, we implemented it for you.
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdatePos(int N, float dt, glm::vec3* pos, glm::vec3* vel) {
	// Update position by velocity
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N) {
		return;
	}
	glm::vec3 thisPos = pos[index];
	thisPos += vel[index] * dt;

	// Wrap the boids around so we don't lose them
	thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x;
	thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y;
	thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z;

	thisPos.x = thisPos.x > scene_scale ? -scene_scale : thisPos.x;
	thisPos.y = thisPos.y > scene_scale ? -scene_scale : thisPos.y;
	thisPos.z = thisPos.z > scene_scale ? -scene_scale : thisPos.z;

	pos[index] = thisPos;
}

// LOOK-2.1 Consider this method of computing a 1D index from a 3D grid index.
// LOOK-2.3 Looking at this method, what would be the most memory efficient
//          order for iterating over neighboring grid cells?
//          for(x)
//            for(y)
//             for(z)? Or some other order?
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
	return x + y * gridResolution + z * gridResolution * gridResolution;
}

__global__ void kernComputeIndices(int N, int gridResolution,
	glm::vec3 gridMin, float inverseCellWidth,
	glm::vec3* pos, int* indices, int* gridIndices) {
	// TODO-2.1
	// - Label each boid with the index of its grid cell.
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index >= N) return;
	//get the coord of boid marked with grid cell
	//assume that pos.x is 0-9.999, divided by 10 then it goes to 0, pos.x is 0 in grid cell coord
	//but pos.x can be minus, so -gridMin makes it positive
	//so that we can get positive gridIndices
	glm::ivec3 boidPos = (pos[index] - gridMin) * inverseCellWidth;
	gridIndices[index] = gridIndex3Dto1D(boidPos.x, boidPos.y, boidPos.z, gridResolution);
	// - Set up a parallel array of integer indices as pointers to the actual
	indices[index] = index;
	//   boid data in pos and vel1/vel2

}

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
__global__ void kernResetIntBuffer(int N, int* intBuffer, int value) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N) {
		intBuffer[index] = value;
	}
}

__global__ void kernIdentifyCellStartEnd(int N, int* particleGridIndices,
	int* gridCellStartIndices, int* gridCellEndIndices) {
	// TODO-2.1
	// Identify the start point of each cell in the gridIndices array.
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index >= N) return;
	// This is basically a parallel unrolling of a loop that goes
	// "this index doesn't match the one before it, must be a new cell!"
	if (index == 0)
	{
		gridCellStartIndices[particleGridIndices[index]] = 0;
	}
	else if (index == N - 1)
	{
		gridCellEndIndices[particleGridIndices[index]] = N - 1;
	}
	else if (particleGridIndices[index] != particleGridIndices[index + 1])//2 boids in different grid cell
		//judge on the boundary wont cause mem write confliction
	{
		gridCellEndIndices[particleGridIndices[index]] = index;//this cell ends at index
		gridCellStartIndices[particleGridIndices[index + 1]] = index + 1;//another cell starts at index+1
	}
}

__global__ void kernUpdateVelNeighborSearchScattered(
	int N, int gridResolution, glm::vec3 gridMin,
	float inverseCellWidth, float cellWidth,
	int* gridCellStartIndices, int* gridCellEndIndices,
	int* particleArrayIndices,
	glm::vec3* pos, glm::vec3* vel1, glm::vec3* vel2) {
	// TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
	// the number of boids that need to be checked.
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index >= N) return;
	glm::vec3 center(0.0f, 0.0f, 0.0f);
	glm::vec3 seperate(0.0f, 0.0f, 0.0f);
	glm::vec3 velocity(0.0f, 0.0f, 0.0f);
	glm::vec3 return_vel(0.0f, 0.0f, 0.0f);
	glm::vec3 thisBoid = pos[index];
	int neighbor_count = 0;
	int neighbor_count3 = 0;
	float distance = 0;
	// - Identify the grid cell that this particle is in
	glm::ivec3 boidPos = (thisBoid - gridMin) * inverseCellWidth;
	int x = boidPos.x;
	int y = boidPos.y;
	int z = boidPos.z;
	// - Identify which cells may contain neighbors. This isn't always 8.
	//so calculate 3*3*3 neighbors, these neighbor's grid cell index can
	//be calculated by gridIndex3Dto1D
	for (int k = -1; k <= 1; k++)
	{
		for (int j = -1; j <= 1; j++)
		{
			for (int i = -1; i <= 1; i++)
			{
				int near_x = x + i;
				int near_y = y + j;
				int near_z = z + k;
				//boundary check, near_?  = [0, gridResolution)
				near_x = imax(near_x, 0);
				near_y = imax(near_y, 0);
				near_z = imax(near_z, 0);
				near_x = imin(near_x, gridResolution - 1);
				near_y = imin(near_y, gridResolution - 1);
				near_z = imin(near_z, gridResolution - 1);
				//near_? to grid cell index
				int nearGridCellIndex = gridIndex3Dto1D(near_x, near_y, near_z, gridResolution);
				// - For each cell, read the start/end indices in the boid pointer array.
				if (gridCellStartIndices[nearGridCellIndex] != -1)//is not empty
				{
					for (int indices = gridCellStartIndices[nearGridCellIndex]; indices <= gridCellEndIndices[nearGridCellIndex]; indices++)
					{
						// - Access each boid in the cell and compute velocity change from
						//   the boids rules, if this boid is within the neighborhood distance.
						int bindex = particleArrayIndices[indices];
						if (bindex != index)
						{
							distance = glm::distance(pos[bindex], thisBoid);
							//Rule1
							if (distance < rule1Distance)
							{
								center += pos[bindex];
								neighbor_count++;
							}
							//Rule2
							if (distance < rule2Distance)
							{
								seperate -= (pos[bindex] - thisBoid);
							}
							//Rule3
							if (distance < rule3Distance)
							{
								velocity += vel1[bindex];
								neighbor_count3++;
							}
						}
					}
				}
			}
		}
	}
	if (neighbor_count)
	{
		center /= neighbor_count;
		center = (center - thisBoid) * rule1Scale;
	}
	if (neighbor_count3)
	{
		velocity /= neighbor_count3;
		velocity *= rule3Scale;
	}
	seperate *= rule2Scale;
	// - Clamp the speed change before putting the new speed in vel2
	return_vel = vel1[index] + center + seperate + velocity;
	if (glm::length(return_vel) > maxSpeed)
	{
		return_vel = glm::normalize(return_vel) * maxSpeed;
	}
	vel2[index] = return_vel;
}
/// <summary>
/// The coherent manner sorts the pos and vel1, so that the memory is
/// continuous, which makes cuda run faster.
/// </summary>
/// <param name="N"></param>
/// <param name="gridResolution"></param>
/// <param name="gridMin"></param>
/// <param name="inverseCellWidth"></param>
/// <param name="cellWidth"></param>
/// <param name="gridCellStartIndices"></param>
/// <param name="gridCellEndIndices"></param>
/// <param name="pos">dev_coherent_pos</param>
/// <param name="vel1">dev_coherent_vel2</param>
/// <param name="vel2"></param>
/// <returns></returns>
__global__ void kernUpdateVelNeighborSearchCoherent(
	int N, int gridResolution, glm::vec3 gridMin,
	float inverseCellWidth, float cellWidth,
	int* gridCellStartIndices, int* gridCellEndIndices,
	glm::vec3* pos, glm::vec3* vel1, glm::vec3* vel2) {
	// TODO-2.3 - This should be very similar to kernUpdateVelNeighborSearchScattered,
	// except with one less level of indirection.
	// This should expect gridCellStartIndices and gridCellEndIndices to refer
	// directly to pos and vel1.
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index >= N) return;
	glm::vec3 center(0.0f, 0.0f, 0.0f);
	glm::vec3 seperate(0.0f, 0.0f, 0.0f);
	glm::vec3 velocity(0.0f, 0.0f, 0.0f);
	glm::vec3 return_vel(0.0f, 0.0f, 0.0f);
	glm::vec3 thisBoid = pos[index];
	int neighbor_count = 0;
	int neighbor_count3 = 0;
	float distance = 0;
	// - Identify the grid cell that this particle is in
	glm::ivec3 boidPos = (thisBoid - gridMin) * inverseCellWidth;
	int x = boidPos.x;
	int y = boidPos.y;
	int z = boidPos.z;
	// - Identify which cells may contain neighbors. This isn't always 8.
	//so calculate 3*3*3 neighbors, these neighbor's grid cell index can
	//be calculated by gridIndex3Dto1D
	//   DIFFERENCE: For best results, consider what order the cells should be
	//   checked in to maximize the memory benefits of reordering the boids data.
	//  Actually no difference, I used to range them as z, y, x, benefit max
	for (int k = -1; k <= 1; k++)
	{
		for (int j = -1; j <= 1; j++)
		{
			for (int i = -1; i <= 1; i++)
			{
				int near_x = x + i;
				int near_y = y + j;
				int near_z = z + k;
				//boundary check, near_?  = [0, gridResolution)
				near_x = imax(near_x, 0);
				near_y = imax(near_y, 0);
				near_z = imax(near_z, 0);
				near_x = imin(near_x, gridResolution - 1);
				near_y = imin(near_y, gridResolution - 1);
				near_z = imin(near_z, gridResolution - 1);
				//near_? to grid cell index
				int nearGridCellIndex = gridIndex3Dto1D(near_x, near_y, near_z, gridResolution);
				// - For each cell, read the start/end indices in the boid pointer array.
				if (gridCellStartIndices[nearGridCellIndex] != -1)//is not empty
				{
					for (int indices = gridCellStartIndices[nearGridCellIndex]; indices <= gridCellEndIndices[nearGridCellIndex]; indices++)
					{
						// - Access each boid in the cell and compute velocity change from
						//   the boids rules, if this boid is within the neighborhood distance.
						int bindex = indices;
						if (bindex != index)
						{
							distance = glm::distance(pos[bindex], thisBoid);
							//Rule1
							if (distance < rule1Distance)
							{
								center += pos[bindex];
								neighbor_count++;
							}
							//Rule2
							if (distance < rule2Distance)
							{
								seperate -= (pos[bindex] - thisBoid);
							}
							//Rule3
							if (distance < rule3Distance)
							{
								velocity += vel1[bindex];
								neighbor_count3++;
							}
						}
					}
				}
			}
		}
	}
	if (neighbor_count)
	{
		center /= neighbor_count;
		center = (center - thisBoid) * rule1Scale;
	}
	if (neighbor_count3)
	{
		velocity /= neighbor_count3;
		velocity *= rule3Scale;
	}
	seperate *= rule2Scale;
	// - Clamp the speed change before putting the new speed in vel2
	return_vel = vel1[index] + center + seperate + velocity;
	if (glm::length(return_vel) > maxSpeed)
	{
		return_vel = glm::normalize(return_vel) * maxSpeed;
	}
	vel2[index] = return_vel;

}

__global__	 void kernCoherentPosVel(int N, int* dev_particleArrayIndices, 
	glm::vec3* pos, glm::vec3* vel,
	glm::vec3* coherentpos, glm::vec3* coherentvel)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index >= N) return;
	//coherentindex is where the value truely locates in pos and vel
	int coherentindex = dev_particleArrayIndices[index];
	//map them
	coherentpos[index] = pos[coherentindex];
	coherentvel[index] = vel[coherentindex];
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
	// TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
	// TODO-1.2 ping-pong the velocity buffers
	dim3 grids((numObjects - 1) / blockSize + 1);

	kernUpdateVelocityBruteForce << <grids, blockSize >> > (numObjects, dev_pos, dev_vel1, dev_vel2);
	kernUpdatePos << <grids, blockSize >> > (numObjects, dt, dev_pos, dev_vel2);
	checkCUDAErrorWithLine("kernel update failed!");
	//Err.. can this be ping-pong?
	//It can! Another way is std::swap()
	cudaMemcpy(dev_vel1, dev_vel2, numObjects * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);

}

void Boids::stepSimulationScatteredGrid(float dt) {
	// TODO-2.1
	dim3 grids((numObjects - 1) / blockSize + 1);
	dim3 gridsGridCell((gridCellCount - 1) / blockSize + 1);
	//reset start and end indices, in case of empty grid cell
	kernResetIntBuffer << <gridsGridCell, blockSize >> > (gridCellCount, dev_gridCellStartIndices, -1);
	kernResetIntBuffer << <gridsGridCell, blockSize >> > (gridCellCount, dev_gridCellEndIndices, -1);
	// Uniform Grid Neighbor search using Thrust sort.
	// In Parallel:
	// - label each particle with its array index as well as its grid index.
	//   Use 2x width grids.
	kernComputeIndices << <grids, blockSize >> > (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
	// - Unstable key sort using Thrust. A stable sort isn't necessary, but you
	//   are welcome to do a performance comparison.
	thrust::device_ptr<int>dev_thrust_arrayIndice(dev_particleArrayIndices);
	thrust::device_ptr<int>dev_thrust_gridIndice(dev_particleGridIndices);
	thrust::sort_by_key(dev_thrust_gridIndice, dev_thrust_gridIndice + numObjects, dev_thrust_arrayIndice);
	checkCUDAErrorWithLine("thrust sort_by_key failed!");
	// - Naively unroll the loop for finding the start and end indices of each
	//   cell's data pointers in the array of boid indices
	kernIdentifyCellStartEnd << <grids, blockSize >> > (numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
	// - Perform velocity updates using neighbor search
	kernUpdateVelNeighborSearchScattered << <grids, blockSize >> > (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth,
		dev_gridCellStartIndices, dev_gridCellEndIndices, dev_particleArrayIndices, dev_pos, dev_vel1, dev_vel2);
	// - Update positions
	kernUpdatePos << <grids, blockSize >> > (numObjects, dt, dev_pos, dev_vel2);
	// - Ping-pong buffers as needed
	cudaMemcpy(dev_vel1, dev_vel2, numObjects * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);

}

void Boids::stepSimulationCoherentGrid(float dt) {
	// TODO-2.3 - start by copying Boids::stepSimulationNaiveGrid
	dim3 grids((numObjects - 1) / blockSize + 1);
	dim3 gridsGridCell((gridCellCount - 1) / blockSize + 1);
	//reset start and end indices, in case of empty grid cell
	kernResetIntBuffer << <gridsGridCell, blockSize >> > (gridCellCount, dev_gridCellStartIndices, -1);
	kernResetIntBuffer << <gridsGridCell, blockSize >> > (gridCellCount, dev_gridCellEndIndices, -1);

	// Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
	// In Parallel:
	// - Label each particle with its array index as well as its grid index.
	//   Use 2x width grids
	kernComputeIndices << <grids, blockSize >> > (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
	// - Unstable key sort using Thrust. A stable sort isn't necessary, but you
	//   are welcome to do a performance comparison.
	thrust::device_ptr<int>dev_thrust_arrayIndice(dev_particleArrayIndices);
	thrust::device_ptr<int>dev_thrust_gridIndice(dev_particleGridIndices);
	thrust::sort_by_key(dev_thrust_gridIndice, dev_thrust_gridIndice + numObjects, dev_thrust_arrayIndice);
	checkCUDAErrorWithLine("thrust sort_by_key failed!");
	// - Naively unroll the loop for finding the start and end indices of each
	//   cell's data pointers in the array of boid indices
	kernIdentifyCellStartEnd << <grids, blockSize >> > (numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
	// - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
	//   the particle data in the simulation array.
	//   CONSIDER WHAT ADDITIONAL BUFFERS YOU NEED
	kernCoherentPosVel<<<grids, blockSize>>>(numObjects, dev_particleArrayIndices, dev_pos, dev_vel1, dev_coherent_pos, dev_coherent_vel2);
	// - Perform velocity updates using neighbor search
	kernUpdateVelNeighborSearchCoherent << <grids, blockSize >> > (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth,
		dev_gridCellStartIndices, dev_gridCellEndIndices, dev_coherent_pos, dev_coherent_vel2, dev_vel1);
	// - Update positions
	kernUpdatePos<<<grids, blockSize>>>(numObjects, dt, dev_coherent_pos, dev_vel1);
	// - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.
	//cudaMemcpy(dev_vel1, dev_vel2, numObjects * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dev_pos, dev_coherent_pos, numObjects * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
}

void Boids::endSimulation() {
	cudaFree(dev_vel1);
	cudaFree(dev_vel2);
	cudaFree(dev_pos);

	// TODO-2.1 TODO-2.3 - Free any additional buffers here.
	cudaFreeAsync(dev_particleArrayIndices, cudaStreamPerThread);
	cudaFreeAsync(dev_particleGridIndices, cudaStreamPerThread);
	cudaFreeAsync(dev_gridCellStartIndices, cudaStreamPerThread);
	cudaFreeAsync(dev_gridCellEndIndices, cudaStreamPerThread);
	cudaFreeAsync(dev_coherent_pos, cudaStreamPerThread);
	cudaFreeAsync(dev_coherent_vel2, cudaStreamPerThread);
}

void Boids::unitTest() {
	// LOOK-1.2 Feel free to write additional tests here.

	// test unstable sort
	int* dev_intKeys;
	int* dev_intValues;
	int N = 10;

	std::unique_ptr<int[]>intKeys{ new int[N] };
	std::unique_ptr<int[]>intValues{ new int[N] };

	intKeys[0] = 0; intValues[0] = 0;
	intKeys[1] = 1; intValues[1] = 1;
	intKeys[2] = 0; intValues[2] = 2;
	intKeys[3] = 3; intValues[3] = 3;
	intKeys[4] = 0; intValues[4] = 4;
	intKeys[5] = 2; intValues[5] = 5;
	intKeys[6] = 2; intValues[6] = 6;
	intKeys[7] = 0; intValues[7] = 7;
	intKeys[8] = 5; intValues[8] = 8;
	intKeys[9] = 6; intValues[9] = 9;

	cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

	cudaMalloc((void**)&dev_intValues, N * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

	dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

	std::cout << "before unstable sort: " << std::endl;
	for (int i = 0; i < N; i++) {
		std::cout << "  key: " << intKeys[i];
		std::cout << " value: " << intValues[i] << std::endl;
	}

	// How to copy data to the GPU
	cudaMemcpy(dev_intKeys, intKeys.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_intValues, intValues.get(), sizeof(int) * N, cudaMemcpyHostToDevice);

	// Wrap device vectors in thrust iterators for use with thrust.
	thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
	thrust::device_ptr<int> dev_thrust_values(dev_intValues);
	// LOOK-2.1 Example for using thrust::sort_by_key
	thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

	// How to copy data back to the CPU side from the GPU
	cudaMemcpy(intKeys.get(), dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
	cudaMemcpy(intValues.get(), dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
	checkCUDAErrorWithLine("memcpy back failed!");

	std::cout << "after unstable sort: " << std::endl;
	for (int i = 0; i < N; i++) {
		std::cout << "  key: " << intKeys[i];
		std::cout << " value: " << intValues[i] << std::endl;
	}

	// cleanup
	cudaFree(dev_intKeys);
	cudaFree(dev_intValues);
	checkCUDAErrorWithLine("cudaFree failed!");
	return;
}
