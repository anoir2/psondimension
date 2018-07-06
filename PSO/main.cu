#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <float.h>
#include <errno.h>
#include <stdint.h>
#include <cuda_runtime_api.h>
#define DIM 2
#define SHARED_MEMORY_DIM ((1<<15)+(1<<14)) // 48KB
#define N_THREAD_GPU (1<<10) // limit is 1024

#define MAX_STEPS (1<<3) /* run for no more than 1Mi steps */
#define TARGET_FITNESS (FLT_EPSILON) /* or until the fitness is less than this much */
#define STEP_CHECK_FREQ 1 /* after how many steps to write the system and check the time */

/*	needed for find fitness min in parallel	*/
typedef struct fitness_pos
{
	int pos;
	float fitness;
} fitness_pos;

/* The whole particle system */

__device__ float *current_best_pos;
__device__ float *global_best_pos;
__constant__ int num_particles;
__device__ float global_fitness = HUGE_VALF;
__device__ float current_fitness;

/* Extents of the domain in each dimension */
#define coord_min -1
#define coord_max 1
#define coord_range 2
float target_pos[DIM];	/* The target position */
__constant__ float target_pos_shared[DIM];
float fitness_min;

/* Particle components*/
__device__ float* pos;
__device__ float* vel;
__device__ float* best_pos;
__device__ uint64_t* prng_state;
__device__ float* best_fit;
__device__ float* fitness_val;




/* Overall weight for the old velocity, best position distance and global
 * best position distance in the computation of the new velocity
 */
#define vel_omega 0.9
#define vel_phi_best 2
#define vel_phi_global 2

/* The contribution of the velocity to the new position. Set to 1
 * to use the standard PSO approach of adding the whole velocity
 * to the position.
 */
#define step_factor 1

__device__ __host__ uint32_t MWC64X(uint64_t *state);
__device__ __host__ float range_rand(float min, float max, uint64_t *prng_state);
__device__ __host__ void init_rand(uint64_t *prng_state, int i);
__device__ float fitness(float *pos);
__device__ void warp_control_float2(float2* smpos, int particleIndexSHM);
__device__ void warp_control_float(float* smpos, int particleIndexSHM);
__global__ void init_particle();
__global__ void find_min_fitness_parallel(__restrict__ const fitness_pos* in, fitness_pos* out,const int offset,const int n_in,const int blocks);
__global__ void new_vel_pos();
__global__ void calc_fitness();
int ceil_log2(unsigned long long x);
void check_error(cudaError_t err, const char *msg);
void start_time_record(cudaEvent_t *before, cudaEvent_t *after);
void stop_time_record(cudaEvent_t *before, cudaEvent_t *after, float *runtime);
void parallel_fitness(const int n_particle, int n_thread);
void write_system(const int step, const float calc_fitness_time, const float new_vel_pos, const float fitness_min, const int n_particles);

void init_mem(int n_particle){
	float *pos_d, *vel_d, *best_pos_d, *best_fit_d,*fitness_val_d, *current_best_pos_d, *global_best_pos_d;
	uint64_t *prng_state_d;
	check_error(cudaMalloc(&pos_d,sizeof(float) * n_particle * DIM),"memory alloc n particle pos");
	check_error(cudaMalloc(&vel_d,sizeof(float) * n_particle * DIM),"memory alloc n particle vel");
	check_error(cudaMalloc(&best_pos_d,sizeof(float) * n_particle * DIM),"memory alloc n particle best_pos");
	check_error(cudaMalloc((uint64_t **)&prng_state_d,sizeof(uint64_t) * n_particle),"memory alloc n particle best_pos");
	check_error(cudaMalloc(&best_fit_d,sizeof(float) * n_particle),"memory alloc n particle best_pos");
	check_error(cudaMalloc(&fitness_val_d,sizeof(float) * n_particle),"memory alloc n particle best_pos");
	check_error(cudaMalloc(&current_best_pos_d,sizeof(float) * DIM),"memory alloc n particle best_pos");
	check_error(cudaMalloc(&global_best_pos_d,sizeof(float) * DIM),"memory alloc n particle best_pos");

	check_error(cudaMemcpyToSymbol(target_pos_shared, &target_pos, sizeof(float)*DIM),"memory cpy to device target_pos");
	check_error(cudaMemcpyToSymbol(num_particles, &n_particle, sizeof(int)),"memory cpy to device num_particle");
	check_error(cudaMemcpyToSymbol(prng_state, &prng_state_d, sizeof(uint64_t)),"memory cpy to device num_particle");
	check_error(cudaMemcpyToSymbol(pos, &pos_d, sizeof(pos_d)),"memory cpy to device target_pos");
	check_error(cudaMemcpyToSymbol(vel, &vel_d, sizeof(vel)),"memory cpy to device target_pos");
	check_error(cudaMemcpyToSymbol(best_pos, &best_pos_d, sizeof(best_pos)),"memory cpy to device target_pos");
	check_error(cudaMemcpyToSymbol(best_fit, &best_fit_d, sizeof(best_fit)),"memory cpy to device target_pos");
	check_error(cudaMemcpyToSymbol(fitness_val, &fitness_val_d, sizeof(fitness_val)),"memory cpy to device target_pos");
	check_error(cudaMemcpyToSymbol(current_best_pos, &current_best_pos_d, sizeof(fitness_val)),"memory cpy to device target_pos");
	check_error(cudaMemcpyToSymbol(global_best_pos, &global_best_pos_d, sizeof(fitness_val)),"memory cpy to device target_pos");

}
int main(int argc, char *argv[])
{

	unsigned step = 0;
	unsigned n_particle;
	cudaEvent_t before, after;
	float calc_fitness_time = 0, new_vel_pos_time = 0, fitness_min_time = 0;
	int j;
	int n_blocks;
	int n_thread = N_THREAD_GPU;
	uint64_t prng_state_h;
	dim3 init_parall(N_THREAD_GPU/DIM,DIM,1);
	/*	Get particle's numbers, default 128	*/
	n_particle = argc > 1 ? atoi(argv[1]) : 128;

	/*	Define n blocks for GPU parallelization	*/
	n_blocks = ceil((float)n_particle / N_THREAD_GPU) == 0 ? 1 : ceil((float)n_particle / N_THREAD_GPU);

	/* Initialize the target position */
	init_rand(&prng_state_h, time(NULL));

	printf("target position: (");
	for(j = 0; j < DIM; j++){
		target_pos[j] = range_rand(coord_min, coord_max, &prng_state_h);
		printf("%f,", target_pos[j]);
	}
	printf(")\n");

	target_pos[0] = -0.287776;
	target_pos[1] = 0.520416;

	/* Initialize a system with the number of particles given
	 * on the command-line or from default value (128) */

	init_mem(n_particle);

	/*	init particle system and calculate initial fitness	*/
	init_particle<<<n_blocks, n_thread>>>();

	parallel_fitness(n_particle, n_thread);
	write_system(step, calc_fitness_time, new_vel_pos_time, fitness_min_time, n_particle);

	while (step < MAX_STEPS) {
		++step;

		int n_thread_pos = SHARED_MEMORY_DIM/(sizeof(float2)*DIM) < N_THREAD_GPU ?
								SHARED_MEMORY_DIM/(sizeof(float2)*DIM) : N_THREAD_GPU;
		int n_blocks_pos = ceil((float)n_particle / N_THREAD_GPU) == 0 ? 1 : ceil((float)n_particle / N_THREAD_GPU);
		/* Compute the new velocity for each particle */
		/* Update the position of each particle, and the global fitness */
		dim3 n_t(DIM,N_THREAD_GPU/DIM);
		start_time_record(&before,&after);
		new_vel_pos<<<n_blocks_pos, n_t>>>();
		stop_time_record(&before,&after,&new_vel_pos_time);
		dim3 n_t_calc_fit(DIM,n_thread_pos/DIM);
		start_time_record(&before,&after);
		calc_fitness<<<n_blocks_pos, n_t_calc_fit, sizeof(float2)*n_thread_pos>>>();
		stop_time_record(&before,&after,&calc_fitness_time);
		/* Calculate min fitness */
		start_time_record(&before,&after);
		parallel_fitness(n_particle, n_thread);
		stop_time_record(&before,&after,&fitness_min_time);

		if (fitness_min < TARGET_FITNESS)
			break;
		if (step % STEP_CHECK_FREQ == 0) {
			write_system(step, calc_fitness_time, new_vel_pos_time, fitness_min_time, n_particle);
		}
	}
	write_system(step, calc_fitness_time, new_vel_pos_time, fitness_min_time, n_particle);
}

void write_system(const int step, const float calc_fitness_time, const float new_vel_pos, const float fitness_min, const int n_particles)
{
	float current_fitness_d;
	float global_fitness_d;
	float *current_best_pos_d_addr = (float*)malloc(sizeof(float));
	float *global_best_pos_addr = (float*)malloc(sizeof(float));
	float *current_best_pos_d = (float*)malloc(sizeof(float) * DIM);
	float *global_best_pos_d = (float*)malloc(sizeof(float) * DIM);
	float *current_fitness_d_addr = (float*)malloc(sizeof(float));
	float *global_fitness_d_addr = (float*)malloc(sizeof(float));
	int j;

	cudaGetSymbolAddress((void **)&current_fitness_d_addr, current_fitness);
	cudaGetSymbolAddress((void **)&global_fitness_d_addr, global_fitness);
	cudaGetSymbolAddress((void **)&current_best_pos_d_addr, current_best_pos);
	cudaGetSymbolAddress((void **)&global_best_pos_addr, global_best_pos);

	//check_error(cudaMemcpyFromSymbol(current_fitness_d, current_fitness_d_addr, sizeof(float)*DIM),"memory cpy to device target_pos");

	check_error(cudaMemcpy(&current_fitness_d, current_fitness_d_addr, sizeof(float),cudaMemcpyDeviceToHost),"refresh current_fitness_d host");
	check_error(cudaMemcpy(&global_fitness_d, global_fitness_d_addr, sizeof(float),cudaMemcpyDeviceToHost),"refresh global_fitness_d host");
//	check_error(cudaMemcpy(current_best_pos_d, current_best_pos_d_addr, sizeof(float)*DIM,cudaMemcpyDeviceToHost),"refresh current_best_pos host");
//	check_error(cudaMemcpy(global_best_pos_d, global_best_pos_addr, sizeof(float)*DIM,cudaMemcpyDeviceToHost),"refresh global_best_pos host");
	printf("step %u, best fitness: current %g, so far %g\n", step,
		current_fitness_d, global_fitness_d);
	if (step > 0) {
		printf("time - calc_fitness_time: %fms new_vel_pos: %fms fitness_min: %f\n",calc_fitness_time,new_vel_pos,fitness_min);
	}

	printf("\ttarget ");
	for(j = 0; j < DIM; j++){
		printf("%g,", target_pos[j]);
	}

//	printf("\n\tda aggiustarecurrent ");
//	for(j = 0; j < DIM; j++){
//		printf( "%g,", current_best_pos_d[j]);
//	}
//
//	printf("\n\tda aggiustareso far ");
//	for(j = 0; j < DIM; j++){
//		printf("%g,", global_best_pos_d[j]);
//	}
	printf("\n");

}

/* Target function to be minimized: this is the square
 * Euclidean distance from target_pos, “perturbed” by the distance
 * to the origin: this puts a local minimum at the origin,
 * which is good to test if the method actually finds the global
 * minimum or not */
__device__ float fitness(float *pos)
{
	int i;
	float fit1 = 0,fit2 = 0, dim_val;
	for(i = 0; i < DIM; i++){
		dim_val = pos[i];

		fit1 += pow(dim_val - target_pos_shared[i],2);
		fit2 += pow(dim_val,2);
	}
	return fit1*(100*fit2+1)/10;
}

/* A function that generates a random float in the given range */
float range_rand(float min, float max, uint64_t *prng_state)
{
	uint32_t r = MWC64X(prng_state);
	return min + r*((max - min)/UINT32_MAX);
}

/* Random number generation: we use the MWC64X PRNG from
 * http://cas.ee.ic.ac.uk/people/dt10/research/rngs-gpu-mwc64x.html
 * which is parallel-friendly (but needs us to keep track of the state)
 */
uint32_t MWC64X(uint64_t *state)
{
	uint64_t x = *state;
	uint32_t c = x >> 32; // the upper 32 bits
	x &= UINT32_MAX; // keep only the lower bits
	*state = x*4294883355U + c;
	return ((uint32_t)x)^c;
}

/* A functio to initialize the PRNG */
__device__ __host__ void init_rand(uint64_t *prng_state, int i)
{
	*prng_state = i;
}

/* Function to initialize a single particle at index i. */
//__global__ void init_particle()
//{
//	const int particleIndexDIM = blockIdx.x * blockDim.x + threadIdx.x * blockDim.y + threadIdx.y;
//	const int particleIndex = blockIdx.x * blockDim.x + threadIdx.x;
//	uint64_t prng_state_l;
//	if(particleIndexDIM >= num_particles*DIM)
//		return;
//	init_rand(&prng_state_l, particleIndex);
//	best_pos [particleIndexDIM] = pos[particleIndexDIM] = range_rand(coord_min, coord_max, &prng_state_l);
//	vel[particleIndexDIM] = range_rand(-coord_range, coord_range, &prng_state_l);
//
//	__syncthreads();
//	if(threadIdx.y == 0){
//		prng_state[particleIndex] = prng_state_l;
//		best_fit[particleIndex] = fitness_val[particleIndex] = fitness(pos);
//		printf("particella %d fitness %f\n", particleIndex,fitness_val[particleIndex]);
//	}
//}

__global__ void init_particle()
{
	const int particleIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int particleIndexDIM = particleIndex * DIM;
	if(particleIndex >= num_particles)
		return;
	uint64_t prng_state_l;
	init_rand(&prng_state_l, particleIndex);
	int j;

	for (j = 0; j < DIM; j++){
		best_pos[particleIndexDIM + j] = pos[particleIndexDIM + j] = range_rand(coord_min, coord_max, &prng_state_l);
	}

	for (j = 0; j < DIM; j++){
		vel[particleIndexDIM + j] = range_rand(-coord_range, coord_range, &prng_state_l);
	}

	best_fit[particleIndex] = fitness_val[particleIndex] = fitness(pos + particleIndexDIM);
	prng_state[particleIndex] = prng_state_l;
	//printf("particella %d fitness %f\n", particleIndex,fitness_val[particleIndex]);
}

/* Kernel function to compute the new position and new velocity of a given particle */
//__global__ void new_vel_pos()
//{
//	extern __shared__ float smpos[];
//	const int particleIndex = blockIdx.x * blockDim.x + threadIdx.x;
//	const int particleIndexSHM = threadIdx.x;
//	const int particleIndexDIM = particleIndex * DIM;
//	if(particleIndex >= num_particles)
//		return;
//	uint64_t prng_state_l = prng_state[particleIndex];
//
//	int i;
//	float fit1 = 0, fit2 = 0;
//
//	const float best_vec_rand_coeff = range_rand(0, 1, &prng_state_l);
//	const float global_vec_rand_coeff = range_rand(0, 1, &prng_state_l);
//	float pbest, gbest;
//	if(particleIndex == 105){
//		fit1 = 0;
//		fit1 = 0;
//	}
//	for(i = 0; i < DIM; i++){
//		float velLocal = vel[particleIndexDIM + i]*vel_omega;
//		smpos[particleIndexSHM] = pos[particleIndexDIM + i];
//		pbest =  (best_pos[particleIndexDIM + i] - smpos[particleIndexSHM]) * best_vec_rand_coeff*vel_phi_best;
//		gbest = (global_best_pos[i] - smpos[particleIndexSHM]) * global_vec_rand_coeff*vel_phi_global;
//
//		velLocal+= (pbest + gbest);
//
//		if(velLocal > coord_range) velLocal = coord_range;
//		else if (velLocal < -coord_range) velLocal = -coord_range;
//
//		smpos[particleIndexSHM] += (step_factor*velLocal);
//		if (smpos[particleIndexSHM] > coord_max) smpos[particleIndexSHM] = coord_max;
//		else if (smpos[particleIndexSHM] < coord_min) smpos[particleIndexSHM] = coord_min;
//
//		fit1 += (smpos[particleIndexSHM] - target_pos_shared[i])*(smpos[particleIndexSHM] - target_pos_shared[i]);
//		fit2 += (smpos[particleIndexSHM]*smpos[particleIndexSHM]);
//		pos[particleIndexDIM + i] = smpos[particleIndexSHM];
//		vel[particleIndexDIM + i] = velLocal;
//	}
//	printf("particella %d posizione %f, %f velocita %f, %f global_best_pos %f, %f best_pos %f, %f\n",particleIndex,
//			pos[particleIndexDIM],pos[particleIndexDIM+1],vel[particleIndexDIM],vel[particleIndexDIM+1], global_best_pos[0],global_best_pos[1],best_pos[particleIndexDIM],best_pos[particleIndexDIM+1]);
//	prng_state[particleIndex] = prng_state_l;
//	fitness_val[particleIndex] = fit1*(100*fit2+1)/10;
//	if (fitness_val[particleIndex] < best_fit[particleIndex]) {
//		best_fit[particleIndex] = fitness_val[particleIndex];
//		memcpy(best_pos + particleIndex,pos + particleIndex,sizeof(float)*DIM);
//		if(fitness_val[particleIndex] > 0.0181 && fitness_val[particleIndex] < 0.0183){
//			printf("posizione %d\n",particleIndex);
//		}
//	}
//}
__global__ void new_vel_pos()
{
	const int particleIndex = blockIdx.x * blockDim.y + threadIdx.y;
	const int particleIndexDIM = particleIndex * DIM + threadIdx.x;
	const int indexDIM = threadIdx.x;

	if(particleIndex >= num_particles)
		return;
	uint64_t prng_state_l = prng_state[particleIndex];

	const float best_vec_rand_coeff = range_rand(0, 1, &prng_state_l);
	const float global_vec_rand_coeff = range_rand(0, 1, &prng_state_l);
//	float velLocal = __ldg(&vel[particleIndexDIM])*vel_omega;
//	float posLocal = __ldg(&pos[particleIndexDIM]);
	float velLocal = vel[particleIndexDIM]*vel_omega;
	float posLocal = pos[particleIndexDIM];
	float pbest =  (best_pos[particleIndexDIM] - posLocal) * best_vec_rand_coeff*vel_phi_best;
	float gbest = (global_best_pos[indexDIM] - posLocal) * global_vec_rand_coeff*vel_phi_global;

	velLocal+= (pbest + gbest);

	if(velLocal > coord_range) velLocal = coord_range;
	else if (velLocal < -coord_range) velLocal = -coord_range;

	posLocal += (step_factor*velLocal);
	if (posLocal > coord_max) posLocal = coord_max;
	else if (posLocal < coord_min) posLocal = coord_min;

	pos[particleIndexDIM] = posLocal;
	vel[particleIndexDIM] = velLocal;
	if(indexDIM == 1)printf("particle %d vel %i %f \n",particleIndex,indexDIM,velLocal);
	prng_state[particleIndex] = prng_state_l;
}
//__global__ void calc_fitness()
//{
//	extern __shared__ float2 smpos[];
//	const int particleIndex = blockIdx.x * blockDim.y + threadIdx.y;
//	const int particleIndexSHM = threadIdx.y * blockDim.x + threadIdx.x;
//	const int particleIndexDIM = particleIndex * DIM + threadIdx.x;
//	const int indexDIM = threadIdx.x;
//	int stride = blockDim.x/2;
//
//	if(particleIndex >= num_particles)
//		return;
//
//	float posLocal = pos[particleIndexDIM];
//	smpos[particleIndexSHM].x = (posLocal - target_pos_shared[indexDIM])*(posLocal - target_pos_shared[indexDIM]);
//	smpos[particleIndexSHM].y = (posLocal*posLocal);
//	__syncthreads();
//
//	for (;stride>0;stride>>=1)
//	{
//		if (indexDIM<stride){
//			smpos[particleIndexSHM].x += smpos[particleIndexSHM+stride].x;
//			smpos[particleIndexSHM].y += smpos[particleIndexSHM+stride].y;
//		}
//		__syncthreads();
//	}
//
//	if (indexDIM==0){
//		float fitness = smpos[particleIndexSHM].x*(100*smpos[particleIndexSHM].y+1)/10;
//		//printf("particle %d dim %f\n",particleIndex,fitness);
//		fitness_val[particleIndex] = fitness;
//		if (fitness < best_fit[particleIndex]) {
//			best_fit[particleIndex] = fitness;
//			memcpy(best_pos + particleIndex,pos + particleIndex,sizeof(float)*DIM);
////			if(fitness_val[particleIndex] > 0.0181 && fitness_val[particleIndex] < 0.0183){
////				printf("posizione %d\n",particleIndex);
////			}
//		}
//	}
//
//}


__global__ void calc_fitness()
{
	extern __shared__ float2 smpos[];
	const int particleIndex = blockIdx.x * blockDim.y + threadIdx.y;
	const int particleIndexSHM = threadIdx.y * blockDim.x + threadIdx.x;
	const int particleIndexDIM = particleIndex * DIM + threadIdx.x;
	const int indexDIM = threadIdx.x;

	if(particleIndex >= num_particles)
		return;

	float posLocal = pos[particleIndexDIM];
	smpos[particleIndexSHM].x = (posLocal - target_pos_shared[indexDIM])*(posLocal - target_pos_shared[indexDIM]);
	smpos[particleIndexSHM].y = (posLocal*posLocal);

	warp_control_float2(smpos,particleIndexSHM);

	if (indexDIM==0){
		float fitness = smpos[particleIndexSHM].x*(100*smpos[particleIndexSHM].y+1)/10;
		//printf("particle %d dim %f\n",particleIndex,fitness);
		fitness_val[particleIndex] = fitness;
		if (fitness < best_fit[particleIndex]) {
			best_fit[particleIndex] = fitness;
			memcpy(best_pos + particleIndex,pos + particleIndex,sizeof(float)*DIM);
//			if(fitness_val[particleIndex] > 0.0181 && fitness_val[particleIndex] < 0.0183){
//				printf("posizione %d\n",particleIndex);
//			}
		}
	}
}

__device__ void warp_control_float2(float2* smpos, int particleIndexSHM)
{
	__syncthreads();
	#if DIM > 1024
	for (int stride = blockDim.x/2;stride>0;stride>>=1)
	{
		if (indexDIM<stride){
			smpos[particleIndexSHM].x += smpos[particleIndexSHM+stride].x;
			smpos[particleIndexSHM].y += smpos[particleIndexSHM+stride].y;
		}
		__syncthreads();
	}
	#else
	if (particleIndexSHM < DIM/2)
	{
		#if DIM >= 512
		smpos[particleIndexSHM].x += smpos[particleIndexSHM + 256].x;
		smpos[particleIndexSHM].y += smpos[particleIndexSHM + 256].y;
		__syncthreads();
		#endif
		#if DIM >= 256
		smpos[particleIndexSHM].x += smpos[particleIndexSHM + 128].x;
		smpos[particleIndexSHM].y += smpos[particleIndexSHM + 128].y;
		__syncthreads();
		#endif
		#if DIM >= 128
		smpos[particleIndexSHM].x += smpos[particleIndexSHM + 64].x;
		smpos[particleIndexSHM].y += smpos[particleIndexSHM + 64].y;
		__syncthreads();
		#endif
		#if DIM >= 64
		smpos[particleIndexSHM].x += smpos[particleIndexSHM + 32].x;
		smpos[particleIndexSHM].y += smpos[particleIndexSHM + 32].y;
		__syncthreads();
		#endif
		#if DIM >= 32
		smpos[particleIndexSHM].x += smpos[particleIndexSHM + 16].x;
		smpos[particleIndexSHM].y += smpos[particleIndexSHM + 16].y;
		__syncthreads();
		#endif
		#if DIM >= 16
		smpos[particleIndexSHM].x += smpos[particleIndexSHM + 8].x;
		smpos[particleIndexSHM].y += smpos[particleIndexSHM + 8].y;
		__syncthreads();
		#endif
		#if DIM >= 8
		smpos[particleIndexSHM].x += smpos[particleIndexSHM + 4].x;
		smpos[particleIndexSHM].y += smpos[particleIndexSHM + 4].y;
		__syncthreads();
		#endif
		#if DIM >= 4
		smpos[particleIndexSHM].x += smpos[particleIndexSHM + 2].x;
		smpos[particleIndexSHM].y += smpos[particleIndexSHM + 2].y;
		__syncthreads();
		#endif
		#if DIM >= 2
		smpos[particleIndexSHM].x += smpos[particleIndexSHM + 1].x;
		smpos[particleIndexSHM].y += smpos[particleIndexSHM + 1].y;
		__syncthreads();
		#endif
	}
	#endif
}

__device__ void warp_control_float(float* smpos, int particleIndexSHM)
{
	__syncthreads();
	#if DIM > 1024
	for (int stride = blockDim.x/2;stride>0;stride>>=1)
	{
		if (indexDIM<stride){
			smpos[particleIndexSHM] += smpos[particleIndexSHM+stride];
		}
		__syncthreads();
	}
	#else
	if (particleIndexSHM < DIM/2)
	{
		#if DIM >= 512
		smpos[particleIndexSHM] += smpos[particleIndexSHM + 256];
		__syncthreads();
		#endif
		#if DIM >= 256
		smpos[particleIndexSHM] += smpos[particleIndexSHM + 128];
		__syncthreads();
		#endif
		#if DIM >= 128
		smpos[particleIndexSHM] += smpos[particleIndexSHM + 64];
		__syncthreads();
		#endif
		#if DIM >= 64
		smpos[particleIndexSHM] += smpos[particleIndexSHM + 32];
		__syncthreads();
		#endif
		#if DIM >= 32
		smpos[particleIndexSHM] += smpos[particleIndexSHM + 16];
		__syncthreads();
		#endif
		#if DIM >= 16
		smpos[particleIndexSHM] += smpos[particleIndexSHM + 8];
		__syncthreads();
		#endif
		#if DIM >= 8
		smpos[particleIndexSHM] += smpos[particleIndexSHM + 4];
		__syncthreads();
		#endif
		#if DIM >= 4
		smpos[particleIndexSHM] += smpos[particleIndexSHM + 2];
		__syncthreads();
		#endif
		#if DIM >= 2
		smpos[particleIndexSHM] += smpos[particleIndexSHM + 1];
		__syncthreads();
		#endif
	}
	#endif
}
/* Kernel function to compute the new global min fitness */
__global__ void find_min_fitness_parallel(__restrict__ const fitness_pos* in, fitness_pos* out,const int offset,const int n_in,const int blocks){
	extern __shared__ fitness_pos sm[];
	const int tid=threadIdx.x;
	const int i=(blockIdx.x*blockDim.x+threadIdx.x) + (offset*blockDim.x);
	int stride;
	sm[tid].fitness = HUGE_VALF;
	if(i >= num_particles || i >= n_in)
			return;
	if(in != NULL){
		sm[tid] = in[i];
	}else{
		sm[tid].fitness = fitness_val[i];
		sm[tid].pos = i;
	}

	//copy to SM
	for (stride = blockDim.x/2;stride>0;stride>>=1)
	{
		__syncthreads();
		if (tid<stride && sm[tid].fitness > sm[tid+stride].fitness){
			sm[tid] = sm[tid+stride];
		}
	}

	if (tid==0){
		out[blockIdx.x+offset]=sm[0];//copy back
		if(blocks == 1){
			current_fitness = sm[0].fitness;
			memcpy(current_best_pos,pos+sm[0].pos*DIM,sizeof(float)*DIM);

			if (sm[0].fitness < global_fitness) {
				global_fitness = sm[0].fitness;
				memcpy(global_best_pos,current_best_pos,sizeof(float)*DIM);
			}
		}
	}
}

void start_time_record(cudaEvent_t *before, cudaEvent_t *after){
	check_error(cudaEventCreate(&(*before)),"create cudaEvent before");
	check_error(cudaEventCreate(&(*after)),"create cudaEvent after");
	check_error(cudaEventRecord(*before),"record cudaEvent before");
}

void stop_time_record(cudaEvent_t *before, cudaEvent_t *after, float *runtime){
	check_error(cudaEventRecord(*after),"record cudaEvent after");
	check_error(cudaEventSynchronize(*after),"synch cudaEvent before");
	check_error(cudaEventElapsedTime(runtime, *before, *after),"calc cudaEvent elapsed time");
}

void check_error(cudaError_t err, const char *msg)
{
  if (err != cudaSuccess) {
    fprintf(stderr, "%s : errore %d (%s)\n",
      msg, err, cudaGetErrorString(err));
    exit(err);
  }
}

/* function to find the ceil of a log2 x value*/
int ceil_log2(unsigned long long x)
{
  static const unsigned long long t[6] = {
    0xFFFFFFFF00000000ull,
    0x00000000FFFF0000ull,
    0x000000000000FF00ull,
    0x00000000000000F0ull,
    0x000000000000000Cull,
    0x0000000000000002ull
  };

  int y = (((x & (x - 1)) == 0) ? 0 : 1);
  int j = 32;
  int i;

  for (i = 0; i < 6; i++) {
    int k = (((x & t[i]) == 0) ? 0 : j);
    y += k;
    x >>= k;
    j >>= 1;
  }

  return 1<<y;
}

/*	Function to handle the Kernel function find_min_fitness_parallel to don't let the shared memory become full */
void parallel_fitness(const int n_particle, int n_thread){
	int shmdim;
	fitness_pos *fitness_device_out,*fitness_device_in = NULL;
	int last_n_block;
	int offset;
	int blocks = n_particle;
	int max_parallel_particle_iteration = SHARED_MEMORY_DIM / sizeof(fitness_pos);
	int iteration;
	int max_blocks_per_iteration = max_parallel_particle_iteration / n_thread;
	while(blocks != 1){
		offset = 0;
		last_n_block = blocks;
		blocks = ceil((float)blocks / n_thread);
		if(blocks == 1){
			n_thread = ceil_log2(last_n_block);
		}
		cudaMalloc(&fitness_device_out, sizeof(fitness_pos) * blocks);
		shmdim = n_thread*sizeof(fitness_pos);
		if(max_parallel_particle_iteration < last_n_block && 0){
			iteration = 0;
			while(iteration + max_parallel_particle_iteration < blocks*n_thread){
				find_min_fitness_parallel<<<max_blocks_per_iteration, n_thread,shmdim>>>
									(fitness_device_in, fitness_device_out, offset, last_n_block, blocks);
				iteration += max_parallel_particle_iteration;
				offset += (max_parallel_particle_iteration/n_thread);

			}
			int x = (blocks*n_thread) - (offset*n_thread);
			x = ceil((float)x / n_thread);
			find_min_fitness_parallel<<<x, n_thread,shmdim>>>
							(fitness_device_in, fitness_device_out, offset, last_n_block, blocks);
		}else{
			find_min_fitness_parallel<<<blocks, n_thread,shmdim>>>
								(fitness_device_in, fitness_device_out, offset, last_n_block,blocks);
		}

		if(fitness_device_in != NULL){
			check_error(cudaFree(fitness_device_in),"free fitness_device_in");
		}
		fitness_device_in = fitness_device_out;
	}
	fitness_device_out = (fitness_pos*)malloc(sizeof(fitness_pos));
	check_error(cudaMemcpy(fitness_device_out, fitness_device_in, sizeof(fitness_pos),cudaMemcpyDeviceToHost),"copy fitness_min");
	fitness_min = fitness_device_out->fitness;
	check_error(cudaFree(fitness_device_in),"free fitness_device_out");
	free(fitness_device_out);
}
