#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <float.h>
#include <errno.h>
#include <stdint.h>
#include <cuda_runtime_api.h>
#define DIM 16
#define SHARED_MEMORY_DIM ((1<<15)+(1<<14)) // 48KB
#define N_THREAD_GPU (1<<8) // limit is 1024

#define MAX_STEPS (1<<20) /* run for no more than 1Mi steps */
#define TARGET_FITNESS (FLT_EPSILON) /* or until the fitness is less than this much */
#define STEP_CHECK_FREQ 50 /* after how many steps to write the system and check the time */

/* n-dimensional space */
typedef struct floatN
{
	int n;
	float dim[DIM];
} floatN;

/*	needed for find fitness min in parallel	*/
typedef struct fitness_pos
{
	int pos;
	float fitness;
} fitness_pos;

/* A particle */
typedef struct Particle
{
	floatN pos;
	floatN vel;
	floatN best_pos;
	float best_fit;
	float fitness;
	uint64_t prng_state;
} Particle;

/* The whole particle system */
typedef struct ParticleSystem
{
	Particle *particle;
	floatN current_best_pos;
	floatN global_best_pos;
	int num_particles;
	int dim_particles;
	float global_fitness;
	float current_fitness;
} ParticleSystem;

/* Extents of the domain in each dimension */
#define coord_min -1
#define coord_max 1
#define coord_range 2
floatN target_pos;	/* The target position */
__device__ floatN target_pos_shared;



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
__device__ float fitness(const floatN *pos);
__global__ void init_particle(ParticleSystem *ps);
__global__ void find_min_fitness_parallel(ParticleSystem *ps, fitness_pos* in, fitness_pos* out, int offset, int n_in, int n_blocks);
__global__ void new_vel(ParticleSystem *ps);
__global__ void new_pos(ParticleSystem *ps);
int ceil_log2(unsigned long long x);
void check_error(cudaError_t err, const char *msg);
void start_time_record(cudaEvent_t *before, cudaEvent_t *after);
void stop_time_record(cudaEvent_t *before, cudaEvent_t *after, float *runtime);
void parallel_fitness(ParticleSystem *ps, int n_particle, int n_thread);
void write_system(ParticleSystem **psdb, int step, float new_vel, float new_pos, float fitness_min);

int main(int argc, char *argv[])
{
	ParticleSystem *ps;
	ParticleSystem *psHost;
	Particle *devicePointer;
	unsigned step = 0;
	unsigned n_particle;
	cudaEvent_t before, after;
	float new_vel_time = 0, new_pos_time = 0, fitness_min_time = 0;
	float fitness_min;
	int j;
	int n_blocks;
	int n_dimensions = DIM;
	int n_thread = N_THREAD_GPU;
	uint64_t prng_state;

	/*	Get particle's numbers, default 128	*/
	n_particle = argc > 1 ? atoi(argv[1]) : 128;

	/*	Define n blocks for GPU parallelization	*/
	n_blocks = ceil((float)n_particle / n_thread) == 0 ? 1 : ceil((float)n_particle / n_thread);

	/* Initialize the target position */
	init_rand(&prng_state, time(NULL));
	target_pos.n = n_dimensions;

	printf("target position: (");
	for(j = 0; j < target_pos.n; j++){
		target_pos.dim[j] = range_rand(coord_min, coord_max, &prng_state);
		printf("%f,", target_pos.dim[j]);
	}
	printf(")\n");

	check_error(cudaMemcpyToSymbol(target_pos_shared, &target_pos, sizeof(floatN)),"memory cpy to device target_pos");

	/* Initialize a system with the number of particles given
	 * on the command-line or from default value (128) */

	check_error(cudaMalloc(&devicePointer,sizeof(Particle) * n_particle),"memory alloc n particle");
	psHost = (ParticleSystem*)malloc(sizeof(ParticleSystem));
	psHost->particle = devicePointer;
	psHost->dim_particles = n_dimensions;
	psHost->num_particles = n_particle;
	psHost->global_fitness = HUGE_VALF;
	check_error(cudaMalloc(&ps,sizeof(ParticleSystem)),"memory alloc ps");
	check_error(cudaMemcpy(ps, psHost, sizeof(ParticleSystem),cudaMemcpyHostToDevice),"memory cpy ps hostToDevice");

	/*	init particle system and calculate initial fitness	*/
	init_particle<<<n_blocks, n_thread>>>(ps);
	parallel_fitness(ps, n_particle, n_thread);

	check_error(cudaMemcpy(psHost, ps, sizeof(ParticleSystem),cudaMemcpyDeviceToHost),"refresh PS host");
	write_system(&psHost, step, new_vel_time, new_pos_time, fitness_min_time);

	while (step < MAX_STEPS) {
		++step;


		/* Compute the new velocity for each particle */
		start_time_record(&before,&after);
		new_vel<<<n_blocks, n_thread>>>(ps);
		stop_time_record(&before,&after,&new_vel_time);

		int n_thread_pos = SHARED_MEMORY_DIM/(sizeof(float)*DIM) < N_THREAD_GPU ?
								SHARED_MEMORY_DIM/(sizeof(float)*DIM) : N_THREAD_GPU;
		int n_blocks_pos = ceil((float)n_particle / n_thread_pos) == 0 ? 1 : ceil((float)n_particle / n_thread_pos);

		/* Update the position of each particle, and the global fitness */
		start_time_record(&before,&after);
		new_pos<<<n_blocks_pos, n_thread_pos, sizeof(float)*n_thread_pos>>>(ps);
		stop_time_record(&before,&after,&new_pos_time);

		/* Calculate min fitness */
		start_time_record(&before,&after);
		parallel_fitness(ps, n_particle, n_thread);
		stop_time_record(&before,&after,&fitness_min_time);

	    check_error(cudaMemcpy(psHost, ps, sizeof(ParticleSystem),cudaMemcpyDeviceToHost),"refresh PS host");

	    fitness_min = psHost->current_fitness;
		if (fitness_min < TARGET_FITNESS)
			break;
		if (step % STEP_CHECK_FREQ == 0) {
			write_system(&psHost, step, new_vel_time, new_pos_time, fitness_min_time);
		}
	}
	write_system(&psHost, step, new_vel_time, new_pos_time, fitness_min_time);
	free(psHost);
	check_error(cudaFree(ps),"free ps");
	check_error(cudaFree(devicePointer),"free devicePointer");
}

void write_system(ParticleSystem **psdb, int step, float new_vel, float new_pos, float fitness_min)
{
	int j;
	ParticleSystem *ps = (*psdb);


	printf("step %u, best fitness: current %g, so far %g\n", step,
		ps->current_fitness, ps->global_fitness);
	if (step > 0) {
		printf("time - new_vel: %fms new_pos: %fms fitness_min: %f\n",new_vel,new_pos,fitness_min);
	}

	printf("\ttarget ");
	for(j = 0; j < target_pos.n; j++){
		printf("%g,", target_pos.dim[j]);
	}

	printf("\n\tcurrent ");
	for(j = 0; j < ps->current_best_pos.n; j++){
		printf( "%g,", ps->current_best_pos.dim[j]);
	}

	printf("\n\tso far ");
	for(j = 0; j < ps->global_best_pos.n; j++){
		printf("%g,", ps->global_best_pos.dim[j]);
	}
	printf("\n");

}

/* Target function to be minimized: this is the square
 * Euclidean distance from target_pos, “perturbed” by the distance
 * to the origin: this puts a local minimum at the origin,
 * which is good to test if the method actually finds the global
 * minimum or not */
__device__ float fitness(const floatN *pos)
{
	int i;
	float fit1 = 0,fit2 = 0, dim_val;
	for(i = 0; i < pos->n; i++){
		dim_val = pos->dim[i];

		fit1 += pow(dim_val - target_pos_shared.dim[i],2);
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
__global__ void init_particle(ParticleSystem *ps)
{
	int particleIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if(particleIndex >= ps->num_particles)
		return;
	uint64_t prng_state;
	init_rand(&prng_state, particleIndex);
	floatN pos;
	int j;
	int nDim = ps->dim_particles;
	Particle *p = &ps->particle[particleIndex];

	for (j = 0; j < nDim; j++){
		pos.dim[j] = ps->particle[particleIndex].pos.dim[j] = range_rand(coord_min, coord_max, &prng_state);
	}
	for (j = 0; j < nDim; j++){
		ps->particle[particleIndex].vel.dim[j] = range_rand(-coord_range, coord_range, &prng_state);
	}
	pos.n = ps->particle[particleIndex].pos.n = ps->particle[particleIndex].vel.n = nDim;
	p->best_pos = pos;
	p->best_fit = p->fitness = fitness(&pos);
	p->prng_state = prng_state;
}

/* Kernel function to compute the new velocity of a given particle */
__global__ void new_vel(ParticleSystem *ps)
{
	int particleIndex = (blockIdx.x * blockDim.x + threadIdx.x);
	if(particleIndex >= ps->num_particles)
		return;

	int dim;
	Particle *p = &ps->particle[particleIndex];
	floatN *nvel = &p->vel;
	const float best_vec_rand_coeff = range_rand(0, 1, &p->prng_state);
	const float global_vec_rand_coeff = range_rand(0, 1, &p->prng_state);

	for(dim = 0; dim < DIM;dim++){
		float pbest =  p->best_pos.dim[dim] - p->pos.dim[dim];
		float gbest = ps->global_best_pos.dim[dim] - p->pos.dim[dim];

		nvel->dim[dim] = vel_omega*nvel->dim[dim] + best_vec_rand_coeff*vel_phi_best*pbest +
				  global_vec_rand_coeff*vel_phi_global*gbest;

		if(nvel->dim[dim] > coord_range) nvel->dim[dim] = coord_range;
		else if (nvel->dim[dim] < -coord_range) nvel->dim[dim] = -coord_range;
	}
}

/* Kernel function to compute the new position of a given particle */
__global__ void new_pos(ParticleSystem *ps)
{
	extern __shared__ float smpos[];
	int particleIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int particleIndexSHM = threadIdx.x;
	if(particleIndex >= ps->num_particles)
		return;
	Particle *p = &ps->particle[particleIndex];

	int i;
	float fit1 = 0, fit2 = 0;
	for(i = 0; i < DIM; i++){
		smpos[particleIndexSHM] = p->pos.dim[i] + (step_factor*p->vel.dim[i]);
		if (smpos[particleIndexSHM] > coord_max) smpos[particleIndexSHM] = coord_max;
		else if (smpos[particleIndexSHM] < coord_min) smpos[particleIndexSHM] = coord_min;

		fit1 += (smpos[particleIndexSHM] - target_pos_shared.dim[i])*(smpos[particleIndexSHM] - target_pos_shared.dim[i]);
		fit2 += (smpos[particleIndexSHM]*smpos[particleIndexSHM]);
		p->pos.dim[i] = smpos[particleIndexSHM];
	}

	p->fitness = fit1*(100*fit2+1)/10;
	if (p->fitness < p->best_fit) {
		p->best_fit = p->fitness;
		p->best_pos = p->pos;
	}
}

/* Kernel function to compute the new global min fitness */
__global__ void find_min_fitness_parallel(ParticleSystem *ps, fitness_pos* in, fitness_pos* out, int offset, int n_in, int blocks){
	extern __shared__ fitness_pos sm[];
	int tid=threadIdx.x;
	uint i=(blockIdx.x*blockDim.x+threadIdx.x) + (offset*blockDim.x);
	int stride;
	sm[tid].fitness = HUGE_VALF;
	if(i >= ps->num_particles || i >= n_in)
			return;

	if(in != NULL){
		sm[tid] = in[i];
	}else{
		sm[tid].fitness = (ps->particle + i)->fitness;
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
			ps->current_fitness = sm[0].fitness;
			ps->current_best_pos = ps->particle[sm[0].pos].pos;

			if (sm[0].fitness < ps->global_fitness) {
				ps->global_fitness = sm[0].fitness;
				ps->global_best_pos = ps->current_best_pos;
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
void parallel_fitness(ParticleSystem *ps, int n_particle, int n_thread){
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
		if(max_parallel_particle_iteration < last_n_block){
			iteration = 0;
			while(iteration + max_parallel_particle_iteration < blocks*n_thread){
				find_min_fitness_parallel<<<max_blocks_per_iteration, n_thread,shmdim>>>
									(ps, fitness_device_in, fitness_device_out, offset, last_n_block, blocks);
				iteration += max_parallel_particle_iteration;
				offset += (max_parallel_particle_iteration/n_thread);

			}
			int x = (blocks*n_thread) - (offset*n_thread);
			x = ceil((float)x / n_thread);
			find_min_fitness_parallel<<<x, n_thread,shmdim>>>
							(ps, fitness_device_in, fitness_device_out, offset, last_n_block, blocks);
		}else{
			find_min_fitness_parallel<<<blocks, n_thread,shmdim>>>
								(ps, fitness_device_in, fitness_device_out, offset, last_n_block,blocks);
		}

		if(fitness_device_in != NULL){
			check_error(cudaFree(fitness_device_in),"free fitness_device_in");
		}
		fitness_device_in = fitness_device_out;
	}
	check_error(cudaFree(fitness_device_out),"free fitness_device_out");
}
