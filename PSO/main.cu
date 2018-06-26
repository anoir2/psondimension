/* Particle Swarm Optimization
 *
 * This is a simple implementation of the Particle Swarm Optimization method:
 * we assume given a system of particles (identified by their coordinates in a
 * given problem space) and a target function to be minimized.
 *
 * For each particle, we keep track of a 'velocity' and the best position it
 * has achieved so far. We also track the best position achieved by the whole
 * system.
 *
 * At each iteration, we compute a new 'velocity' as a linear combination of:
 *  - the velocity at the previous step
 *  - the distance vector to the particle's best position
 *  - the distance vector to the global best position
 *
 * The initial position and velocity of each particle, as well as the
 * weight of the distance vector contributions to the velocity, are random.
 *
 * In our case, we assume that the target function is the distance
 * from a given target point (perturbed), that the domain has DIM dimensions,
 * and that each coordinate can vary between -1 and 1
 *
 * This version is specialized (and optimized) for 2 dimensions, named x and y.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <errno.h>
#include <stdint.h>
#include <cuda_runtime_api.h>
#define DIM 2
#define SHARED_MEMORY_DIM (1<<15)+(1<<10) // 48KB

/* A vector in 2-dimensional space */
typedef struct floatN
{
	int n;
	float dim[DIM];
} floatN;

typedef struct fitness_pos
{
	int pos;
	float fitness;
} fitness_pos;

/* A particle */
typedef struct Particle
{
	floatN pos; /* particle position */
	floatN vel; /* particle velocity */
	floatN best_pos; /* best position so far */
	float best_fit; /* fitness value of the best position so far */
	float fitness; /* fitness value of the current position */
	uint64_t prng_state; /* state of the PRNG for this particle */
} Particle;

/* The whole particle system */
typedef struct ParticleSystem
{
	Particle *particle; /* array of particles */
	floatN current_best_pos; /* best position in the whole system a this iteration */
	floatN global_best_pos; /* best position in the whole system so far */
	int num_particles; /* number of particles */
	int dim_particles;
	float global_fitness; /* fitness value for global_best_pos */
	float current_fitness; /* fitness value for global_best_pos */
} ParticleSystem;

/* Extents of the domain in each dimension */
static const float coord_min = -1;
static const float coord_max = 1;
static const float coord_range = 2; // coord_max - coord_min, but the expression is not const in C 8-/
floatN target_pos;	/* The target position */
__device__ floatN target_pos_shared;



/* Overall weight for the old velocity, best position distance and global
 * best position distance in the computation of the new velocity
 */
__const__ float vel_omega = .9;
__const__ float vel_phi_best = 2;
__const__ float vel_phi_global = 2;

/* The contribution of the velocity to the new position. Set to 1
 * to use the standard PSO approach of adding the whole velocity
 * to the position.
 * In my experience a smaller factor speeds up convergence
 */
__const__ float step_factor = 1;

__device__ __host__ uint32_t MWC64X(uint64_t *state);
__device__ __host__ float range_rand(float min, float max, uint64_t *prng_state);
__device__ __host__ void init_rand(uint64_t *prng_state, int i);
__device__ float fitness(const floatN *pos);
__device__ void warp_reduce_min( float smem[64]);
__global__ void init_particle(ParticleSystem *ps);
__global__ void find_min_fitness(ParticleSystem *ps);
__global__ void find_min_fitness_parallel(ParticleSystem *ps, float* in, float* out, int* in_pos, int* out_pos, int n_in);
__global__ void new_vel(ParticleSystem *ps);
__global__ void new_pos(ParticleSystem *ps);

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


void parallel_fitness(ParticleSystem *ps, int n_particle, int n_thread){
	int shmdim;
	float *fitness_device_out,*fitness_device_in = NULL;
	int *fitness_device_pos_out, *fitness_device_pos_in = NULL, *check;
	int last_n_block;
	int blocks = n_particle;
	int max_block_for_iteration = SHARED_MEMORY_DIM / sizeof(fitness_pos);
	while(blocks != 1){
		last_n_block = blocks;
		blocks = ceil((float)blocks / n_thread);
		if(blocks == 1){
			n_thread = ceil_log2(last_n_block);
		}
		shmdim = n_thread*sizeof(float);
		cudaMalloc(&fitness_device_out, sizeof(float) * blocks);
		cudaMalloc(&fitness_device_pos_out, sizeof(int) * blocks);
		find_min_fitness_parallel<<<blocks, n_thread,shmdim>>>
					(ps, fitness_device_in, fitness_device_out, fitness_device_pos_in, fitness_device_pos_out, last_n_block);
		if(fitness_device_in != NULL){
			cudaFree(fitness_device_in);
			cudaFree(fitness_device_pos_in);
		}
		fitness_device_in = fitness_device_out;
		fitness_device_pos_in = fitness_device_pos_out;
	}
	fitness_device_out = (float*)malloc(sizeof(float));
	fitness_device_pos_out = (int*)malloc(sizeof(int));
	cudaMemcpy(fitness_device_out, fitness_device_in, sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(fitness_device_pos_out, fitness_device_pos_in, sizeof(int),cudaMemcpyDeviceToHost);

	printf("risultato fitness parallela %f posizione %d \n", *fitness_device_out, *fitness_device_pos_out);
}

void write_system(ParticleSystem **ps, int step);
void check_error(cudaError_t err, const char *msg)
{
  if (err != cudaSuccess) {
    fprintf(stderr, "%s : errore %d (%s)\n",
      msg, err, cudaGetErrorString(err));
    exit(err);
  }
}
int main(int argc, char *argv[])
{
	ParticleSystem *ps;
	ParticleSystem *hostPS;
	Particle *devicePointer;
	//Particle* devicePointer;
	unsigned step = 0;
	int n_particle = argc > 1 ? atoi(argv[1]) : 129;

//#define MAX_STEPS (1<<20) /* run for no more than 1Mi steps */
//#define TARGET_FITNESS (FLT_EPSILON) /* or until the fitness is less than this much */
//#define STEP_CHECK_FREQ 100 /* after how many steps to write the system and check the time */
//	int dimensions = DIM;
//	/* Initialize the target position */
//	uint64_t prng_state;
//	init_rand(&prng_state, time(NULL));
//	target_pos.n = dimensions;
//	printf("target position: (");
//	for(j = 0; j < target_pos.n; j++){
//		target_pos.dim[j] = range_rand(coord_min, coord_max, &prng_state);
//		printf("%g,", target_pos.dim[j]);
//	}
//	printf(")\n");
	#define MAX_STEPS (200) /* run for no more than 1Mi steps */
	#define TARGET_FITNESS (FLT_EPSILON) /* or until the fitness is less than this much */
	#define STEP_CHECK_FREQ 1 /* after how many steps to write the system and check the time */
	int n_dimensions = DIM;
	int n_thread = 32;
	int n_blocks = ceil((float)n_particle / n_thread) == 0 ? 1 : ceil((float)n_particle / n_thread);

	dim3 block_thread(n_thread,n_dimensions,1);

	/* Initialize the target position */
	uint64_t prng_state;
	init_rand(&prng_state, time(NULL));
//	target_pos.x = range_rand(coord_min, coord_max, &prng_state);
//	target_pos.y = range_rand(coord_min, coord_max, &prng_state);
//	target_pos.n = n_dimensions;

	target_pos.n = n_dimensions;
	target_pos.dim[0] = -0.287776;
	target_pos.dim[1] = 0.520416;
	cudaMemcpyToSymbol(target_pos_shared, &target_pos, sizeof(floatN));

	/* Initialize a system with the number of particles given
	 * on the command-line, defaulting to something which is
	 * related to the number of dimensions */



	cudaMalloc(&devicePointer,sizeof(Particle) * n_particle);
	hostPS = (ParticleSystem*)malloc(sizeof(ParticleSystem));
	hostPS->particle = devicePointer;
	hostPS->dim_particles = n_dimensions;
	hostPS->num_particles = n_particle;
	hostPS->global_fitness = HUGE_VALF;
	cudaMalloc(&ps,sizeof(ParticleSystem));
	cudaMemcpy(ps, hostPS, sizeof(ParticleSystem),cudaMemcpyHostToDevice);

	//init_system(&ps,n_particle, dimensions);
	//ps->num_particles = n_particle;
	/* Allocate array of particles */
	//Particle* hostPointer = (Particle*)malloc(n_particle*sizeof(Particle));
	//cudaMalloc(&devicePointer,sizeof(Particle) * n_particle);
	/* Compute the current fitness and best position */

	init_particle<<<n_blocks, n_thread>>>(ps);
	parallel_fitness(ps, n_particle, n_thread);
	find_min_fitness<<< 1,1 >>>(ps);

	ParticleSystem *psHost = (ParticleSystem*)malloc(sizeof(ParticleSystem));
    cudaMemcpy(psHost, ps, sizeof(ParticleSystem),cudaMemcpyDeviceToHost);
	// end init system


	write_system(&psHost, step);

	while (step < MAX_STEPS) {
		++step;

		/* Compute the new velocity for each particle */

		cudaEvent_t before_init_particle, after_init_particle;
		cudaEventCreate(&before_init_particle);
		cudaEventCreate(&after_init_particle);
		cudaEventRecord(before_init_particle);
		new_vel<<<n_blocks, block_thread>>>(ps);
		cudaEventRecord(after_init_particle);
		cudaEventSynchronize(after_init_particle);
		float runtime;
		cudaEventElapsedTime(&runtime, before_init_particle, after_init_particle);
		printf("step %d new_vel tempo %f\n",step, runtime);


		/* Update the position of each particle, and the global fitness */
		cudaEventCreate(&before_init_particle);
		cudaEventCreate(&after_init_particle);
		cudaEventRecord(before_init_particle);
		new_pos<<<n_blocks, n_thread>>>(ps);
		cudaEventRecord(after_init_particle);
		cudaEventSynchronize(after_init_particle);
		cudaEventElapsedTime(&runtime, before_init_particle, after_init_particle);
		printf("step %d new_pos tempo %f\n",step, runtime);
		//new_pos_vel<<<n_particle, 1>>>(ps);
		find_min_fitness<<<1,1>>>(ps);
	    cudaMemcpy(psHost, ps, sizeof(ParticleSystem),cudaMemcpyDeviceToHost);
	    float fitness_min = psHost->current_fitness;
		if (fitness_min < TARGET_FITNESS)
			break;
//		if (step % STEP_CHECK_FREQ == 0) {
//			write_system(&psHost, step);
//		}
	}
	write_system(&psHost, step);
}

void write_csv(ParticleSystem *ps, int step)
{
#define FNSZ 1024 /* maximum size for the file name */
	char fname[FNSZ+1];
	fname[FNSZ] = '\0';
	int j;
	snprintf(fname, FNSZ, "pso.%u.csv", step);

	FILE *fp = fopen(fname, "w");
	if (!fp) {
		fprintf(stderr, "failed to open %s - %d: %s\n", fname, errno, strerror(errno));
		exit(1);
	}

	/* Write header */
	fprintf(fp, "x,y,vx,vy,current_fitness,bx,by,best_fit");

	/* Write out each particle */
	floatN vec;
	for (int i = 0; i < ps->num_particles; ++i)
	{
		const Particle *p = ps->particle + i;

		vec = p->pos;
		for(j = 0; j < vec.n; j++){
			fprintf(fp, "%g,", vec.dim[j]);
		}

		vec = p->vel;
		for(j = 0; j < vec.n; j++){
			fprintf(fp, "%g,", vec.dim[j]);
		}

		fprintf(fp, "%g,", p->fitness);

		vec = p->best_pos;
		for(j = 0; j < vec.n; j++){
			fprintf(fp, "%g,", vec.dim[j]);
		}

		fprintf(fp, "%g\n", p->best_fit);
	}

	/* Write out the target point */
	vec = target_pos;
	for(j = 0; j < vec.n; j++){
		fprintf(fp, "%g,", vec.dim[j]);
	}

	for(j = 0; j < vec.n; j++){
		fprintf(fp, "%g,", 0.0);
	}
	fprintf(fp, "%g,", 0.0); /* null fitness */
	vec = target_pos;
	for(j = 0; j < vec.n; j++){
		fprintf(fp, "%g,", vec.dim[j]); /* best pos is the same as target pos */
	}
	fprintf(fp, "%g\n", 0.0); /* null best fit */

	fclose(fp);
}

/* Save the particle system to disk */
void write_system(ParticleSystem **psdb, int step)
{
	/* Step at which we wrote last time */
	static int last_step = 0;
	/* structs used for timing */
	static struct timespec start;
	static struct timespec stop;
	int j;
	ParticleSystem *ps = (*psdb);

	double runtime_ms;
	if (step > 0) {
		clock_gettime(CLOCK_MONOTONIC, &stop);
		runtime_ms = (stop.tv_sec - start.tv_sec)*1000.0;
		runtime_ms += (stop.tv_nsec - start.tv_nsec)/1.0e6;
	}

	printf("step %u, best fitness: current %g, so far %g\n", step,
		ps->current_fitness, ps->global_fitness);
	if (step > 0) {
		int nsteps = step - last_step;
		/* MIPPS: Millions of Iterations*Particles per seconds */
		printf("runtime: %d iterations in %gms, %g iterations/s (%u particles, %gMIPPS)\n",
			nsteps, runtime_ms, nsteps*1000/runtime_ms, ps->num_particles,
			(nsteps*ps->num_particles)/runtime_ms/1000);
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

	//write_csv(ps, step);

	last_step = step;
	clock_gettime(CLOCK_MONOTONIC, &start);
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

/* Function to initialize a single particle at index i.
 * Returns fitness of initial position. */
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

/* Function to compute the new velocity of a given particle */
__global__ void new_vel(ParticleSystem *ps)
{
		int particleIndex = (blockIdx.x * blockDim.x + threadIdx.x);
		if(particleIndex >= ps->num_particles)
			return;

		int dim = threadIdx.y;
		Particle *p = &ps->particle[particleIndex];
		floatN *nvel = &p->vel;
		const float best_vec_rand_coeff = range_rand(0, 1, &p->prng_state);
		const float global_vec_rand_coeff = range_rand(0, 1, &p->prng_state);

		float pbest =  p->best_pos.dim[dim] - p->pos.dim[dim];
		float gbest = ps->global_best_pos.dim[dim] - p->pos.dim[dim];

		nvel->dim[dim] = vel_omega*nvel->dim[dim] + best_vec_rand_coeff*vel_phi_best*pbest +
				  global_vec_rand_coeff*vel_phi_global*gbest;

		if(nvel->dim[dim] > coord_range) nvel->dim[dim] = coord_range;
		else if (nvel->dim[dim] < -coord_range) nvel->dim[dim] = -coord_range;

}

__global__ void new_pos(ParticleSystem *ps)
{
	int particleIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if(particleIndex >= ps->num_particles)
		return;
	Particle *p = &ps->particle[particleIndex];
	int i;
	float dim_val, fit1 = 0, fit2 = 0;

	for(i = 0; i < p->pos.n; i++){
		p->pos.dim[i] += step_factor*p->vel.dim[i];
		if (p->pos.dim[i] > coord_max) p->pos.dim[i] = coord_max;
		else if (p->pos.dim[i] < coord_min) p->pos.dim[i] = coord_min;

		dim_val = p->pos.dim[i];
		fit1 += pow(dim_val - target_pos_shared.dim[i],2);
		fit2 += pow(dim_val,2);

	}

	// ### newpos
	p->fitness = fit1*(100*fit2+1)/10;
	if (p->fitness < p->best_fit) {
		p->best_fit = p->fitness;
		p->best_pos = p->pos;
	}
}

__global__ void find_min_fitness(ParticleSystem *ps){
	float fitness_min = HUGE_VALF;
	float fitness;
	int i, i_min = 0;
	for (i = 0; i < ps->num_particles; i++) {
		/* initialize the i-th particle */
		//Particle p = ps.particle[0];
		fitness = (ps->particle + i)->fitness;
		/* use its position as global best position if the fitness is
		 * less than the current global fitness */
		if (fitness < fitness_min) {
			i_min = i;
			fitness_min = fitness;
		}
	}
	ps->current_fitness = fitness_min;
	ps->current_best_pos = ps->particle[i_min].pos;

	if (fitness_min < ps->global_fitness) {
		ps->global_fitness = fitness_min;
		ps->global_best_pos = ps->current_best_pos;
	}
}

__global__ void find_min_fitness_parallel(ParticleSystem *ps, float* in, float* out, int* in_pos, int* out_pos, int n_in){
	extern __shared__ float sm[];
	int tid=threadIdx.x;
	uint i=blockIdx.x*blockDim.x+threadIdx.x;
	int stride;
	sm[tid] = HUGE_VALF;
	if(i >= ps->num_particles || i >= n_in)
			return;

	sm[tid] = in != NULL ? in[i] : (ps->particle + i)->fitness;

	//copy to SM
	for (stride = blockDim.x/2;stride>0;stride>>=1)
	{
		__syncthreads();
		if (tid<stride && sm[tid] > sm[tid+stride]){
			sm[tid] = sm[tid+stride];
		}
	}

	if (tid==0){
		out[blockIdx.x]=sm[0];//copy back
		out_pos[blockIdx.x] = i;
		printf("parallelo blocco %d =  %f\n", blockIdx.x,sm[0]);
	}
	//d[blockIdx.x] containts the sum of the block
//	__shared__ float smem_min[64];
//
//	int tid = threadIdx.x + blockIdx.x*els_per_block;
//
//	float min = HUGE_VALF;
//	float val;
//
//	const int iters = els_per_block/threads;
//
//	for(int i = 0; i < iters; i++)
//	{
//		if(tid + i*threads > ps->num_particles)
//			continue;
//		val = (ps->particle + tid + i*threads)->fitness;
//		min = val < min ? val : min;
//		printf("ci sono arrivato");
//	}
//
//
//	if(threads == 32)
//		smem_min[threadIdx.x+32] = 0.0f;
//
//	smem_min[threadIdx.x] = min;
//
//	__syncthreads();
//
//	if(threadIdx.x < 32)
//		warp_reduce_min(smem_min);
//
//	if(threadIdx.x == 0){
//		printf("parallelo ris best fit: %f",smem_min[threadIdx.x]);
////		ps->current_fitness = smem_min[threadIdx.x];
////		ps->current_best_pos = ps->particle[i_min].pos;
////
////		if (smem_min[threadIdx.x] < ps->global_fitness) {
////			ps->global_fitness = smem_min[threadIdx.x];
////			ps->global_best_pos = ps->current_best_pos;
////		}
//	}
}

__device__ void warp_reduce_min( float smem[64])
{

	smem[threadIdx.x] = smem[threadIdx.x+32] < smem[threadIdx.x] ?
						smem[threadIdx.x+32] : smem[threadIdx.x];

	smem[threadIdx.x] = smem[threadIdx.x+16] < smem[threadIdx.x] ?
						smem[threadIdx.x+16] : smem[threadIdx.x];

	smem[threadIdx.x] = smem[threadIdx.x+8] < smem[threadIdx.x] ?
						smem[threadIdx.x+8] : smem[threadIdx.x];

	smem[threadIdx.x] = smem[threadIdx.x+4] < smem[threadIdx.x] ?
						smem[threadIdx.x+4] : smem[threadIdx.x];

	smem[threadIdx.x] = smem[threadIdx.x+2] < smem[threadIdx.x] ?
						smem[threadIdx.x+2] : smem[threadIdx.x];

	smem[threadIdx.x] = smem[threadIdx.x+1] < smem[threadIdx.x] ?
						smem[threadIdx.x+1] : smem[threadIdx.x];

}
