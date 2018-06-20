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
#define DIM 2


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

/* A function that generates a random float in the given range */
float range_rand(float min, float max, uint64_t *prng_state)
{
	uint32_t r = MWC64X(prng_state);
	return min + r*((max - min)/UINT32_MAX);
}

/* A functio to initialize the PRNG */
void init_rand(uint64_t *prng_state, int i)
{
	*prng_state = i;
}

/* A vector in 2-dimensional space */
typedef struct floatN
{
	int n;
	float dim[DIM];
} floatN;

/* Extents of the domain in each dimension */
static const float coord_min = -1;
static const float coord_max = 1;
static const float coord_range = 2; // coord_max - coord_min, but the expression is not const in C 8-/

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
	float global_fitness; /* fitness value for global_best_pos */
	float current_fitness; /* fitness value for global_best_pos */
} ParticleSystem;

/* The target position */
floatN target_pos;

/* Target function to be minimized: this is the square
 * Euclidean distance from target_pos, “perturbed” by the distance
 * to the origin: this puts a local minimum at the origin,
 * which is good to test if the method actually finds the global
 * minimum or not */
float fitness(const floatN *pos)
{

	float* dim_vect = (float*)malloc(sizeof(float)*pos->n);
	int i;
	float fit1 = 0,fit2 = 0;
	for(i = 0; i < pos->n; i++){
		dim_vect[i] = pos->dim[i] - target_pos.dim[i];
	}
	/* Euclidean square distance to the target position:
	 * sum of the squares of the differences
	 * of corresponding coordinates */

	for(i = 0; i < pos->n; i++){
		fit1 = fit1 + (dim_vect[i]*dim_vect[i]);
	}
	/* Euclidean square distance to the origin:
	 * sum of the squares of the coordinates */
	for(i = 0; i < pos->n; i++){
		fit2 = fit2 + (pos->dim[i]*pos->dim[i]);
	}
	return fit1*(100*fit2+1)/10;
}

/* Function to initialize a single particle at index i.
 * Returns fitness of initial position. */
float init_particle(Particle *p, int i, int nDim)
{
	uint64_t prng_state;
	init_rand(&prng_state, i);
	floatN pos,vel;
	int j;
	/* we initialize each component of the position
	 * to a random number between coord_min and coord_max,
	 * each component of the velocity to a random number
	 * between -coord_range and coord_range.
	 * The initial position is also the current best_pos,
	 * for which we also compute the fitness */
	pos.dim[0] = range_rand(coord_min, coord_max, &prng_state);
	pos.dim[1] = range_rand(coord_min, coord_max, &prng_state);

	for (j = 0; j < nDim; j++){
		vel.dim[j] = range_rand(-coord_range, coord_range, &prng_state);
	}
	pos.n = vel.n = nDim;
	p->pos = pos;
	p->vel = vel;
	p->best_pos = pos;
	p->best_fit = p->fitness = fitness(&pos);
	p->prng_state = prng_state;
	return p->fitness;

}

/* Overall weight for the old velocity, best position distance and global
 * best position distance in the computation of the new velocity
 */
const float vel_omega = .9;
const float vel_phi_best = 2;
const float vel_phi_global = 2;
/* Function to compute the new velocity of a given particle */
void new_vel(Particle *p, const ParticleSystem *ps)
{
	/* Each velocity component gets updated by a randomly
	 * weighted factor given by the distance from the best
	 * particle position and a randomly weighted factor given
	 * by the distance from the global best position.
	 */
	uint64_t prng_state = p->prng_state;

	const float best_vec_rand_coeff = range_rand(0, 1, &prng_state);
	const float global_vec_rand_coeff = range_rand(0, 1, &prng_state);
	const floatN pos = p->pos;
	int i = 0;

	// best pos - pos
	floatN pbest = p->best_pos;

	// global best post - pos
	floatN gbest = ps->global_best_pos;

	for(i = 0; i<p->pos.n;i++){
		pbest.dim[i] -=  pos.dim[i];
		gbest.dim[i] -= pos.dim[i];
	}

	const floatN vel = p->vel;

	floatN nvel;
	nvel.n = p->pos.n;

	for(i = 0; i < p->pos.n;i++){
		nvel.dim[i] = vel_omega*vel.dim[i] + best_vec_rand_coeff*vel_phi_best*pbest.dim[i] +
				  global_vec_rand_coeff*vel_phi_global*gbest.dim[i];

		if(nvel.dim[i] > coord_range) nvel.dim[i] = coord_range;
		else if (nvel.dim[i] < -coord_range) nvel.dim[i] = -coord_range;
	}

	p->vel = nvel;
	p->prng_state = prng_state;
}

/* The contribution of the velocity to the new position. Set to 1
 * to use the standard PSO approach of adding the whole velocity
 * to the position.
 * In my experience a smaller factor speeds up convergence
 */
const float step_factor = 1;

/* Function to update the position (and possibly best position) of a given particle.
 * Returns the fitness of the new position. */
float new_pos(Particle *p)
{
	/* Update the position, clamping at the domain boundaries */
	const floatN vel = p->vel;
	floatN pos = p->pos;
	int i;

	for(i = 0; i < pos.n; i++){
		pos.dim[i] += step_factor*vel.dim[i];
		if (pos.dim[i] > coord_max) pos.dim[i] = coord_max;
		else if (pos.dim[i] < coord_min) pos.dim[i] = coord_min;
	}

	p->pos = pos;

	/* Compute the new fitness */
	const float fit = p->fitness = fitness(&pos);
	if (fit < p->best_fit) {
		p->best_fit = p->fitness;
		p->best_pos = pos;
	}
	return fit;
}

/* Function to initialize a particle system with n particles.
 * Returns the best global fitness. */
float init_system(ParticleSystem *ps, int n, int nDim)
{
	ps->num_particles = n;
	/* Allocate array of particles */
	ps->particle = malloc(n*sizeof(Particle));
	/* Compute the current fitness and best position */
	int i_min = 0;
	float fitness_min = HUGE_VALF;
	for (int i = 0; i < n; ++i) {
		/* initialize the i-th particle */
		float fitness = init_particle(ps->particle + i, i,nDim);
		/* use its position as global best position if the fitness is
		 * less than the current global fitness */
		if (fitness < fitness_min) {
			i_min = i;
			fitness_min = fitness;
		}
	}

	ps->global_fitness = ps->current_fitness = fitness_min;
	ps->global_best_pos = ps->current_best_pos = ps->particle[i_min].pos;

	return fitness_min;
}

/* Function to step the particle system.
 * Returns the current global fitness. */
float step_system(ParticleSystem *ps)
{
	/* Compute the new velocity for each particle */
	for (int i = 0; i < ps->num_particles; ++i) {
		new_vel(ps->particle + i, ps);
	}
	/* Update the position of each particle, and the global fitness */
	int i_min = 0;
	float fitness_min = HUGE_VALF;
	for (int i = 0; i < ps->num_particles; ++i) {
		/* initialize the i-th particle */
		float fitness = new_pos(ps->particle + i);
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
	return fitness_min;
}

/* Save the particle system to disk */
void write_system(const ParticleSystem* ps, int step);

int main(int argc, char *argv[])
{
	ParticleSystem ps;
	unsigned step = 0;

	int j;
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
	int dimensions = DIM;
		/* Initialize the target position */
		uint64_t prng_state;
		init_rand(&prng_state, time(NULL));
	//	target_pos.x = range_rand(coord_min, coord_max, &prng_state);
	//	target_pos.y = range_rand(coord_min, coord_max, &prng_state);
		target_pos.n = dimensions;

		target_pos.n = dimensions;
		target_pos.dim[0] = -0.287776;
		target_pos.dim[1] = 0.520416;


	/* Initialize a system with the number of particles given
	 * on the command-line, defaulting to something which is
	 * related to the number of dimensions */
	init_system(&ps,
		argc > 1 ? atoi(argv[1]) : 128, dimensions);

	write_system(&ps, step);

	while (step < MAX_STEPS) {
		++step;
		float fit = step_system(&ps);
		if (fit < TARGET_FITNESS)
			break;
		if (step % STEP_CHECK_FREQ == 0) {
			write_system(&ps, step);
		}
	}
	write_system(&ps, step);
}

void write_csv(const ParticleSystem *ps, int step)
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

void write_system(const ParticleSystem *ps, int step)
{
	/* Step at which we wrote last time */
	static int last_step = 0;
	/* structs used for timing */
	static struct timespec start;
	static struct timespec stop;
	int j;

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
