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
//typedef struct float2
//{
//	float dim[DIM];
//} float2;

/* Extents of the domain in each dimension */
static const float coord_min = -1;
static const float coord_max = 1;
static const float coord_range = 2; // coord_max - coord_min, but the expression is not const in C 8-/

/* A particle */
typedef struct Particle
{
	float *pos; /* particle position */
	float *vel; /* particle velocity */
	float *best_pos; /* best position so far */
	float best_fit; /* fitness value of the best position so far */
	float fitness; /* fitness value of the current position */
	uint64_t prng_state; /* state of the PRNG for this particle */
} Particle;

/* The whole particle system */
typedef struct ParticleSystem
{
	Particle *particle; /* array of particles */
	float *current_best_pos; /* best position in the whole system a this iteration */
	float *global_best_pos; /* best position in the whole system so far */
	int num_particles; /* number of particles */
	float global_fitness; /* fitness value for global_best_pos */
	float current_fitness; /* fitness value for global_best_pos */
} ParticleSystem;

/* The target position */
float *target_pos;

/* Target function to be minimized: this is the square
 * Euclidean distance from target_pos, “perturbed” by the distance
 * to the origin: this puts a local minimum at the origin,
 * which is good to test if the method actually finds the global
 * minimum or not */
float fitness(const float **pos)
{
	float dim_vect[DIM];
	int i;
	for(i = 0; i < DIM; i++){
		dim_vect[i] = pos[0][i] - target_pos[i];
	}

	/* Euclidean square distance to the target position:
	 * sum of the squares of the differences
	 * of corresponding coordinates */
	float fit1 = 0;

	for(i = 0; i < DIM; i++){
		fit1 = fit1 + (dim_vect[i]*dim_vect[i]);
	}
	/* Euclidean square distance to the origin:
	 * sum of the squares of the coordinates */
	float fit2 = 0;
	for(i = 0; i < DIM; i++){
		fit2 = fit2 + (pos[0][i]*pos[0][i]);
	}
	return fit1*(100*fit2+1)/10;
}

/* Function to initialize a single particle at index i.
 * Returns fitness of initial position. */
float init_particle(Particle *p, int i)
{
	uint64_t prng_state;
	init_rand(&prng_state, i);
	float *pos = (float*)malloc(sizeof(float)*DIM);
	float *vel = (float*)malloc(sizeof(float)*DIM);
	int j;
	/* we initialize each component of the position
	 * to a random number between coord_min and coord_max,
	 * each component of the velocity to a random number
	 * between -coord_range and coord_range.
	 * The initial position is also the current best_pos,
	 * for which we also compute the fitness */

	for (j = 0; j < DIM; j++){
		pos[j] = range_rand(coord_min, coord_max, &prng_state);
	}

	for (j = 0; j < DIM; j++){
		vel[j] = range_rand(-coord_range, coord_range, &prng_state);
	}
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
	const float *pos = p->pos;
	int i = 0;

	const float *vel = p->vel;
	float *nvel = (float*)malloc(sizeof(float)*DIM);

	// best pos - pos
	float *pbest = p->best_pos;
	// global best post - pos
	float *gbest = ps->global_best_pos;

	//FIXME: controllare se è ottimizzabile accorpandolo al for sotto
	for(i = 0; i<DIM;i++){
		pbest[i] -= pos[i];
		gbest[i] -= pos[i];
	}

	for(i = 0; i<DIM;i++){
		nvel[i] = vel_omega*vel[i] + best_vec_rand_coeff*vel_phi_best*pbest[i] +
				  global_vec_rand_coeff*vel_phi_global*gbest[i];

		if(nvel[i] > coord_range) nvel[i] = coord_range;
		else if (nvel[i] < -coord_range) nvel[i] = -coord_range;
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
	const float *vel = p->vel;
	float *pos = p->pos;
	int i;

	for(i = 0; i < DIM; i++){
		pos[i] += step_factor*vel[i];
		if (pos[i] > coord_max) pos[i] = coord_max;
		else if (pos[i] < coord_min) pos[i] = coord_min;
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
float init_system(ParticleSystem *ps, int n)
{
	ps->num_particles = n;
	/* Allocate array of particles */
	ps->particle = malloc(n*sizeof(Particle));
	/* Compute the current fitness and best position */
	int i_min = 0;
	float fitness_min = HUGE_VALF;
	for (int i = 0; i < n; ++i) {
		/* initialize the i-th particle */
		float fitness = init_particle(ps->particle + i, i);
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
	target_pos = (float*)malloc(DIM*sizeof(float));

#define MAX_STEPS (1<<20) /* run for no more than 1Mi steps */
#define TARGET_FITNESS (FLT_EPSILON) /* or until the fitness is less than this much */
#define STEP_CHECK_FREQ 100 /* after how many steps to write the system and check the time */

	/* Initialize the target position */
	uint64_t prng_state;
	init_rand(&prng_state, time(NULL));
//	printf("target position: (");
//	for(j = 0; j < DIM; j++){
//		target_pos[j] = range_rand(coord_min, coord_max, &prng_state);
//		printf("%g,", target_pos[j]);
//	}
//
//	printf(")\n");

	// -0.287776,0.520416
	target_pos[0] = -0.287776;
	target_pos[1] = 0.520416;

	/* Initialize a system with the number of particles given
	 * on the command-line, defaulting to something which is
	 * related to the number of dimensions */
	init_system(&ps,
		argc > 1 ? atoi(argv[1]) : 128);

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
	int i,j;
	fname[FNSZ] = '\0';
	snprintf(fname, FNSZ, "pso.%u.csv", step);

	FILE *fp = fopen(fname, "w");
	if (!fp) {
		fprintf(stderr, "failed to open %s - %d: %s\n", fname, errno, strerror(errno));
		exit(1);
	}

	/* Write header */
	fprintf(fp, "x,y,vx,vy,current_fitness,bx,by,best_fit\n");

	/* Write out each particle */
	float *vec;
	for (i = 0; i < ps->num_particles; ++i)
	{
		const Particle *p = ps->particle + i;

		vec = p->pos;
		for(j = 0; j < DIM; j++){
			fprintf(fp, "%g,", vec[j]);
		}

		vec = p->vel;
		for(j = 0; j < DIM; j++){
			fprintf(fp, "%g,", vec[j]);
		}

		fprintf(fp, "%g,", p->fitness);

		vec = p->best_pos;
		for(j = 0; j < DIM; j++){
			fprintf(fp, "%g,", vec[j]);
		}

		fprintf(fp, "%g\n", p->best_fit);
	}


	/* Write out the target point */
	vec = target_pos;
	for(j = 0; j < DIM; j++){
		fprintf(fp, "%g,", vec[j]);
		vec[j] = 0;
	}

	for(j = 0; j < DIM; j++){
		fprintf(fp, "%g,", vec[j]);
		vec[j] = 0;
	}
	fprintf(fp, "%g,", 0.0); /* null fitness */
	vec = target_pos;
	for(j = 0; j < DIM; j++){
		fprintf(fp, "%g,", vec[j]); /* best pos is the same as target pos */
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
	float *vec;
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
	vec = target_pos;
	printf("\ttarget ");
	for(j = 0; j < DIM; j++){
		printf("%g,", vec[j]);

	}
	vec = ps->current_best_pos;
	printf("\n\tcurrent ");
	for(j = 0; j < DIM; j++){
		printf( "%g,", vec[j]);
	}
	vec = ps->global_best_pos;

	printf("\n\tso far ");
	for(j = 0; j < DIM; j++){
		printf("%g,", vec[j]);
	}
	printf("\n");

	write_csv(ps, step);

	last_step = step;
	clock_gettime(CLOCK_MONOTONIC, &start);
}
