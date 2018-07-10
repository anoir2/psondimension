Particle Swarm Optimization
===================
A CUDA Implementation
-----------------------
### Anoir Boudoudouh
___

## Overview
Il seguente repository contiene l'implementazione dell'algoritmo Particle Swarm Optimization. Introdotta da James Kennedy e Russell Eberhart nel 1995 e sviluppata ampiamente nel 2001 come metodo per l’ottimizzazione di funzioni non lineari continue, si ispira alla simulazione di un modello sociale semplificato (lo stormo = flock, o meglio ancora lo sciame = swarm). La seguente implementazione prevede un sistema di N particelle con M dimensioni che si muoveranno con una velocità randomica. La funzione di fitness calcola la distanza euclidea tra le N particelle e il target finale e quando andrà sotto la soglia da noi impostata, si può dire che l'algoritmo sia concluso. Nel file RESULTS.md si trova l'output generato da nvprof.

## Implementazione
### main_v2.cu
Partendo da una versione base in C, si è provveduto ad implementare la prima versione CUDA-Based (**main_v2.cu**) creando i kernel **new_vel**, **new_pos** e **find_min_fitness_parallel**.
#### Strutture dati utilizzate
```c
/* n-dimensional space */
typedef struct floatN
{
	int n;
	float dim[DIM];
} floatN;

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
	int dim_particles; /* number of dimensions */
	float global_fitness; /* fitness value for global_best_pos */
	float current_fitness; /* fitness value for global_best_pos */
} ParticleSystem;
```
##### **new_vel**
Questo kernel si occupa di calcolare le nuove velocità delle varie particelle passando come parametro il riferimento alla struttura ParticleSystem che, a sua volta, contiene il riferimento al vettore particle di tipo Particle.
```c
...
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
...
```
##### **new_pos**
Questo kernel si occupa di calcolare le nuove posizioni delle varie particelle e i relativi fitness passando come parametro il riferimento alla struttura ParticleSystem che, a sua volta, contiene il riferimento al vettore particle di tipo Particle. L'aggiornamento delle dimensioni della singola particella vengono fatti all'interno di un ciclo for e viene utilizzata la funzione pow per l'elevazione a potenza.
```c
...
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

	p->fitness = fit1*(100*fit2+1)/10;
	if (p->fitness < p->best_fit) {
		p->best_fit = p->fitness;
		p->best_pos = p->pos;
	}
}
...
```
##### **find_min_fitness_parallel**
Questo kernel si occupa di trovare il fitness minimo calcolato nella iterazione corrente. L'idea di scorporare new_pos e questo kernel è dovuta ad un boost di prestazioni non indifferenti applicando una reduction iterativa, cosa che non sarebbe stata possibile (o molto più limitata) se i due kernel fossero stati lasciati come un unico solo. Al kernel verrà passato come parametro il riferimento alla struttura ParticleSystem che, a sua volta, contiene il riferimento al vettore particle di tipo Particle, il parametro ***in*** che servirà per le riduzioni successive alla prima (visto che per la prima riduzione, prenderemo i fitness direttamente dal puntatore ps->particle), il parametro ***out*** che conterrà i minimi locali per ogni blocco da dare in input alla prossima iterazione, il parametro ***offset*** utile a capire quale particella è stata elaborata, il parametro ***n_in*** che ci dirà quanti sono i valori nel vettore in e il parametro ***blocks*** che, arrivato a 1, ci dirà che siamo alla iterazione finale. 
```c
...
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
...
```
### main_v3.cu
I cambiamenti fatti in questa versione sono la trasformazione di tutte le variabili costanti in ***#DEFINE***, la modifica della funzione ***new_vel*** mettendo la gestione delle dimensioni di una singola particella all'interno di un ciclo for e aggiungendo la shared memory al kernel ***new_pos***. La trasformazione di tutte le costanti in ***#DEFINE*** ha rimosso i tempi di lettura da parte dei 3 kernel facendo si che ci fosse un boost generale delle prestazioni del 40%. 

##### **new_vel**
Spostando la gestione delle dimensioni, per ogni singola particella, dentro il kernel new_vel, ha diminuito i tempi di esecuzione del 30% rispetto al precedente.
```c
...
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
...
```
##### **new_pos**
La modifica apportata ha diminuito i tempi di esecuzione del 20% rispetto al precedente. L'utilizzo della shared memory non è, tuttavia, utilizzata in modo totalmente corretto e ottimale.
```c
...
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
...
```
### main_v4.cu
I cambiamenti fatti in questa versione sono l'unione dei kernel **new_vel** e **new_pos** per dare origine al kernel **new_vel_pos** che si occuperà di calcolare la nuova velocità e posizione di ogni particella. Il kernel utilizza la shared memory e, nel possibile, si cerca la coalescenza della memoria e di evitare i bank conflict. Inoltre vengono applicati degli accorgimenti come la direttiva ***#pragma unroll*** e il settare i parametri delle varie funzioni __restricted__ ove possibile. Le modifiche fatte hanno portato ad un boost di circa 20% rispetto alla versione precedente

##### **new_vel_pos**
Nell'unione dei due kernel, si è voluto iniziare ad usare al meglio i registri dei thread per diminuire gli accessi alla memoria globale e questo ha dato dei benefici (ad esempio, l'uso della variabile nvel ha portato un boost del 2% circa).
```c
...
__global__ void new_vel_pos(__restrict__ ParticleSystem *ps)
{
	extern __shared__ float smpos[];
	const int particleIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int particleIndexSHM = threadIdx.x;
	if(particleIndex >= ps->num_particles)
		return;
	Particle *p = &ps->particle[particleIndex];

	int i;
	float fit1 = 0, fit2 = 0;

	floatN *nvel = &p->vel;
	const float best_vec_rand_coeff = range_rand(0, 1, &p->prng_state);
	const float global_vec_rand_coeff = range_rand(0, 1, &p->prng_state);

	#pragma unroll
	for(i = 0; i < DIM; i++){
		smpos[particleIndexSHM] = p->pos.dim[i];
		float pbest =  p->best_pos.dim[i] - smpos[particleIndexSHM];
		float gbest = ps->global_best_pos.dim[i] - smpos[particleIndexSHM];

		nvel->dim[i] = vel_omega*nvel->dim[i] + best_vec_rand_coeff*vel_phi_best*pbest +
				  global_vec_rand_coeff*vel_phi_global*gbest;

		if(nvel->dim[i] > coord_range) nvel->dim[i] = coord_range;
		else if (nvel->dim[i] < -coord_range) nvel->dim[i] = -coord_range;

		smpos[particleIndexSHM] += (step_factor*nvel->dim[i]);
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
...
```
### main_v5.cu
I cambiamenti fatti in questa versione sono una rifattorizzazione totale della gestione delle particelle che consiste nell'eliminazione della struttura dati ParticleSystem e Particle creando dei vettori di tipo float che contenessero i valori per tutte le dimensioni, la rimozione dei cicli all'interno del kernel ***new_vel_pos***, il trasferimento del calcolo del fitness della particella è stato spostato da ***new_vel_pos*** ad un nuovo kernel di nome ***calc_fitness*** che si vede applicata una reduction dopo aver calcolato il fitness per ogni dimensione di ogni particella, tutti i kernel sono stati ottimizzati per l'uso dei registri dei thread e si è fatto un unroll nella reduction del kernel ***calc_fitness*** nella funzione ***warp_control_float2*** utilizzando l'istruzione ***#if*** a livello di precompilatore. Il boost ottenuto dalle seguenti modifiche si attesta all'incirca al 70% e si è raggiunto l'obiettivo della memory bandwith al 70%.

##### **new_vel_pos**
In questa nuova versione, il calcolo del fitness non è più presente e questo ha fatto si che tutte le ottimizzazioni applicabili per raggiungere la coalescenza di memoria fossero applicabili come un utilizzo performante della shared memory, l'uso dei registri in modo corretto e l'uso di istruzioni LOAD/STORE dalla global memory al minimo indispendabile.
```c
...
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
	prng_state[particleIndex] = prng_state_l;
}
...
```

##### **calc_fitness**
Il seguente kernel, come precedentemente detto, è nato dalla necessità di ottimizzare il calcolo delle velocità/posizioni delle particelle e il calcolo dei vari fitness. Questo kernel si occupa del calcolo del fitness di ogni componente di ogni particella e della loro somma tramite una reduction. Successivamente controlla se il best fitness locale è maggiore di quello appena calcolato e, se cosi fosse, procede a copiare il vettore delle posizioni attuale come quello migliore.
```c
...
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
		fitness_val[particleIndex] = fitness;
		if (fitness < best_fit[particleIndex]) {
			best_fit[particleIndex] = fitness;
			memcpy(best_pos + particleIndex,pos + particleIndex,sizeof(float)*DIM);
		}
	}
}
...
```
### main_v6.cu
I cambiamenti fatti in questa versione sono la parallelizzazione della funzione memcpy all'interno del kernel ***calc_fitness***. La modifica fatta ha portato un boost al kernel del 70%.

##### **calc_fitness**
Il seguente kernel, come precedentemente detto, è nato dalla necessità di ottimizzare il calcolo delle velocità/posizioni delle particelle e il calcolo dei vari fitness. Questo kernel si occupa del calcolo del fitness di ogni componente di ogni particella e della loro somma tramite una reduction. Successivamente controlla se il best fitness locale è maggiore di quello appena calcolato e, se cosi fosse, procede a copiare il vettore delle posizioni attuale come quello migliore.
```c
...
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

	warp_control_float2(smpos,particleIndexSHM, indexDIM);

	if (indexDIM==0){
		fitness_val[particleIndex] = smpos[particleIndexSHM].x*(100*smpos[particleIndexSHM].y+1)/10;
	}
	__syncthreads();
	if (fitness_val[particleIndex] < best_fit[particleIndex]) {
		best_fit[particleIndex] = fitness_val[particleIndex];
		memcpy(best_pos + particleIndexDIM,pos + particleIndexDIM,sizeof(float));
	}
}
...
```
