/*
	Running & Testing Commands:
	g++ -o PSO PSO.c rand.cpp
	./PSO 2 100 100 3 -3 1
	./PSO 2 100 100 3.5 2.3 0.9 1
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rand.h"

#define PI 3.141592654
#define UB 5
#define LB -5

typedef struct{
	double *x; // current position
	double *v; // current velocity vector
	double *y; // local best position
	double f;
}Particle;

Particle *Swarm, gBest;

int n, N, Gmax;
double c1, c2, *r1, *r2, w;

double f(double *x){
	double tmp = 0.0;
	int i;
	for(i = 0; i < n; i++)
		tmp += x[i]*x[i];
	return tmp;
}

void initialize(){
    for (int i = 0; i < N; ++i){
        for (int j = 0; j < n; ++j){
        	Swarm[i].x[j] = rndreal(LB, UB); 
        	Swarm[i].y[j] = Swarm[i].x[j];      
        }
        Swarm[i].f = INFINITY;
    }
    gBest.f = INFINITY;
}

void allocate_mem(){
	int i;
	Swarm = (Particle *)malloc(N*sizeof(Particle));
	for(i = 0; i < N; i++){
		Swarm[i].x = (double *)calloc(n, sizeof(double));
		Swarm[i].v = (double *)calloc(n, sizeof(double));
		Swarm[i].y = (double *)calloc(n, sizeof(double));
	}
	gBest.x = (double *)calloc(n, sizeof(double));
	gBest.v = (double *)calloc(n, sizeof(double));
	gBest.y = (double *)calloc(n, sizeof(double));

	r1 = (double *)calloc(n, sizeof(double));
	r2 = (double *)calloc(n, sizeof(double));
}

void gBestPSO(int run){
	FILE *file;
	FILE *fileMov;
	char filename[50];
	sprintf(filename, "conv_%d.dat", run);
	file = fopen(filename, "w");
	fileMov = fopen("answer.txt","w");
	if(file == NULL){
		printf("Error! file %s couldn't be opened!\n", filename);
		exit(-1);
	}
	initialize();
	int t = 0;
	while(t < Gmax){
		for(int i = 0; i < N; i++){
			double fx = f(Swarm[i].x);
			if(fx < Swarm[i].f){
				// Update local best position
				memcpy(Swarm[i].y, Swarm[i].x, n*sizeof(double));
				Swarm[i].f = fx;
			}

			if(Swarm[i].f < gBest.f){
				// Update gBest based on comparisons with all particles
				memcpy(gBest.x, Swarm[i].y, n*sizeof(double));
				gBest.f = Swarm[i].f;
			}
		}

		for(int r = 0; r < n; r++){
			r1[r] = rndreal(0, 1);
			r2[r] = rndreal(0, 1);
		}

		for(int i = 0; i < N; i++){
			// Set values for vectors r1 and r2 from a uniform distribution in range [0, 1]
			for(int j = 0; j < n; j++){
				Swarm[i].v[j] = w*Swarm[i].v[j] + c1*r1[j]*(Swarm[i].y[j] - Swarm[i].x[j]) + c2*r2[j]*(gBest.x[j] - Swarm[i].x[j]);
				Swarm[i].x[j] = Swarm[i].x[j] + Swarm[i].v[j];
				// If out of bounds, force x_j to be inside the feasible region
				if(Swarm[i].x[j] < LB)
					Swarm[i].x[j] = LB;
				else if(Swarm[i].x[j] > UB)
					Swarm[i].x[j] = UB;
				// Save in file position of particle
				fprintf(fileMov,"%lf " ,Swarm[i].x[j]);
				fprintf(file, "%lf ", Swarm[i].x[j]);	
			}
		}
		fprintf(file, "%n");
		fprintf(fileMov,"\n ");	
		t++;
	}
	fclose(file);
	fclose(fileMov);
}


int main(int argc, char *argv[]){
	int RUN_MAX;
	if(argc != 8){
		printf("Syntax error! %s dim numParticles Gmax c1 c2 w RUNs\n", argv[0]);
		exit(-1);
	}
	n = atoi(argv[1]);
	N = atoi(argv[2]);
	Gmax = atoi(argv[3]);
	c1 = atof(argv[4]);
	c2 = atof(argv[5]);
	w = atof(argv[6]);
	RUN_MAX = atoi(argv[7]);

	initrandom(time(0)); 
    allocate_mem();
    int run;
    for(run = 0; run < RUN_MAX; run++){
        gBestPSO(run);
        printf("%lf\n", gBest.f);
    }
}