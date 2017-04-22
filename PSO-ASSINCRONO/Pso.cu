/*
 *
 * PSO-CUDA - it's like bolt, usain
 *
 */

// includes, system
#include <stdio.h>

// includes, project
#include <cutil_inline.h>

// includes, kernels

#include <inicializaParticulas_kernel.cu>
#include <loopParticulas_kernel.cu>
//#include <atualizaParticulas_kernel.cu>

void rodarPSO(int, int, int, int, int, float, float, float, float);
void print(float*, int, int);

int main(int argc, char** argv) {
	// $pso numero-particulas numero-dimensoes funcao-id iteracoes numero-de-vezes
	if (argc != 10) {
		printf("use: $pso numero-de-particulas numero-de-dimensoes funcao-id iteracoes numero-de-vezes IRang_L IRang_R MINX MAXX\nfuncao-id:  1: rosenbrock 2: sphere 3: rastrigin 4: griewank 5: schwefel1_2 6: p16\n");
		exit(1);
	}
	int particulas = atoi(argv[1]);
	int dimensoes = atoi(argv[2]);
	int funcao = atoi(argv[3]);
	int iteracoes = atoi(argv[4]);
	int run_no = atoi(argv[5]);
	float IRang_L = atof(argv[6]);
	float IRang_R = atof(argv[7]);
	float MINX = atof(argv[8]);
	float MAXX = atof(argv[9]);
	rodarPSO(particulas, dimensoes, funcao, iteracoes, run_no,IRang_L, IRang_R, MINX, MAXX);
}

void rodarPSO(int NUMBER_OF_AGENTS, int DIMENSION, int funcao, int MAXITER, int run_no, float IRang_L, float IRang_R, float MINX, float MAXX) {

	float MAXV = 0.5*(MAXX - MINX);

	cudaSetDevice(cutGetMaxGflopsDeviceId());

	unsigned int memoriaMatrizes = (DIMENSION * NUMBER_OF_AGENTS) * sizeof(float);
	unsigned int memoriaMatriz = (NUMBER_OF_AGENTS) * sizeof(float);
	float* vx;
	float* xx;
	float* tx;
	float* pbestx;
	float* pbest;
	//float* maxx;
	int* gbest;

	// alocando memoria com cudaMalloc:
	cutilSafeCall(cudaMalloc((void**) &gbest, sizeof(int)));
	cutilSafeCall(cudaMalloc((void**) &vx, memoriaMatrizes));
	cutilSafeCall(cudaMalloc((void**) &xx, memoriaMatrizes));
	cutilSafeCall(cudaMalloc((void**) &tx, memoriaMatrizes));
	cutilSafeCall(cudaMalloc((void**) &pbestx, memoriaMatrizes));
	cutilSafeCall(cudaMalloc((void**) &pbest, memoriaMatriz));
	//cutilSafeCall(cudaMalloc((void**) &maxx, memoriaMatriz));
	// setando parametros do CUDA

//	NAO-P16:
	dim3 threads(32, 16); // 512 threads (maximum)
	dim3 grid(1, 2); // 3 x 3


	for (int itera=0; itera < run_no; itera++) {

	        // criando timer para cada run:
        	unsigned int timer = 0;
	        cutilCheckError(cutCreateTimer(&timer));
	        cutilCheckError(cutStartTimer(timer));

		// inicializando as particulas
		inicializaParticulas<<<grid, threads >>>(xx, vx, pbestx, gbest, DIMENSION, NUMBER_OF_AGENTS, IRang_L, IRang_R, MAXV);

		int iter = 0;
		do {
			loopParticulas<<<grid, threads >>>(xx, vx, pbestx, pbest, gbest, MAXV, MAXX, MINX, DIMENSION, NUMBER_OF_AGENTS, iter, funcao);

			//debug
			//pbest
			float* pbestHost = (float*) malloc(memoriaMatriz);
			cutilSafeCall(cudaMemcpy(pbestHost, pbest, memoriaMatriz, cudaMemcpyDeviceToHost));

			int* gbestHost = (int*) malloc(sizeof (int));
			cutilSafeCall(cudaMemcpy(gbestHost, gbest, sizeof(int), cudaMemcpyDeviceToHost));
			printf("#%d# %.20f\n", iter+1, pbestHost[*gbestHost]);
			free(pbestHost);
			free(gbestHost);
			iter++;
		} while (iter < MAXITER);
	}

	// check if kernel execution generated and error
	cutilCheckMsg("Kernel execution failed");
	cutilSafeCall(cudaFree(xx));
	cutilSafeCall(cudaFree(vx));
	cutilSafeCall(cudaFree(tx));
	cutilSafeCall(cudaFree(pbestx));
	cutilSafeCall(cudaFree(pbest));
	cutilSafeCall(cudaFree(gbest));
	//cutilSafeCall(cudaFree(maxx));
	cudaThreadExit();
}

