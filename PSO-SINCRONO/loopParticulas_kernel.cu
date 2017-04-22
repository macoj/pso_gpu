/*
 * loop particulas
 */

#ifndef LOOPPARTICULAS
#define LOOPPARTICULAS

#include "Pso.h"
//#include <math.h>
#include <cufft.h>
#include <cutil.h>

__device__ float rosenbrock(float*, int, int);
__device__ float sphere (float*, int, int);
__device__ float rastrigin (float*, int, int);
__device__ float griewank (float*, int, int);
__device__ float schwefel1_2 (float*, int, int);
__device__ float p16 (float*, int, int);


// loop particulas deve ser executado com N threads, onde N é o número de agentes
__global__ void
loopParticulas(float* xx, float* atual, int dimensoes, int agentes, int funcao)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int solucaobase = tx + ty*blockDim.x;//bx*gridDim.x*blockDim.x*blockDim.y + by*blockDim.x*blockDim.y + ty*blockDim.x + tx;

	int solucao = solucaobase*dimensoes;

	if (solucaobase < agentes) {
		float minval = 0;
	//	minval = rosenbrock(xx, dimensoes, solucao);

		switch (funcao) {
			case 1 :
				minval = rosenbrock(xx, dimensoes, solucao);
				break;
			case 2 :
				minval = sphere(xx, dimensoes, solucao);
				break;
			case 3 :
				minval = rastrigin(xx, dimensoes, solucao);
				break;
			case 4 :
				minval = griewank(xx, dimensoes, solucao);
				break;
			case 5 :
				minval = schwefel1_2(xx, dimensoes, solucao);
				break;
			case 6 :
				//minval = p16(xx, dimensoes, solucao);
				break;
		}

		atual[solucaobase] = minval;
	}

}
__global__ void
calculaGbest (int* gbest, float* pbest)
{
	__shared__ float pbestdobloco[30];
	__shared__ int indice[30];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int particula = tx + ty*blockDim.x;

	if (particula < 30) {

		pbestdobloco[particula] = pbest[particula];
		indice[particula] = particula;

		__syncthreads();

		for (int s = 1; s <= 16; s = s*2) {
			if (particula % 2*s == 0)
				if (particula + s < 30)
						if (pbestdobloco[indice[particula]] > pbestdobloco[indice[particula+s]])
							indice[particula] = indice[particula + s];
			__syncthreads();
		}

		/*
		0  1  2  3  4  5  6  7  8  9  10  11  12  13  14  15
		1     |     |     |     |      |       |       |
		2           |           |              |
		4                       |
		8
		*/
		if (particula == 0) *gbest = indice[0];
	}

}

__global__ void
atualizaPbestx (float* atual, float* pbest, float* pbestx, float* xx, int iteracao)
{
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int index = bx*gridDim.x*blockDim.x*blockDim.y + by*blockDim.x*blockDim.y + ty*blockDim.x + tx;
	int agente = index / 30;
	__shared__ float pbestdobloco[30];
	__shared__ float atualdobloco[30];

	atualdobloco[agente] = atual[agente];
	pbestdobloco[agente] = pbest[agente];

	if (index < 900) {
		if (iteracao == 0) {
			pbestx[index] = xx[index];
			pbest[agente] = atual[agente];
		} else {
			if (atualdobloco[agente] < pbestdobloco[agente]) {
				pbestx[index] = xx[index];
			}

		}
	}
}

__global__ void
atualizaPbest (float* atual, float* pbest) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int agente = tx + ty*blockDim.x;
	if (agente < 30)
		if (atual[agente] < pbest[agente])
			pbest[agente] = atual[agente];
}


// OK
//UNROLLED
__device__ float rosenbrock(float* xx, int dimensao, int solucao) {
	int i;
	float result;

	result = 0.0;

	for (i = 0; i < dimensao - 1; i++)
		result += 100.0 * (xx[solucao + i + 1] - xx[solucao + i]*xx[solucao + i])
						* (xx[solucao + i + 1] - xx[solucao + i]*xx[solucao + i])
						+ (xx[solucao + i] - 1)*(xx[solucao + i] - 1);
	if (result < 0)
		result = -result;
	return result;
}



__device__ float rastrigin (float* xx, int dimensao, int solucao) {
	int i;
	float result;
	result = 0.0;
	for (i = 0; i < dimensao; i++)
		result += xx[solucao + i] * xx[solucao + i] - 10*__cosf(2*M_PI*xx[solucao + i]) + 10;
	return result;
}

__device__ float griewank (float* xx, int dimensao, int solucao) {
	int i;
	float sumaoquadrado;
	float produtorio;
	sumaoquadrado = 0.0;
	produtorio = 1.0;
	for (i = 0; i < dimensao; i++) {
		sumaoquadrado += xx[solucao + i] * xx[solucao + i];
		produtorio *= __cosf(xx[solucao + i]/sqrtf(i+1));
	}
	sumaoquadrado = sumaoquadrado/4000 - produtorio + 1;
	return sumaoquadrado;
}


// OK
__device__ float schwefel1_2 (float* xx, int dimensao, int solucao) {
	int i;

	float result, result2;
	result = 0.0;
	for (i = 0; i < dimensao; i++) {
		result2 = 0.0;
		for (int j = 0; j < i; j++)
			result2 += xx[solucao + j];
		result += result2*result2;
	}
	return result;
}


__device__ float sphere (float* xx, int dimensao, int solucao) {
	int i;
	float result;
	result = 0.0;
	for (i = 0; i < dimensao; i++)
		result += xx[solucao + i] * xx[solucao + i];
	return result;
}
#endif
