/*
 * loop particulas
 */

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


__global__ void
loopParticulas(float* xx, float* vx, float* pbestx, float* pbest, int* gbest, float MAXV, float MAXX, float MINX, int dimensoes, int agentes, int iteracao, int funcao)
{
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	__shared__ float minval[30];

	int index = bx*gridDim.x*blockDim.x*blockDim.y + by*blockDim.x*blockDim.y + ty*blockDim.x + tx;

	index = index - 2*(bx + by*gridDim.x);	// 2 threads perdidas por bloco
	//index = index - 12*(bx + by*gridDim.x); 	// 12 threads perdidas por bloco
	//	index = index - 4*(bx + by*3); 		// 4 threads perdidas por bloco
	//index = index - 8*(bx + by*gridDim.x); 		// 8 threads perdidas por bloco

	int id = tx + ty*blockDim.x;

	if (id >= 510) index = agentes*dimensoes;	// máximo 510/30 = 17 particulas por bloco
	//if (id >= 240) index = agentes*dimensoes;	// máximo 240/30 = 8 particulas por bloco
	//if (id >= 60) index = agentes*dimensoes;	// máximo 60/30 = 2 particulas por bloco
	//if (id >= 120) index = agentes*dimensoes;	// máximo 120/30 = 4 particulas por bloco

	//if ((id >=180) && (bx == 1) && (by == 0)) index = agentes*dimensoes;

	if (index < agentes*dimensoes) {

		int solucao = index / dimensoes;
		int pbestsolucao = solucao;

		solucao = solucao * dimensoes;

		if (index == solucao)		// apenas uma thread da particula irá executar a função
			//minval[pbestsolucao] = rosenbrock(xx, index, dimensoes, agentes, solucao);
			minval[pbestsolucao] =  schwefel1_2 (xx, dimensoes, solucao);
	//		minval[pbestsolucao] = sphere(xx, dimensoes, solucao);
			//minval[pbestsolucao] = p16(xx, dimensoes, solucao);

		/*
			switch (funcao) {
	 			case 1 :
					minval[pbestsolucao] = rosenbrock(xx, dimensoes, solucao);
					break;
				case 2 :
					minval[pbestsolucao] = sphere(xx, dimensoes, solucao);
					break;
				case 3 :
					minval[pbestsolucao] = rastrigin(xx, dimensoes, solucao);
					break;
				case 4 :
					minval[pbestsolucao] = griewank(xx, dimensoes, solucao);
					break;
				case 5 :
					minval[pbestsolucao] = schwefel1_2(xx, dimensoes, solucao);
					break;
				case 6 :
					minval[pbestsolucao] = p16(xx, dimensoes, solucao);
					break;
			}
*/
//		__syncthreads();

		if (iteracao == 0) {
			if (index == solucao)
				pbest[pbestsolucao] = minval[pbestsolucao];
			pbestx[index] = xx[index];
		} else if (minval[pbestsolucao] <= pbest[pbestsolucao]) {
			if (index == solucao)
				pbest[pbestsolucao] = minval[pbestsolucao];
			pbestx[index] = xx[index];
			if (index == solucao)
				if (pbest[pbestsolucao] < pbest[*gbest])	// RACE CONDITION?
					*gbest = pbestsolucao;
		}

		__syncthreads();
		int dimensao = index % dimensoes;
                float factor = 0.7298437881283576;
                float c = 2.05;
                // ATUALIZA VELOCIDADE
                vx[index] = factor*(vx[index] + c * ((float) numeroRandom(index)/(float) INT_MAX)*(pbestx[index]-xx[index]) +
                                                c * ((float) numeroRandom(index+1)/ (float) INT_MAX) * (pbestx[dimensao + gbest[0]*dimensoes] - xx[index]));
                // LIMITA VELOCIDADE
                if (vx[index] > MAXV)
                        vx[index] = MAXV;
                else if (vx[index] < -MAXV)
                        vx[index] = -MAXV;

                // ATUALIZA POSICAO
               xx[index] = xx[index] + vx[index];

                // LIMITA POSICAO
                if (xx[index] > MAXX) {
                        xx[index] = MAXX;
                        vx[index] = -vx[index];
                }
                if (xx[index] < MINX) {
                        xx[index] = MINX;
                        vx[index] = -vx[index];
                }

	}
}

/****************************************************
 * FUNCOES ******************************************
 ****************************************************/

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


// OK
// UNROLLED
__device__ float rastrigin (float* xx, int dimensao, int solucao) {
	int i;
	float result;
	result = 0.0;
	for (i = 0; i < dimensao; i++)
		result += xx[solucao + i] * xx[solucao + i] - 10*__cosf(2*M_PI*xx[solucao + i]) + 10;
	return result;
}

// OK
// UNROLLED
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

// OK
// UNROLLED
__device__ float sphere (float* xx, int dimensao, int solucao) {
	int i;
	float result;
	result = 0.0;

	for (i = 0; i < dimensao; i++)
		result += xx[solucao + i] * xx[solucao + i];

	return result;
}

