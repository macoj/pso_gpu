/*
 * atualiza velocidades
 *
 */


#include "Pso.h"
#include "NumeroRandom.cu"

__global__ void
atualizaParticulas(float* xx, float* vx, float* pbestx, int* gbest, float MAXV, float MAXX, float MINX, int agentes, int dimensoes) {
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int index = bx*gridDim.x*blockDim.x*blockDim.y + by*blockDim.x*blockDim.y + ty*blockDim.x + tx;
	int dimensao = index % dimensoes;
	if (index < agentes*dimensoes) {
//		float factor = 0.72894;
		float factor = 0.7298437881283576;
		float c = 2.05;
		// ATUALIZA VELOCIDADE
	/*	vx[index] = factor*(vx[index] + c * ((float) numeroRandom(index)/(float) INT_MAX)*(pbestx[index]-xx[index]) +
  		                                c * ((float) numeroRandom(index+1)/ (float) INT_MAX) * (pbestx[dimensao*agentes + gbest[0]] - xx[index]));
*/
		vx[index] = factor*(vx[index] + c * ((float) numeroRandom(index)/(float) INT_MAX)*(pbestx[index]-xx[index]) +
						c * ((float) numeroRandom(index+1)/ (float) INT_MAX) * (pbestx[dimensao + gbest[0]*dimensoes] - xx[index]));
//		vx[index] = factor*(vx[index] + c * (0.5)*(pbestx[index]-xx[index]) + c * (0.5) * (pbestx[dimensao + gbest[0]*dimensoes] - xx[index]));

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
