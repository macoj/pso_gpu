/*
 * inicializa particulas
 */


#ifndef _INICIAPARTICULAS_KERNEL_H_
#define _INICIAPARTICULAS_KERNEL_H_

#include "Pso.h"
#include "NumeroRandom.cu"

__global__ void
inicializaParticulas(float* xx, float* vx, float* pbestx, int* gbest, int dimensoes, int agentes, float IRang_L, float IRang_R, float MAXV)
{
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int index = bx*gridDim.x*blockDim.x*blockDim.y + by*blockDim.x*blockDim.y + ty*blockDim.x + tx;

	if (index < dimensoes*agentes) {
		xx[index] = (float) ((IRang_R - IRang_L) * ((float) numeroRandom(index) / (float) INT_MAX) + (float) IRang_L);
//		xx[index] = (float) (IRang_L + ((float) numeroRandom(index) / (float) INT_MAX)*
		pbestx[index] = xx[index];

		float rnd = ((float) numeroRandom(index+1) / ((float) INT_MAX));
		vx[index] = (-MAXV + rnd*(MAXV - (-MAXV)));

//		if (rnd > 0.5)
//			vx[index] = -vx[index];

//		vx[index] = 25;

		if (index == 0) *gbest = 0;

	}
}

#endif
