/*
 * numero random
*/

 #ifndef _NUMERORANDOM_KERNEL_H_
#define _NUMERORANDOM_KERNEL_H_

// usa clock() como 'seed'

__device__ int numeroRandom(int id) {
	int numeroRandom;
	numeroRandom = clock() + clock()*id;
	numeroRandom = numeroRandom^(numeroRandom << 21);
//	numeroRandom == numeroRandom^(numeroRandom >> 30);
	numeroRandom = numeroRandom^(numeroRandom << 4);
	if (numeroRandom < 0)
		numeroRandom = -numeroRandom;

	return numeroRandom;
}


__device__ int numeroRandomNeg(int id) {
        int numeroRandom;
        numeroRandom = clock() + clock()*id;
        numeroRandom = numeroRandom^(numeroRandom << 21);
  //      numeroRandom = numeroRandom^(numeroRandom >> 35);
        numeroRandom = numeroRandom^(numeroRandom << 4);
        return numeroRandom;
}


#endif
