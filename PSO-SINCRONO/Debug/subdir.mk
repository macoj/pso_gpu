################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../Pso.cu 

CU_DEPS += \
./Pso.d 

OBJS += \
./Pso.o 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C Compiler'
	/opt/cuda/bin/nvcc --compiler-options -fno-strict-aliasing -I.  -I/home/macoj2/workspace/PSO-SINCRONO -I/opt/cuda/include -I/opt/cuda/common/inc -DUNIX -O3 -o "$@" -c "$<"
	@echo 'Finished building: $<'
	@echo ' '


