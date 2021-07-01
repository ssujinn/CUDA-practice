
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<stdio.h>
#include<stdlib.h>
#include <math.h>
#include <Windows.h>
#include <time.h>
#include <assert.h>

int BLOCK_SIZE_X = 16;
int BLOCK_SIZE_Y = 16;

#define CUDA_CALL(x) { const cudaError_t a = (x); if(a != cudaSuccess) { printf("\nCuda Error: %s (err_num=%d) at line:%d\n", cudaGetErrorString(a), a, __LINE__); cudaDeviceReset(); assert(0);}}
typedef float TIMER_T;

__int64 start, freq, end;
#define CHECK_TIME_START { QueryPerformanceFrequency((LARGE_INTEGER*)&freq); QueryPerformanceCounter((LARGE_INTEGER*)&start); }
#define CHECK_TIME_END(a) { QueryPerformanceCounter((LARGE_INTEGER*)&end); a = (float)((float)(end - start) / (freq / 1000.0f)); }


cudaEvent_t cuda_timer_start, cuda_timer_stop;
#define CUDA_STREAM_0 (0)

// CUDA event 객체를 사용하여 커널 실행시간 측정
void create_device_timer()
{
	CUDA_CALL(cudaEventCreate(&cuda_timer_start));
	CUDA_CALL(cudaEventCreate(&cuda_timer_stop));
}

void destroy_device_timer()
{
	CUDA_CALL(cudaEventDestroy(cuda_timer_start));
	CUDA_CALL(cudaEventDestroy(cuda_timer_stop));
}

inline void start_device_timer()
{
	cudaEventRecord(cuda_timer_start, CUDA_STREAM_0);
}

inline TIMER_T stop_device_timer()
{
	TIMER_T ms;
	cudaEventRecord(cuda_timer_stop, CUDA_STREAM_0);
	cudaEventSynchronize(cuda_timer_stop);

	cudaEventElapsedTime(&ms, cuda_timer_start, cuda_timer_stop);
	return ms;
}

#define CHECK_TIME_INIT_GPU() { create_device_timer(); }
#define CHECK_TIME_START_GPU() { start_device_timer(); }
#define CHECK_TIME_END_GPU(a) { a = stop_device_timer(); }
#define CHECK_TIME_DEST_GPU() { destroy_device_timer(); }



TIMER_T compute_time = 0;
TIMER_T device_time = 0;

typedef struct {
	int width;
	int height;
	float *elements;
} Array;


#define MAX_N_ELEMENTS	(1 << 20)
#define WIDTH 1024

void generate_random_float_array(float *array, int n) {
	int i;
	for (i = 0; i < n; i++) {
		array[i] = 3.1415926f*((float)rand() / RAND_MAX);
	}
}

void combine_two_arrays(float *A, float *B, float *C, int n) {
	int i;

	for (i = 0; i < n; i++) {
		C[i] = A[i] * A[i] + B[i] * B[i];
	}
}

__global__ void CombineTwoArrraysKernel(Array A, Array B, Array C) {

	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int id = gridDim.x * blockDim.x * row + col;
	//TODO

	C.elements[id] = A.elements[id] * A.elements[id] + B.elements[id] *B.elements[id];
		//A.elements

}

cudaError_t combine_two_arrays_GPU(const Array A, const Array B, Array C);


bool check_equal(Array C, Array G, int n) {

	for (int i = 0; i < n; i++) {
		if (fabs(C.elements[i] - G.elements[i]) > 0.00001) {
			printf("%d different C : %.6f, G : %.6f, diff : %.6f\n", i, C.elements[i], G.elements[i], C.elements[i] - G.elements[i]);
			return false;
		}
	}
	return true;
}

int main()
{

	int n_elements;

	srand(0);
	n_elements = MAX_N_ELEMENTS;

	printf("*** Data size : %d\n\n", n_elements);

	Array A, B, C, G;
	A.width = B.width = C.width = G.width = WIDTH;
	A.height = B.height = C.height = G.height = MAX_N_ELEMENTS / WIDTH;

	A.elements = (float *)malloc(sizeof(float)*MAX_N_ELEMENTS);
	B.elements = (float *)malloc(sizeof(float)*MAX_N_ELEMENTS);
	C.elements = (float *)malloc(sizeof(float)*MAX_N_ELEMENTS);
	G.elements = (float *)malloc(sizeof(float)*MAX_N_ELEMENTS);
	generate_random_float_array(A.elements, MAX_N_ELEMENTS);
	generate_random_float_array(B.elements, MAX_N_ELEMENTS);

	CHECK_TIME_START;
	combine_two_arrays(A.elements, B.elements, C.elements, n_elements);
	CHECK_TIME_END(compute_time);

	printf("***CPU Time taken = %.6fms\n", compute_time);

	cudaError_t cudaStatus = combine_two_arrays_GPU(A, B, G);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "combine_two_arrays_GPU failed!");
		return 1;
	}
	//printf("\nblock size : %d*%d\n", BLOCK_SIZE_X, BLOCK_SIZE_Y);
	printf("***GPU Time taken = %.6fms\n", device_time);

	bool check = check_equal(C, G, n_elements);
	if (check)
		printf("CPU and GPU calculate same\n");
	else
		printf("CPU and GPU calculate difference\n");

	printf("\nCPU [10] : %f, GPU [10] : %f\n", C.elements[10], G.elements[10]);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}
cudaError_t combine_two_arrays_GPU(const Array A, const Array B, Array C) {

	//아래 함수들을 사용하여 어떻게 하면 가급적 정확한 시간을 측정할 수 있을지 생각해볼 것.
	CHECK_TIME_INIT_GPU();
	CHECK_TIME_START_GPU();
	CHECK_TIME_END_GPU(device_time);
	CHECK_TIME_DEST_GPU();

	cudaError_t cudaStatus;
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}/////////////  if(cu.....  ==CUDA_CALL

	cudaDeviceProp deviceProp;
	CUDA_CALL(cudaGetDeviceProperties(&deviceProp, 0));

	Array d_A, d_B, d_C;
	size_t size;

	d_A.width = A.width; d_A.height = A.height;
	size = A.width * A.height * sizeof(float);

	//GPU 메모리 할당, 데이터를 CPU 메모리에서 GPU 메모리로 전송
	CUDA_CALL(cudaMalloc(&d_A.elements, size));
	CUDA_CALL(cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice));

	d_B.width = B.width; d_B.height = B.height;
	size = B.width * B.height * sizeof(float);
	CUDA_CALL(cudaMalloc(&d_B.elements, size));
	CUDA_CALL(cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice));

	d_C.width = C.width; d_C.height = C.height;
	size = C.width * C.height * sizeof(float);
	CUDA_CALL(cudaMalloc(&d_C.elements, size));

	//block size, grid size(블록 수) 를 선언
	dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	dim3 dimGrid(A.width / dimBlock.x, A.height / dimBlock.y);

	CHECK_TIME_INIT_GPU();
	CHECK_TIME_START_GPU();
	//커널 호출(실행)
	CombineTwoArrraysKernel << < dimGrid, dimBlock >> > (d_A, d_B, d_C);
	CHECK_TIME_END_GPU(device_time);
	CHECK_TIME_DEST_GPU();

	//결과 데이터를 GPU 메모리에서 CPU 메모리로 이동
	CUDA_CALL(cudaDeviceSynchronize());
	CUDA_CALL(cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost));


Error:
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
	return cudaStatus;
}

