#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<stdio.h>
#include<stdlib.h>
#include <math.h>
#include <Windows.h>
#include <time.h>
#include <assert.h>

#define CUDA_CALL(x) { const cudaError_t a = (x); if(a != cudaSuccess) { printf("\nCuda Error: %s (err_num=%d) at line:%d\n", cudaGetErrorString(a), a, __LINE__); cudaDeviceReset(); assert(0);}}
typedef float TIMER_T;

#define	IN
#define OUT
#define INOUT

__int64 start, freq, end;
#define CHECK_TIME_START { QueryPerformanceFrequency((LARGE_INTEGER*)&freq); QueryPerformanceCounter((LARGE_INTEGER*)&start); }
#define CHECK_TIME_END(a) { QueryPerformanceCounter((LARGE_INTEGER*)&end); a = (float)((float)(end - start) / (freq / 1000.0f)); }


cudaEvent_t cuda_timer_start, cuda_timer_stop;
#define CUDA_STREAM_0 (0)

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


#define N_SIZE (1 << 20)													// ��ü ������ ������
#define NF_SIZE (1 << 6)													// Nf ũ��

#define BLOCK_SIZE (1 << 6)													// CUDA Ŀ�� thread block ������

#define BLOCK_WIDTH (1 << 3)
#define BLOCK_HEIGHT (BLOCK_SIZE / BLOCK_WIDTH)

#define N_ITERATION (1 << 0)												// ���� �ݺ� Ƚ��

TIMER_T compute_time = 0;
TIMER_T device_time = 0;

int N;
int Nf;

int *h_ArrayElements;
int *h_SumOfArrayElements_CPU;
int *h_SumOfArrayElements_GPU;

cudaError_t Sum_n_elements_GPU(IN int *p_ArrayElements, OUT int *p_SumOfElements_GPU, int Nf);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//	
//	�迭�� index - Nf ���� index + Nf ������ ������ ���� ����ϴ� Ŀ�� �ڵ�
//	
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void Sum_n_elements_Kernel(IN int *d_ArrayElements, OUT int *d_SumOfArrayElements, int N, int Nf) {

	//TODO
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int k;

	int sum = 0;
	for (k = id - Nf; k <= id + Nf; k++) {
		if (k >= 0 && k < N)
			sum += d_ArrayElements[k];
	}
	d_SumOfArrayElements[id] = sum;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//	
//	�迭�� index - Nf ���� index + Nf ������ ������ ���� ����ϴ� C �ڵ�
//	GPU kernel�� ����� �񱳸� ���� ���� ����� �Ͽ����� �Ǵ��ϴ� �����ͷ� Ȱ��
//	
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Sum_n_elements_CPU(IN int *p_ArrayElements, OUT int *p_SumOfElements_CPU, int Nf) {

	//TODO
	int i, k;

	for (i = 0; i < N_SIZE; i++) {
		int sum = 0;
		for (k = i - Nf; k <= i + Nf; k++) {
			if (k >= 0 && k < N_SIZE)
				sum += p_ArrayElements[k];
		}
		p_SumOfElements_CPU[i] = sum;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//	
//	�־��� bin ������ �д� �ڵ�
//	ù 4����Ʈ�� ��ü �������� ����, ���� 4����Ʈ�� Nf�� ũ��, �� ���� N���� int�� �����Ͱ� ����
//	�����ʹ� -100 ~ 100 ������ ���� ���� ����
//	
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void read_bin_file() {
	printf("***Binary File Read Start!!\n");
	FILE *fp = fopen("gen.bin", "rb");
	fread(&N, sizeof(int), 1, fp);
	fread(&Nf, sizeof(int), 1, fp);

	h_ArrayElements = (int *)malloc(N * sizeof(int));
	h_SumOfArrayElements_CPU = (int *)malloc(N * sizeof(int));
	h_SumOfArrayElements_GPU = (int *)malloc(N * sizeof(int));

	fread(h_ArrayElements, sizeof(int), N, fp);

	fclose(fp);
	printf("***Binary File Read End!!\n\n");
}

void init_bin_file(IN int n, IN int nf) {
	printf("***Binary File Create Start!!\n");
	srand(0);
	FILE *fp = fopen("gen.bin", "wb");
	fwrite(&n, sizeof(int), 1, fp);
	fwrite(&nf, sizeof(int), 1, fp);

	int i, input;

	for (i = 0; i < n; i++) {
		input = (int)((float)rand() / RAND_MAX * 200 - 100);
		fwrite(&input, sizeof(int), 1, fp);
	}

	fclose(fp);
	printf("***Binary File Create End!!\n\n");
}

int main()
{
	int i;
	init_bin_file(N_SIZE, NF_SIZE);
	read_bin_file();

	TIMER_T CPU_time = 0.0f, GPU_time_NO_SHARED = 0.0f;

	for (i = 0; i < N_ITERATION; i++) {
		CHECK_TIME_START;
		Sum_n_elements_CPU(h_ArrayElements, h_SumOfArrayElements_CPU, Nf);
		CHECK_TIME_END(compute_time);
		CPU_time += compute_time;

		Sum_n_elements_GPU(h_ArrayElements, h_SumOfArrayElements_GPU, Nf);
		GPU_time_NO_SHARED += device_time;
	}

	for (i = 0; i < N; i++) {
		if (h_SumOfArrayElements_CPU[i] != h_SumOfArrayElements_GPU[i]) {
			printf("%d : CPU : %d,\tGPU : %d\n", i, h_SumOfArrayElements_CPU[i], h_SumOfArrayElements_GPU[i]);
			break;
		}
	}
	if (i == N)
		printf("***Kernel execution Success!!\n\n");

	printf("***CPU compute time : %.3f ms\n", CPU_time / N_ITERATION);
	printf("***GPU NO SHARED compute time : %.3f ms\n", GPU_time_NO_SHARED / N_ITERATION);

	printf("\n*** CPU[10] : %d GPU[10] : %d\n", h_SumOfArrayElements_CPU[10], h_SumOfArrayElements_GPU[10]);

	free(h_ArrayElements);
	free(h_SumOfArrayElements_CPU);
	free(h_SumOfArrayElements_GPU);

	return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//	
//	Ŀ���� �����ϱ� �� �ʿ��� �ڷ�� �غ� �� Ŀ���� ������ ����̽��� ����
//	
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

cudaError_t Sum_n_elements_GPU(IN int *p_ArrayElements, OUT int *p_SumOfElements_GPU, int Nf) {
	cudaError_t cudaStatus = cudaSetDevice(0);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaDeviceProp deviceProp;
	CUDA_CALL(cudaGetDeviceProperties(&deviceProp, 0));

	int *d_X, *d_S;
	size_t size;

	size = N * sizeof(int);

	//GPU �޸� �Ҵ�, �����͸� CPU �޸𸮿��� GPU �޸𸮷� ����
	CUDA_CALL(cudaMalloc(&d_X, size));
	CUDA_CALL(cudaMemcpy(d_X, p_ArrayElements, size, cudaMemcpyHostToDevice));

	CUDA_CALL(cudaMalloc(&d_S, size));
	//CUDA_CALL(cudaMemcpy(d_S, p_SumOfElements_GPU, size, cudaMemcpyHostToDevice));

	//block size, grid size(���� ��) �� ����
	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(N_SIZE / dimBlock.x);

	CHECK_TIME_INIT_GPU();
	CHECK_TIME_START_GPU();
	//Ŀ�� ȣ��(����)
	Sum_n_elements_Kernel << < dimGrid, dimBlock >> > (d_X, d_S, N_SIZE, Nf);
	CHECK_TIME_END_GPU(device_time);
	CHECK_TIME_DEST_GPU();

	//��� �����͸� GPU �޸𸮿��� CPU �޸𸮷� �̵�
	CUDA_CALL(cudaDeviceSynchronize());
	CUDA_CALL(cudaMemcpy(p_SumOfElements_GPU, d_S, size, cudaMemcpyDeviceToHost));


Error:
	cudaFree(d_S);
	cudaFree(d_X);

	return cudaStatus;
}