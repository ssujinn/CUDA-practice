#include "kernel.h"
#define TX 16
#define TY 16


#define CUDA_CALL(x) { const cudaError_t a = (x); if(a != cudaSuccess) { printf("\nCuda Error: %s (err_num=%d) at line:%d\n", cudaGetErrorString(a), a, __LINE__); cudaDeviceReset(); assert(0);}}
typedef float TIMER_T;

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
	//intf("%f\n", ms);
	return ms;
}

#define CHECK_TIME_INIT_GPU() { create_device_timer(); }
#define CHECK_TIME_START_GPU() { start_device_timer(); }
#define CHECK_TIME_END_GPU(a) { a = stop_device_timer(); }
#define CHECK_TIME_DEST_GPU() { destroy_device_timer(); }


__global__ void distanceKernel(uchar4 *d_out, int w, int h, int2 pos)
{
	const int c = blockIdx.x * blockDim.x + threadIdx.x;	//블록의 id * 블록의 수 + 블록 내에서의 thread id
	const int r = blockIdx.y * blockDim.y + threadIdx.y;	//블록의 id * 블록의 수 + 블록 내에서의 thread id

	//원하는 범위 밖의 데이터는 계산하지 않는다.
	if ((c >= w) || (r >= h)) return;
	const int i = r * w + c;	//전체 thread 에서의 순서(id)
	
	int tmp = ((c - pos.x)*(c - pos.x) + (r - pos.y)*(r - pos.y)) / 100;
	if (tmp > 255)
		tmp = 255;
	if (tmp < 0)
		tmp = 0;

	d_out[i].x = 255 - tmp;		//R
	d_out[i].y = 255 - tmp;		//G
	d_out[i].z = 0;		//B
	d_out[i].w = 255;		//A (불투명)	
}

//커널을 호출하는 CPU 함수. 
float kernelLauncher(uchar4 *d_out, int w, int h, int2 pos) {
	float time=0;
	//블록의 크기. 가로가 TX개, 세로가 TY개
	const dim3 blockSize(TX, TY);

	//grid, 즉 thread block의 수. 블록이 grid.x * grid.y 개 존재.
	const dim3 gridSize = dim3((w + TX - 1) / TX, (h + TY - 1) / TY);

	CHECK_TIME_INIT_GPU();
	CHECK_TIME_START_GPU();
	distanceKernel << <gridSize, blockSize >> > (d_out, w, h, pos);
	CHECK_TIME_END_GPU(time);
	CHECK_TIME_DEST_GPU();
	return time;
}