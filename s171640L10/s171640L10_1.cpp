#include <stdio.h>
#include <random>
#include <time.h>
#include <Windows.h>

__int64 start, freq, end;
#define CHECK_TIME_START QueryPerformanceFrequency((LARGE_INTEGER*)&freq); QueryPerformanceCounter((LARGE_INTEGER*)&start)
#define CHECK_TIME_END(a) QueryPerformanceCounter((LARGE_INTEGER*)&end); a = (float)((float)(end - start) / (freq / 1000.0f))
float compute_time;

#define MATDIM 1024
double *pMatA, *pMatB, *pMatC;
void MultiplySquareMatrices_1(double *pDestMatrix, double *pLeftMatrix, double *pRightMatrix, int MatSize);
void MultiplySquareMatrices_2(double *pDestMatrix, double *pLeftMatrix, double *pRightMatrix, int MatSize);
void MultiplySquareMatrices_3(double *pDestMatrix, double *pLeftMatrix, double *pRightMatrix, int MatSize);
void MultiplySquareMatrices_4(double *pDestMatrix, double *pLeftMatrix, double *pRightMatrix, int MatSize);

void init_MatMat(void);

int main()
{
	init_MatMat();

	CHECK_TIME_START;
	MultiplySquareMatrices_1(pMatC, pMatA, pMatB, MATDIM);
	CHECK_TIME_END(compute_time);
	printf("MultiplySquareMatrices_1 : %f ms\n", compute_time);

	CHECK_TIME_START;
	MultiplySquareMatrices_2(pMatC, pMatA, pMatB, MATDIM);
	CHECK_TIME_END(compute_time);
	printf("MultiplySquareMatrices_2 = %f ms\n", compute_time);

	CHECK_TIME_START;
	MultiplySquareMatrices_3(pMatC, pMatA, pMatB, MATDIM);
	CHECK_TIME_END(compute_time);
	printf("MultiplySquareMatrices_3 = %f ms\n", compute_time);

	CHECK_TIME_START;
	MultiplySquareMatrices_4(pMatC, pMatA, pMatB, MATDIM);
	CHECK_TIME_END(compute_time);
	printf("MultiplySquareMatrices_4 = %f ms\n", compute_time);
}


void MultiplySquareMatrices_1(double *pDestMatrix, double *pLeftMatrix, double *pRightMatrix, int MatSize)
{
	int i, j, k;

	memset(pDestMatrix, 0, sizeof(double) * MatSize * MatSize);

	for (i = 0; i < MatSize; i++) {
		for (j = 0; j < MatSize; j++) {
			for (k = 0; k < MatSize; k++) {
				pDestMatrix[i * MatSize + j] += pLeftMatrix[i * MatSize + k] * pRightMatrix[k * MatSize + j];
			}
		}
	}
}
void MultiplySquareMatrices_2(double *pDestMatrix, double *pLeftMatrix, double *pRightMatrix, int MatSize)
{
	int i, j, k;

	memset(pDestMatrix, 0, sizeof(double) * MatSize * MatSize);

	for (i = 0; i < MatSize; i++) {
		for (j = 0; j < MatSize; j++) {
			for (k = 0; k < MatSize; k++) {
				pDestMatrix[i * MatSize + k] += pLeftMatrix[i * MatSize + j] * pRightMatrix[j * MatSize + k];
			}
		}
	}
}
void MultiplySquareMatrices_3(double *pDestMatrix, double *pLeftMatrix, double *pRightMatrix, int MatSize)
{
	int i, j, k;

	memset(pDestMatrix, 0, sizeof(double) * MatSize * MatSize);

	for (i = 0; i < MatSize; i++) {
		for (j = 0; j < MatSize; j++) {
			for (k = 0; k < MatSize; k += 8) {
				pDestMatrix[i * MatSize + k] += pLeftMatrix[i * MatSize + j] * pRightMatrix[j * MatSize + k];
				pDestMatrix[i * MatSize + k + 1] += pLeftMatrix[i * MatSize + j] * pRightMatrix[j * MatSize + k + 1];
				pDestMatrix[i * MatSize + k + 2] += pLeftMatrix[i * MatSize + j] * pRightMatrix[j * MatSize + k + 2];
				pDestMatrix[i * MatSize + k + 3] += pLeftMatrix[i * MatSize + j] * pRightMatrix[j * MatSize + k + 3];
				pDestMatrix[i * MatSize + k + 4] += pLeftMatrix[i * MatSize + j] * pRightMatrix[j * MatSize + k + 4];
				pDestMatrix[i * MatSize + k + 5] += pLeftMatrix[i * MatSize + j] * pRightMatrix[j * MatSize + k + 5];
				pDestMatrix[i * MatSize + k + 6] += pLeftMatrix[i * MatSize + j] * pRightMatrix[j * MatSize + k + 6];
				pDestMatrix[i * MatSize + k + 7] += pLeftMatrix[i * MatSize + j] * pRightMatrix[j * MatSize + k + 7];
			}
		}
	}
}
void MultiplySquareMatrices_4(double *pDestMatrix, double *pLeftMatrix, double *pRightMatrix, int MatSize)
{
	int i, j, k;

	memset(pDestMatrix, 0, sizeof(double) * MatSize * MatSize);

	for (i = 0; i < MatSize; i++) {
		for (j = 0; j < MatSize; j++) {
			for (k = 0; k < MatSize; k += 8) {
				pDestMatrix[i * MatSize + k] += pLeftMatrix[i * MatSize + j] * pRightMatrix[j * MatSize + k];
				pDestMatrix[i * MatSize + k + 1] += pLeftMatrix[i * MatSize + j] * pRightMatrix[j * MatSize + k + 1];
				pDestMatrix[i * MatSize + k + 2] += pLeftMatrix[i * MatSize + j] * pRightMatrix[j * MatSize + k + 2];
				pDestMatrix[i * MatSize + k + 3] += pLeftMatrix[i * MatSize + j] * pRightMatrix[j * MatSize + k + 3];
				pDestMatrix[i * MatSize + k + 4] += pLeftMatrix[i * MatSize + j] * pRightMatrix[j * MatSize + k + 4];
				pDestMatrix[i * MatSize + k + 5] += pLeftMatrix[i * MatSize + j] * pRightMatrix[j * MatSize + k + 5];
				pDestMatrix[i * MatSize + k + 6] += pLeftMatrix[i * MatSize + j] * pRightMatrix[j * MatSize + k + 6];
				pDestMatrix[i * MatSize + k + 7] += pLeftMatrix[i * MatSize + j] * pRightMatrix[j * MatSize + k + 7];
			}
		}
	}
}


void init_MatMat(void)
{
	double *ptr;
	pMatA = (double *)malloc(sizeof(double)*MATDIM*MATDIM);
	pMatB = (double *)malloc(sizeof(double)*MATDIM*MATDIM);
	pMatC = (double *)malloc(sizeof(double)*MATDIM*MATDIM);
	srand((unsigned)time(NULL));
	ptr = pMatA;
	for (int i = 0; i < MATDIM*MATDIM; i++)
		*ptr++ = (double)rand() / ((double)RAND_MAX);
	ptr = pMatB;
	for (int i = 0; i < MATDIM*MATDIM; i++)
		*ptr++ = (double)rand() / ((double)RAND_MAX);
}
