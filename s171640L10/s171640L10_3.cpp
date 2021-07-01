#include <stdio.h>
#include <random>
#include <time.h>
#include <math.h>
#include <Windows.h>

#define N 25
double Taylor_series(double x, int n);
double Taylor_series_horner(double x, int n);
float Taylor_series_horner_float(float x, int n);
double Taylor_series_improve(double x, int n);

__int64 start, freq, end;
#define CHECK_TIME_START QueryPerformanceFrequency((LARGE_INTEGER*)&freq); QueryPerformanceCounter((LARGE_INTEGER*)&start)
#define CHECK_TIME_END(a) QueryPerformanceCounter((LARGE_INTEGER*)&end); a = (float)((float)(end - start) / (freq / 1000.0f))
float compute_time;

void main(void)
{
	double res;
	float resf;

	CHECK_TIME_START;
	res = Taylor_series(-8.3, N);
	CHECK_TIME_END(compute_time);
	printf("*** normal  f<-8.3> = %.6e , time : %lf\n", res, compute_time);

	CHECK_TIME_START;
	res = Taylor_series_horner(-8.3, N);
	CHECK_TIME_END(compute_time);
	printf("*** horner  f<-8.3> = %.6e , time : %lf\n", res, compute_time);

	CHECK_TIME_START;
	resf = Taylor_series_horner_float(-8.3, N);
	CHECK_TIME_END(compute_time);
	printf("*** float   f<-8.3> = %.6e , time : %lf\n", resf, compute_time);

	CHECK_TIME_START;
	res = Taylor_series_improve(-8.3, N);
	CHECK_TIME_END(compute_time);
	printf("*** improve f<-8.3> = %.6e , time : %lf\n", res, compute_time);

	CHECK_TIME_START;
	res = pow(2.71828182846, -8.3);
	CHECK_TIME_END(compute_time);
	printf("*** correct f<-8.3> = %.6e , time : %lf\n", res, compute_time);

}


double Taylor_series(double x, int n)
{
	double answer = 0.0f;
	int i;
	double fac = 1;

	for (i = 1; i < n; i++) {
		fac /= i;
		answer += (pow(x, i) * fac);
	}

	answer += 1;

	return answer;
}

double Taylor_series_horner(double x, int n)
{
	double answer = 0.0f;
	int i;
	double fac = 1;

	for (i = 1; i < n; i++)
		fac /= i;

	answer = fac;
	for (i = n - 1; i >= 1; i--) {
		fac *= i;
		answer = answer * x + fac;
	}

	answer = answer - fac + 1;

	return answer;
}

float Taylor_series_horner_float(float x, int n)
{
	float answer = 0.0f;
	int i;
	float fac = 1;

	for (i = 1; i < n; i++)
		fac /= i;

	answer = fac;
	for (i = n - 1; i >= 1; i--) {
		fac *= i;
		answer = answer * x + fac;
	}

	answer = answer - fac + 1;

	return answer;
}

double Taylor_series_improve(double x, int n)
{
	double answer = 0.0f;
	int i;
	double fac = 1;
	n *= 2;
	for (i = 1; i < n; i++)
		fac /= i;

	answer = fac;
	for (i = n - 1; i >= 1; i--) {
		fac *= i;
		answer = answer * x + fac;
	}

	answer = answer - fac + 1;

	return answer;
}