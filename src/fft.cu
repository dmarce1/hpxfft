#include <stdio.h>

#include <hpxfft/cuda_check.hpp>
#include <vector>
#include <complex>

__global__ void fft_kernel_step1(float *cosi, float *sine, int N) {
	for (int i = threadIdx.x; i < N / 2; i += blockDim.x) {
		if (i < N / 2) {
			const float omega = -2.0 * M_PI * i / N;
			cosi[i] = std::cos(omega);
			sine[i] = std::sin(omega);
		}
	}

}

__global__ void fft_kernel_step2(float *Aptr, float *Bptr, float *cosi, float *sine, int N) {
	int level = 0;
	for (auto i = N; i > 1; i >>= 1) {
		level++;
	}
	if ((1 << level) != N) {
		if (threadIdx.x == 0) {
			printf("FFT requires power of two!\n");
		}
	}

	auto* A = Aptr + blockIdx.x * N;
	auto* B = Bptr + blockIdx.x * N;

	for (auto i = threadIdx.x; i < N; i += blockDim.x) {
		auto j = 0;
		int l = i;
		for (int k = 0; k < level; k++) {
			j = (j << 1) | (l & 1);
			l >>= 1;
		}
		if (j > i) {
			float tmp = A[i];
			A[i] = A[j];
			A[j] = tmp;
			tmp = B[i];
			B[i] = B[j];
			B[j] = tmp;
		}
	}

	for (int P = 2; P <= N; P *= 2) {
		const int s = N / P;
		if (N / P <= blockDim.x) {
			const int imax = ((N - 1) / blockDim.x + 1) * blockDim.x;
			for (int i = threadIdx.x * P; i < imax; i += blockDim.x * P) {
				if (i < N) {
					int k = 0;
					for (int j = i; j < i + P / 2; j++) {
						const auto treal = A[j + P / 2] * cosi[k] - B[j + P / 2] * sine[k];
						const auto timag = A[j + P / 2] * sine[k] + B[j + P / 2] * cosi[k];
						A[j + P / 2] = A[j] - treal;
						B[j + P / 2] = B[j] - timag;
						A[j] += treal;
						B[j] += timag;
						k += s;
					}
				}
				__syncthreads();
			}
		} else {
			for (int i = 0; i < N; i += P) {
				int k = threadIdx.x * s;
				const int jmax = ((P / 2 - 1) / blockDim.x + 1) * blockDim.x + i;
				for (int j = i + threadIdx.x; j < jmax; j += blockDim.x) {
					if (j < i + P / 2) {
						const auto treal = A[j + P / 2] * cosi[k] - B[j + P / 2] * sine[k];
						const auto timag = A[j + P / 2] * sine[k] + B[j + P / 2] * cosi[k];
						A[j + P / 2] = A[j] - treal;
						B[j + P / 2] = B[j] - timag;
						A[j] += treal;
						B[j] += timag;
						k += blockDim.x * s;
					}
					__syncthreads();
				}
			}
		}
	}
}

void fft_cuda(std::vector<std::vector<std::complex<float>> > &X) {
	float *A;
	float *B;
	float *A_dev;
	float *B_dev;
	float *cosi_dev;
	float *sine_dev;
	const int cnt = X.size();
	const int N = X[0].size();
	const int size = N * sizeof(float);
	CUDA_CHECK(cudaMallocHost((void**) &A, cnt * size));
	CUDA_CHECK(cudaMallocHost((void**) &B, cnt * size));
	CUDA_CHECK(cudaMalloc((void**) &A_dev, cnt * size));
	CUDA_CHECK(cudaMalloc((void**) &B_dev, cnt * size));
	CUDA_CHECK(cudaMalloc((void**) &cosi_dev, size));
	CUDA_CHECK(cudaMalloc((void**) &sine_dev, size));
	for (int j = 0; j < cnt; j++) {
		for (int i = 0; i < N; i++) {
			A[i + j * N] = X[j][i].real();
			B[i + j * N] = X[j][i].imag();
		}
	}
	fft_kernel_step1<<<1,32>>>(cosi_dev,sine_dev,N);
	cudaMemcpy(A_dev, A, cnt * size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_dev, B, cnt * size, cudaMemcpyHostToDevice);
	fft_kernel_step2<<<cnt,32>>>(A_dev, B_dev, cosi_dev,sine_dev,N);
	CUDA_CHECK(cudaMemcpy(A, A_dev, cnt * size, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(B, B_dev, cnt * size, cudaMemcpyDeviceToHost));
	for (int j = 0; j < cnt; j++) {
		for (int i = 0; i < N; i++) {
			reinterpret_cast<float (&)[2]>(X[j][i])[0] = A[i + j * N];
			reinterpret_cast<float (&)[2]>(X[j][i])[1] = B[i + j * N];
		}
	}
	CUDA_CHECK(cudaFreeHost(A));
	CUDA_CHECK(cudaFreeHost(B));
	CUDA_CHECK(cudaFree(A_dev));
	CUDA_CHECK(cudaFree(B_dev));
	CUDA_CHECK(cudaFree(cosi_dev));
	CUDA_CHECK(cudaFree(sine_dev));
}

