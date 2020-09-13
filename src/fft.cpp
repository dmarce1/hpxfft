#include <hpx/hpx_init.hpp>
#include <complex>
#include <iostream>
#include <valarray>

#include <vector>
#include <complex>
#include <cmath>
#include <hpxfft/fft.hpp>

long long fft(std::vector<std::complex<float> > &X) {
	// Length variables
	const auto N = X.size();
	std::vector<float> A(N);
	std::vector<float> B(N);
	for (int i = 0; i < N; i++) {
		A[i] = X[i].real();
		B[i] = X[i].imag();
	}
	int level = 0;
	long long flop = 0;
	for (auto i = N; i > 1; i >>= 1) {
		level++;
	}
	if ((1 << level) != N) {
		return 0;
	}

	std::vector<float> cosi(N / 2);
	std::vector<float> sine(N / 2);
	for (int i = 0; i < N / 2; i++) {
		const float omega = -2.0 * M_PI * i / N;
		cosi[i] = std::cos(omega);
		sine[i] = std::sin(omega);
		flop += 38;
	}

	for (auto i = 0; i < N; i++) {
		auto j = 0;
		int l = i;
		for (int k = 0; k < level; k++) {
			j = (j << 1) | (l & 1);
			l >>= 1;
		}
		if (j > i) {
			std::swap(A[i], A[j]);
			std::swap(B[i], B[j]);
		}
	}

	for (int P = 2; P <= N; P *= 2) {
		const int s = N / P;
		for (int i = 0; i < N; i += P) {
			int k = 0;
			for (int j = i; j < i + P / 2; j++) {
				const auto treal = A[j + P / 2] * cosi[k] - B[j + P / 2] * sine[k];
				const auto timag = A[j + P / 2] * sine[k] + B[j + P / 2] * cosi[k];
				A[j + P / 2] = A[j] - treal;
				B[j + P / 2] = B[j] - timag;
				A[j] += treal;
				B[j] += timag;
				k += s;
				flop += 10;
			}
		}
	}
	for (int i = 0; i < N; i++) {
		reinterpret_cast<float (&)[2]>(X[i])[0] = A[i];
		reinterpret_cast<float (&)[2]>(X[i])[1] = B[i];
	}
	return flop;
}

#include <hpxfft/fft3d.hpp>

int hpx_main(int argc, char *argv[]) {
	const int N = 8;
	array3d<std::complex<real>> sub(0, 0, 0, N, N, N);
	for (int i = 0; i < N ; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N; k++) {
				sub(i, j, k) = std::complex < real > (k, k);
			}
		}
	}
	fft3d test(N);
	test.zero().get();
	test.inc_subarray(sub);
//	test.transpose_yz().get();
	test.transpose_xz().get();
	test.to_silo("X").get();
	return hpx::finalize();
}

int main(int argc, char *argv[]) {

	std::vector < std::string > cfg = { "hpx.commandline.allow_unknown=1" };

	hpx::init(argc, argv, cfg);
}

