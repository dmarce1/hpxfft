/*
 * fft3d.hpp
 *
 *  Created on: Sep 12, 2020
 *      Author: dmarce1
 */

#ifndef HPXFFT_FFT3D_HPP_
#define HPXFFT_FFT3D_HPP_

#include <hpx/include/components.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <unordered_map>
#include <complex>
#include <vector>

using real = double;

namespace hpx {
namespace serialization {
template<class A>
inline void serialize(A &arc, std::complex<real> &z, const unsigned) {
	arc & reinterpret_cast<real (&)[2]>(z)[0];
	arc & reinterpret_cast<real (&)[2]>(z)[1];
}
}
}

using mutex_type = hpx::lcos::local::spinlock;

template<class T>
class array3d {
	int xmin;
	int ymin;
	int zmin;
	int xspan;
	int yspan;
	int zspan;
	int yzspan;
	int xmax;
	int ymax;
	int zmax;
	int size;
	std::vector<T> X;
	inline int index(int xi, int yi, int zi) const {
		if (xi < xmin) {
			printf("X %i below range %i\n", xi, xmin);
			abort();
		}
		if (xi >= xmax) {
			printf("X %i above range %i\n", xi, xmax);
			abort();
		}
		if (yi < ymin) {
			printf("Y %i below range %i\n", yi, ymin);
			abort();
		}
		if (yi >= ymax) {
			printf("Y %i above range %i\n", yi, ymax);
			abort();
		}
		if (zi < zmin) {
			printf("Z %i below range %i\n", zi, zmin);
			abort();
		}
		if (zi >= zmax) {
			printf("Z %i above range %i\n", zi, zmax);
			abort();
		}
		return (zi - zmin) + (yi - ymin) * zspan + (xi - xmin) * yzspan;
	}
public:
	template<class A>
	void serialize(A &&arc, unsigned) {
		arc & xmin;
		arc & ymin;
		arc & zmin;
		arc & xspan;
		arc & yspan;
		arc & zspan;
		arc & yzspan;
		arc & xmax;
		arc & ymax;
		arc & zmax;
		arc & size;
		arc & X;
	}
	T* data() {
		return X.data();
	}
	void resize(int xmin_, int ymin_, int zmin_, int xspan_, int yspan_, int zspan_) {
		xmin = xmin_;
		ymin = ymin_;
		zmin = zmin_;
		xspan = xspan_;
		yspan = yspan_;
		zspan = zspan_;
		yzspan = zspan * yspan;
		xmax = xmin + xspan;
		ymax = ymin + yspan;
		zmax = zmin + zspan;
		size = xspan * yspan * zspan;
		X.resize(size);
	}
	array3d() = default;
	array3d(int xmin_, int ymin_, int zmin_, int xspan_, int yspan_, int zspan_) {
		resize(xmin_, ymin_, zmin_, xspan_, yspan_, zspan_);
	}
	T operator()(int xi, int yi, int zi) const {
		return X[index(xi, yi, zi)];
	}
	T& operator()(int xi, int yi, int zi) {
		return X[index(xi, yi, zi)];
	}
	int get_xmin() const {
		return xmin;
	}
	int get_xmax() const {
		return xmax;
	}
	int get_ymin() const {
		return ymin;
	}
	int get_ymax() const {
		return ymax;
	}
	int get_zmin() const {
		return zmin;
	}
	int get_zmax() const {
		return zmax;
	}
	array3d<T> get_subarray(int xm, int ym, int zm, int xs, int ys, int zs) const {
		array3d sub;
		sub.resize(xm, ym, zm, xs, ys, zs);
		for (int i = xm; i < xm + xs; i++) {
			for (int j = ym; j < ym + ys; j++) {
				for (int k = zm; k < zm + zs; k++) {
					sub(i, j, k) = (*this)(i, j, k);
				}
			}
		}
	}
	void set_subarray(const array3d<T> &sub) {
		for (int i = sub.xmin; i < sub.xmax; i++) {
			for (int j = sub.ymin; j < sub.ymax; j++) {
				for (int k = sub.zmin; k < sub.zmax; k++) {
					(*this)(i, j, k) = sub(i, j, k);
				}
			}
		}
	}

};

class fft3d_block: public hpx::components::component_base<fft3d_block> {
	array3d<std::complex<real>> X;
	array3d<std::complex<real>> X0;
	int handle;
	int N;
	int M;
	int xi;
	int yi;
	int B;
	int P;
	std::vector<hpx::future<void>> futures;
	std::vector<hpx::promise<void>> promises;

	void do_fft();
public:
	fft3d_block(int N, int blockdim, int yi, int zi, int P, int handle);
	void step1();
	void step2();
	void step3();
	void to_silo(std::string);
	void send(array3d<std::complex<real>>);
	void send_inc(array3d<std::complex<real>>);
	void send_sync(array3d<std::complex<real>>, int, int);
	void zero();
	HPX_DEFINE_COMPONENT_ACTION(fft3d_block,step3);
	HPX_DEFINE_COMPONENT_ACTION(fft3d_block,send_inc);
	HPX_DEFINE_COMPONENT_ACTION(fft3d_block,zero);
	HPX_DEFINE_COMPONENT_ACTION(fft3d_block,step1);
	HPX_DEFINE_COMPONENT_ACTION(fft3d_block,step2);
	HPX_DEFINE_COMPONENT_ACTION(fft3d_block,send);
	HPX_DEFINE_COMPONENT_ACTION(fft3d_block,send_sync);
	HPX_DEFINE_COMPONENT_ACTION(fft3d_block,to_silo);
};

class fft3d_server;

struct fft3d_directory {
	std::vector<hpx::id_type> blocks;
public:
	template<class A>
	void serialize(A &&arc, unsigned) {
		arc & blocks;
	}
};

struct fft3d_params {
	int B;
	int P;
	int handle;
	template<class A>
	void serialize(A &&arc, unsigned) {
		arc & B;
		arc & P;
		arc & handle;
	}
};

class fft3d_server: public hpx::components::component_base<fft3d_server> {
	int handle;
	int N;
	int B;
	int P;
public:
	fft3d_server(int N, int oversubscription);
	~fft3d_server();

	fft3d_params get_params() const {
		fft3d_params p;
		p.handle = handle;
		p.P = P;
		p.B = B;
		return p;
	}
	HPX_DEFINE_COMPONENT_DIRECT_ACTION(fft3d_server,get_params);
};

class fft3d {
	hpx::id_type server;
	int handle;
	int N;
	int B;
	int P;
	hpx::future<void> step1();
	hpx::future<void> step2();
	hpx::future<void> step3();
public:
	fft3d(int N);
	hpx::future<void> fft();
	hpx::future<void> zero();
	hpx::future<void> to_silo(std::string);
	hpx::future<void> inc_subarray(const array3d<std::complex<real>>&);
	hpx::future<void> set_subarray(const array3d<std::complex<real>>&);
};

#endif /* HPXFFT_FFT3D_HPP_ */
