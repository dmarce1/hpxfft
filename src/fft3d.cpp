/*
 * fft3d.cpp
 *
 *  Created on: Sep 12, 2020
 *      Author: dmarce1
 */

#include <hpxfft/fft3d.hpp>
#include <silo.h>

#include <hpx/include/run_as.hpp>

static mutex_type mtx;
static std::atomic<int> next_handle(0);
static std::unordered_map<int, std::shared_ptr<fft3d_directory>> directories;

static std::vector<hpx::id_type> localities;
static int myid;

HPX_REGISTER_COMPONENT(hpx::components::component<fft3d_block>, fft3d_block);
HPX_REGISTER_COMPONENT(hpx::components::component<fft3d_server>, fft3d_server);

static void add_directory(int handle, fft3d_directory dir);
static void remove_directory(int handle);

HPX_PLAIN_ACTION (add_directory);
HPX_PLAIN_ACTION (remove_directory);

static void init() {
	static mutex_type mtx;
	std::lock_guard<mutex_type> lock(mtx);
	if (localities.size() == 0) {
		localities = hpx::find_all_localities();
		myid = hpx::get_locality_id();
	}
}

fft3d_block::fft3d_block(int N_, int blocksize_, int xi_, int yi_, int P_, int h) {
	P = P_;
	N = N_;
	handle = h;
	B = blocksize_;
	xi = xi_;
	yi = yi_;
	M = P * B;
	X.resize(xi * B, yi * B, 0, B, B, M);
	X0.resize(xi * B, yi * B, 0, B, B, M);
	futures.resize(P * P);
	promises.resize(P * P);
	for (int i = 0; i < P * P; i++) {
		futures[i] = promises[i].get_future();
	}
}

void fft3d_block::transpose_yz() {
	const fft3d_directory *dir;
	{
		std::lock_guard<mutex_type> lock(mtx);
		dir = &(*directories[handle]);
	}
	std::vector<hpx::future<void>> futs;

	int me = xi * P + yi;
	//printf( "entering %i\n", me);
	int l = (((me + 1) << 1) + 0) - 1;
	int r = (((me + 1) << 1) + 1) - 1;
//	printf( "Children %i %i %i\n", l, r, P);
	if (l < P * P) {
		futs.push_back(hpx::async < transpose_yz_action > (dir->blocks[l]));
	}
	if (r < P * P) {
		futs.push_back(hpx::async < transpose_yz_action > (dir->blocks[r]));
	}
	std::vector<int> others;
	for (int k0 = 0; k0 < M; k0 += B) {
		array3d<std::complex<real>> sub(B * xi, k0, B * yi, B, B, B);
		for (int i = B * xi; i < B * (xi + 1); i++) {
			for (int j = B * yi; j < B * (yi + 1); j++) {
				for (int k = k0; k < k0 + B; k++) {
					sub(i, k, j) = X(i, j, k);
				}
			}
		}
		const int oxi = xi;
		const int oyi = k0 / B;
		const int oii = oxi * P + oyi;
		others.push_back(oii);
	//	printf("%i sending to %i\n", xi * P + yi, oxi * P + oyi);
		futs.push_back(hpx::async < send_sync_action > (dir->blocks[oii], std::move(sub), xi, yi));
	}

	hpx::wait_all(futs.begin(), futs.end());
	for (auto i : others) {
		futures[i].get();
		promises[i] = hpx::promise<void>();
		futures[i] = promises[i].get_future();
	}
	X = X0;
}


void fft3d_block::transpose_xz() {
	const fft3d_directory *dir;
	{
		std::lock_guard<mutex_type> lock(mtx);
		dir = &(*directories[handle]);
	}
	std::vector<hpx::future<void>> futs;

	int me = xi * P + yi;
	//printf( "entering %i\n", me);
	int l = (((me + 1) << 1) + 0) - 1;
	int r = (((me + 1) << 1) + 1) - 1;
//	printf( "Children %i %i %i\n", l, r, P);
	if (l < P * P) {
		futs.push_back(hpx::async < transpose_xz_action > (dir->blocks[l]));
	}
	if (r < P * P) {
		futs.push_back(hpx::async < transpose_xz_action > (dir->blocks[r]));
	}
	std::vector<int> others;
	for (int k0 = 0; k0 < M; k0 += B) {
		array3d<std::complex<real>> sub(k0, B * yi, B * xi, B, B, B);
		for (int i = B * xi; i < B * (xi + 1); i++) {
			for (int j = B * yi; j < B * (yi + 1); j++) {
				for (int k = k0; k < k0 + B; k++) {
					sub(k, j, i) = X(i, j, k);
				}
			}
		}
		const int oxi = k0 / B;
		const int oyi = yi;
		const int oii = oxi * P + oyi;
		others.push_back(oii);
	//	printf("%i sending to %i\n", xi * P + yi, oxi * P + oyi);
		futs.push_back(hpx::async < send_sync_action > (dir->blocks[oii], std::move(sub), xi, yi));
	}

	hpx::wait_all(futs.begin(), futs.end());
	for (auto i : others) {
		futures[i].get();
		promises[i] = hpx::promise<void>();
		futures[i] = promises[i].get_future();
	}
	X = X0;
}

void fft3d_block::zero() {
	const fft3d_directory *dir;
	{
		std::lock_guard<mutex_type> lock(mtx);
		dir = &(*directories[handle]);
	}
	std::vector<hpx::future<void>> futs;
	int me = xi * P + yi;
	int l = (((me + 1) << 1) + 0) - 1;
	int r = (((me + 1) << 1) + 1) - 1;
	if (l < P * P) {
		futs.push_back(hpx::async < zero_action > (dir->blocks[l]));
	}
	if (r < P * P) {
		futs.push_back(hpx::async < zero_action > (dir->blocks[r]));
	}
	for (int i = X.get_xmin(); i < X.get_xmax(); i++) {
		for (int j = X.get_ymin(); j < X.get_ymax(); j++) {
			for (int k = X.get_zmin(); k < X.get_zmax(); k++) {
				X(i, j, k) = std::complex < real > (0.0, 0.0);
			}
		}
	}
	hpx::wait_all(futs.begin(), futs.end());
}

void fft3d_block::to_silo(std::string basename) {
	const fft3d_directory *dir;
	{
		std::lock_guard<mutex_type> lock(mtx);
		dir = &(*directories[handle]);
	}
	std::vector<hpx::future<void>> futs;
	int me = xi * P + yi;
	std::string dirname = basename + ".data";
	if (me == 0) {
		std::string cm = std::string("mkdir -p ") + dirname;
		if (system(cm.c_str()) != 0) {
			printf("Fatal error unable to make %s\n", dirname.c_str());
			abort();
		}
	}
	int l = (((me + 1) << 1) + 0) - 1;
	int r = (((me + 1) << 1) + 1) - 1;
	if (l < P * P) {
		futs.push_back(hpx::async < to_silo_action > (dir->blocks[l], basename));
	}
	if (r < P * P) {
		futs.push_back(hpx::async < to_silo_action > (dir->blocks[r], basename));
	}

	hpx::threads::run_as_os_thread([&]() {
		static mutex_type mutex;
		std::lock_guard<mutex_type> lock(mutex);
		std::string filename = dirname + "/" + std::to_string(me) + ".silo";
		DBfile *db = DBCreateReal(filename.c_str(), DB_CLOBBER, DB_LOCAL, "FFT3D", DB_PDB);
		auto opts = DBMakeOptlist(1);
		int one = 1;
		DBAddOption(opts, DBOPT_HIDE_FROM_GUI, &one);
		const char *coordnames[] = { "x", "y", "z" };
		const int xmin = xi * B;
		const int ymin = yi * B;
		const int zmin = 0;
		const int xmax = std::min((xi + 1) * B, N);
		const int ymax = std::min((yi + 1) * B, N);
		const int zmax = N;
		std::vector<double> x(xmax - xmin + 1);
		std::vector<double> y(ymax - ymin + 1);
		std::vector<double> z(zmax - zmin + 1);
		for (int i = xmin; i <= xmax; i++) {
			x[i - xmin] = i - 0.5;
		}
		for (int i = ymin; i <= ymax; i++) {
			y[i - ymin] = i - 0.5;
		}
		for (int i = zmin; i <= zmax; i++) {
			z[i - zmin] = i - 0.5;
		}
		int dims1[] = { xmax - xmin + 1, ymax - ymin + 1, zmax - zmin + 1 };
		int dims2[] = { xmax - xmin, ymax - ymin, zmax - zmin };
		void *coords[] = { x.data(), y.data(), z.data() };
		DBPutQuadmesh(db, "mesh", coordnames, coords, dims1, 3, DB_DOUBLE, DB_COLLINEAR, opts);
		array3d<real> v(zmin, ymin, xmin, zmax - zmin, ymax - ymin, xmax - xmin);
		for (int i = xmin; i < xmax; i++) {
			for (int j = ymin; j < ymax; j++) {
				for (int k = zmin; k < zmax; k++) {
					v(k, j, i) = X(i, j, k).real();
				}
			}
		}
		DBPutQuadvar1(db, "r", "mesh", v.data(), dims2, 3, NULL, 0, DB_DOUBLE, DB_ZONECENT, opts);
		for (int i = xmin; i < xmax; i++) {
			for (int j = ymin; j < ymax; j++) {
				for (int k = zmin; k < zmax; k++) {
					v(k, j, i) = X(i, j, k).imag();
				}
			}
		}
		DBPutQuadvar1(db, "i", "mesh", v.data(), dims2, 3, NULL, 0, DB_DOUBLE, DB_ZONECENT, opts);
		DBFreeOptlist(opts);
		DBClose(db);
	}).get();
	hpx::wait_all(futs.begin(), futs.end());
	if (me == 0) {
		hpx::threads::run_as_os_thread([&]() {
			std::string filename = basename + ".silo";
			DBfile *db = DBCreateReal(filename.c_str(), DB_CLOBBER, DB_LOCAL, "FFT3D", DB_PDB);
			std::vector<int> meshtypes(P * P, DB_QUAD_RECT);
			std::vector<int> vartypes(P * P, DB_QUADVAR);
			std::vector<char*> meshnames(P * P);
			std::vector<char*> inames(P * P);
			std::vector<char*> rnames(P * P);
			for (int i = 0; i < P * P; i++) {
				std::string base = dirname + "/" + std::to_string(i) + ".silo:";
				std::string mname = base + "mesh";
				std::string rname = base + "r";
				std::string iname = base + "i";
				meshnames[i] = new char[mname.size() + 1];
				inames[i] = new char[iname.size() + 1];
				rnames[i] = new char[rname.size() + 1];
				std::strcpy(meshnames[i], mname.c_str());
				std::strcpy(rnames[i], rname.c_str());
				std::strcpy(inames[i], iname.c_str());
			}
			DBPutMultimesh(db, "mesh", P * P, meshnames.data(), meshtypes.data(), NULL);
			DBPutMultivar(db, "r", P * P, rnames.data(), vartypes.data(), NULL);
			DBPutMultivar(db, "i", P * P, inames.data(), vartypes.data(), NULL);
			for (int i = 0; i < P * P; i++) {
				delete[] meshnames[i];
				delete[] rnames[i];
				delete[] inames[i];
			}
			DBClose(db);
		}).get();
	}
}

void fft3d_block::send_sync(array3d<std::complex<real>> sub, int i, int j) {
	X0.set_subarray(sub);
	promises[i * P + j].set_value();
}

void fft3d_block::send(array3d<std::complex<real>> sub) {
	X.set_subarray(sub);
}

void fft3d_block::send_inc(array3d<std::complex<real>> sub) {
	for (int i = sub.get_xmin(); i < sub.get_xmax(); i++) {
		for (int j = sub.get_ymin(); j < sub.get_ymax(); j++) {
			for (int k = sub.get_zmin(); k < sub.get_zmax(); k++) {
				X(i, j, k) += sub(i, j, k);
			}
		}
	}
}

fft3d_server::fft3d_server(int N_, int oversubscription) {
	N = N_;
	handle = next_handle++ * localities.size() + myid;

	const int ncomp_max = oversubscription * localities.size();
	P = 1;
	while (P * P <= ncomp_max && P <= N) {
		P++;
	}
	P--;
//	printf("Creating %i x %i modules\n", P, P);
	B = ((N - 1) / P + 1);
	fft3d_directory dir;
	dir.blocks.resize(P * P);
	int k = 0;
	std::vector < hpx::future < hpx::id_type >> futs;
	futs.reserve(P * P);
	for (int i = 0; i < P; i++) {
		for (int j = 0; j < P; j++) {
			const int locid = k * localities.size() / (P * P);
			futs.push_back(hpx::new_ < fft3d_block > (localities[locid], N, B, i, j, P, handle));
			k++;
		}
	}
	for (int i = 0; i < P; i++) {
		for (int j = 0; j < P; j++) {
			dir.blocks[i * P + j] = futs[i * P + j].get();
		}
	}
	add_directory_action()(localities[0], handle, dir);
}

void add_directory(int handle, fft3d_directory dir) {
	init();
	std::vector<hpx::future<void>> futs;
	const int l = (((myid + 1) << 1) + 0) - 1;
	const int r = (((myid + 1) << 1) + 1) - 1;
	if (l < localities.size()) {
		futs.push_back(hpx::async < add_directory_action > (localities[l], handle, dir));
	}
	if (r < localities.size()) {
		futs.push_back(hpx::async < add_directory_action > (localities[r], handle, dir));
	}
	{
		std::lock_guard<mutex_type> lock(mtx);
		directories[handle] = std::make_shared < fft3d_directory > (std::move(dir));
	}
	hpx::wait_all(futs.begin(), futs.end());
}

void remove_directory(int handle) {
	std::vector<hpx::future<void>> futs;
	const int l = (((myid + 1) << 1) + 0) - 1;
	const int r = (((myid + 1) << 1) + 1) - 1;
	if (l < localities.size()) {
		futs.push_back(hpx::async < remove_directory_action > (localities[l], handle));
	}
	if (r < localities.size()) {
		futs.push_back(hpx::async < remove_directory_action > (localities[r], handle));
	}
	{
		std::lock_guard<mutex_type> lock(mtx);
		directories.erase(handle);
	}
	hpx::wait_all(futs.begin(), futs.end());
}

fft3d_server::~fft3d_server() {
	remove_directory_action()(localities[0], handle);
}

fft3d::fft3d(int N_) {
	init();
	N = N_;
	server = hpx::new_ < fft3d_server > (hpx::find_here(), N, 9).get();
	auto tmp = fft3d_server::get_params_action()(server);
	handle = tmp.handle;
	B = tmp.B;
	P = tmp.P;
}

hpx::future<void> fft3d::transpose_yz() {
	const fft3d_directory *dir;
	{
		std::lock_guard<mutex_type> lock(mtx);
		dir = &(*directories[handle]);
	}
	return hpx::async < fft3d_block::transpose_yz_action > (dir->blocks[0]);

}


hpx::future<void> fft3d::transpose_xz() {
	const fft3d_directory *dir;
	{
		std::lock_guard<mutex_type> lock(mtx);
		dir = &(*directories[handle]);
	}
	return hpx::async < fft3d_block::transpose_xz_action > (dir->blocks[0]);

}

hpx::future<void> fft3d::zero() {
	const fft3d_directory *dir;
	{
		std::lock_guard<mutex_type> lock(mtx);
		dir = &(*directories[handle]);
	}
	return hpx::async < fft3d_block::zero_action > (dir->blocks[0]);

}

hpx::future<void> fft3d::to_silo(std::string name) {
	const fft3d_directory *dir;
	{
		std::lock_guard<mutex_type> lock(mtx);
		dir = &(*directories[handle]);
	}
	return hpx::async < fft3d_block::to_silo_action > (dir->blocks[0], name);

}

hpx::future<void> fft3d::set_subarray(const array3d<std::complex<real>> &x) {
	const fft3d_directory *dir;
	{
		std::lock_guard<mutex_type> lock(mtx);
		dir = &(*directories[handle]);
	}
	const int Ib = x.get_xmin() / B;
	const int Ie = (x.get_xmax() - 1) / B + 1;
	const int Jb = x.get_ymin() / B;
	const int Je = (x.get_ymax() - 1) / B + 1;
	std::vector<hpx::future<void>> futs;
	for (int I = Ib; I < Ie; I++) {
		for (int J = Jb; J < Je; J++) {
			const int xmin = std::max(I * B, x.get_xmin());
			const int ymin = std::max(J * B, x.get_ymin());
			const int zmin = std::max(0, x.get_zmin());
			const int xmax = std::min((I + 1) * B, x.get_xmax());
			const int ymax = std::min((J + 1) * B, x.get_ymax());
			const int zmax = std::min(B * P, x.get_zmax());
			array3d<std::complex<real>> sub(xmin, ymin, zmin, xmax - xmin, ymax - ymin, zmax - zmin);
			for (int i = xmin; i < xmax; i++) {
				for (int j = ymin; j < ymax; j++) {
					for (int k = zmin; k < zmax; k++) {
						sub(i, j, k) = x(i, j, k);
					}
				}
			}
			futs.push_back(hpx::async < fft3d_block::send_action > (dir->blocks[I * P + J], std::move(sub)));
		}
	}
	return hpx::async([](std::vector<hpx::future<void>> &&futs) {
		hpx::wait_all(futs.begin(), futs.end());
	}, std::move(futs));
}

hpx::future<void> fft3d::inc_subarray(const array3d<std::complex<real>> &x) {
	const fft3d_directory *dir;
	{
		std::lock_guard<mutex_type> lock(mtx);
		dir = &(*directories[handle]);
	}
	const int Ib = x.get_xmin() / B;
	const int Ie = (x.get_xmax() - 1) / B + 1;
	const int Jb = x.get_ymin() / B;
	const int Je = (x.get_ymax() - 1) / B + 1;
	std::vector<hpx::future<void>> futs;
	for (int I = Ib; I < Ie; I++) {
		for (int J = Jb; J < Je; J++) {
			const int xmin = std::max(I * B, x.get_xmin());
			const int ymin = std::max(J * B, x.get_ymin());
			const int zmin = std::max(0, x.get_zmin());
			const int xmax = std::min((I + 1) * B, x.get_xmax());
			const int ymax = std::min((J + 1) * B, x.get_ymax());
			const int zmax = std::min(B * P, x.get_zmax());
			array3d<std::complex<real>> sub(xmin, ymin, zmin, xmax - xmin, ymax - ymin, zmax - zmin);
			for (int i = xmin; i < xmax; i++) {
				for (int j = ymin; j < ymax; j++) {
					for (int k = zmin; k < zmax; k++) {
						sub(i, j, k) = x(i, j, k);
					}
				}
			}
			futs.push_back(hpx::async < fft3d_block::send_inc_action > (dir->blocks[I * P + J], std::move(sub)));
		}
	}
	return hpx::async([](std::vector<hpx::future<void>> &&futs) {
		hpx::wait_all(futs.begin(), futs.end());
	}, std::move(futs));
}

