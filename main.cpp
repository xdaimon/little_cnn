#include <iostream>
using std::cout; using std::endl;

#include "common.h"
#include "mnist_loader.h"

using R = double; // real
using I = int;    // integer

template<I rows, I cols>
using M = Matrix<R, rows, cols>;
template<I sz>
using MD = DiagonalMatrix<R, sz>;
template<I rows, I cols>
using View = Map<Matrix<R, rows, cols>>;

// gradient check
#define GC

/*
The jacobian of fog(x) wrt x is J[fog](x) = J[f](g(x)) * J[g](x)
The component wise description of this matrix multiplication is
df(g(x))_i/dx_j = sum_k df(g(x))_i/dg(x)_k * dg(x)_k/dx_j
dfog(x)/dx    = a;
dfog(x)/dg(x) = b;
dg(x)/dx      = c;
a_ij = sum_k( b_ik * c_kj )
*/

class cnn {
	static constexpr I is = 28;      // image size
	static constexpr I fs = 5;       // convolution filter size, must be odd
	static constexpr I nf = 10;      // # of conv filters, # of feature maps
	static constexpr I pws = 2;      // pool window size, must be 2
	static constexpr I as = is-fs+1; // convolution activation size
	static constexpr I ps = as/pws;  // pooled activation size
	static constexpr I os = 10;      // output vector size

	template<typename T>
	static T f(const T& A) { return 1./(1.+(-A).array().exp()); }
	template<typename T>
	static T df(const T& A) { return A.array() * (1 - A.array()); }

	template<typename T>
	static R c(const T& A, const T& Y) {
		const auto& y = Y.array();
		const auto& a = A.array();
		return -((y*a.log() + (1 - y)*(1 - a).log()).sum());
	}
	template<typename T>
	static T dc(const T& A, const T& Y) { return A-Y; }

	public:
	static void run() {
		Data test, train, validation;
		load_data(train, test, validation);

		M<is,is>       x;

		// Conv layer variables
		M<fs,fs>       w[nf]; for (auto &X : w) X = M<fs, fs>::Random()/double(fs*fs);
		R              b[nf]; for (auto &X : b) X = rand() / double(RAND_MAX);
		M<as,as>       z;
		M<as,as>       a[nf];
		M<ps*ps,1>     m;

		M<os,ps*ps>    dfzdm;
		M<ps*ps,as*as> dmda[nf]; // ones and zeros, contains ps*ps ones
		MD<as*as>      dadz[nf];
		M<1,ps*ps>     dcdm;
		M<1,as*as>     dcdz[nf];

		M<as*as,fs*fs> dzdw;
		M<1,fs*fs>     dcdw[nf];
		R              dcdb[nf];

		// FC layer variables
		M<os,ps*ps>    fw = M<os, ps*ps>::Random() / double(ps*ps);
		M<os,1>        fb; fb.setZero();
		M<os,1>        fz;
		M<os,1>        fa;
		M<os,1>        y;

		M<1,os>        dcdfa;
		MD<os>         dfadfz;
		M<1,os>        dcdfz;

		M<os,os*ps*ps> dfzdfw;

		M<1,os*ps*ps>  dcdfw;
		M<1,os>        dcdfb;

		// w z a m fw fz fa c


		// TODO recompute all gradients on paper

		auto forward = [&]() {
			// Compute the convolution for each feature map
			for (I k = 0; k < nf; ++k) {
				for (I i = 0; i < as; ++i)
					for (I j = 0; j < as; ++j)
						z(i,j) = (w[k].array() * x.block<fs,fs>(i,j).array()).sum() + b[k];
				a[k] = f(z);
			}

			// Compute the pool and also the derivative of the pooling wrt its inputs
			m.setZero();
			for (I k = 0; k < nf; ++k) {
				dmda[k].setZero();
				for (I i = 0; i < ps; ++i) {
					I row,col;
					for (I j = 0; j < ps; ++j) {
						m(i*ps+j) += a[k].block<pws,pws>(pws*i,pws*j).maxCoeff(&row, &col);
						dmda[k](i*ps + j, (i*pws + row)*as + j * pws + col) = 1;
					}
				}
			}

			// Compute the fully connected network
			fz = fw * m + fb;
			fa = f(fz);
		};

		auto grad_check = [&]() {
			const float eps = 0.00001f;
			// double cost = c(fa,y);
			double cost1, cost2;
			double g;

			//dcdb
			w[3](2,3) += eps;
			forward();
			cost1 = c(fa,y);
			w[3](2,3) -= 2.*eps;
			forward();
			cost2 = c(fa,y);
			w[3](2,3) += eps;
			g = (cost1-cost2)/(2*eps);
			cout << "dcdw   : " << dcdw[3](0,2*fs+3) << endl;
			cout << "diffquo: " << g << endl;
		};

		I e = 0;
		const R u = .02;
		cout.precision(20);
		while (++e) {
			// COMPUTE NETWORK
			I example = e % train.labels.size();
			x = View<is, is>(train.examples.col(example).data());
			y.setZero();
			y(train.labels(example)) = 1;
			forward();

			// FULLY CONNECTED LAYER DERIVATIVE
			dcdfa = dc(fa, y).transpose();
			for (I i = 0; i < os; ++i) // the ith row of fz is sensitive to elements in the ith row of fw
				for (I j = 0; j < ps*ps; ++j) // but not to elements in other rows of fw
					dfzdfw(i,j + i * ps*ps) = m(j);
			dcdfz = dcdfa; // = dcdfa * dfadfz // dfadfz cancelled by dc
			dcdfw = dcdfz * dfzdfw;
			dcdfb = dcdfz; // = dcdfz * dfzdfb // dfzdfb == identity
			dfzdm = fw;
			dcdm = dcdfz * dfzdm;

			// MAX POOL LAYER DERIVATIVE
			// dmda computed during forward pass

			// CONVOLUTIONAL LAYER DERIVATIVE
			for (I k = 0; k < nf; ++k)
				a[k] = df(a[k]);
			for (I k = 0; k < nf; ++k)
				for (I i = 0; i < as; ++i)
					dadz[k].diagonal().transpose().block<1,as>(0,i*as) = a[k].row(i);
			for (I i = 0; i < as*as; ++i) { // Build the jacobian of the convolution wrt the filter parameters
				I z_row = i / as;           // Each filter has the same input so dzdw is the same for each filter
				I z_col = i % as;
				const auto &submat = x.block<fs, fs>(z_row, z_col);
				for (I j = 0; j < fs*fs; ++j) {
					I w_row = j / fs;
					I w_col = j % fs;
					dzdw(i, j) = submat(w_row, w_col);
				}
			}
			for (I k = 0; k < nf; ++k) {
				dcdz[k] = dcdm * dmda[k] * dadz[k];
				dcdw[k] = dcdz[k] * dzdw;
				dcdb[k] = dcdz[k].sum(); // dzdb == [as*as, 1] vector of ones
			}

			// GRADIENT CHECK
			#ifdef GC
			if (e % 1000==0) {
				grad_check();
			}
			#endif

			// UPDATE WEIGHTS
			for (I i = 0; i < os; ++i)
				fw.row(i) += (-u) * dcdfw.block<1,ps*ps>(0,i*ps*ps);
			fb += (-u) * dcdfb.transpose();
			for (I k = 0; k < nf; ++k) {
				for (I i = 0; i < fs; ++i)
					w[k].row(i) += (-u) * dcdw[k].block<1,fs>(0,i*fs);
				b[k] += (-u) * dcdb[k];
			}

			// ACCURACY CHECK
			if (e % 1000 == 0) {
				double acc = 0.;
				I itters = test.labels.size();
				for (I example = 0; example < itters; ++example) {
					x = View<is, is>(test.examples.col(example).data());
					forward();
					I indx;
					fa.maxCoeff(&indx);
					if (indx == test.labels(example))
						acc += 1.;
				}
				acc /= R(itters);
				cout << "Itter " << e << endl;
				cout << "Accuracy: " << acc << endl;
			}
		}
	}
};

I main() {
	cnn::run();
	return 0;
}
