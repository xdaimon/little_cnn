#include <iostream>
using std::cout; using std::endl;
#include <chrono>

#include "common.h"
#include "mnist_loader.h"

using R = double; // real
using I = int;    // integer

template<I rows, I cols>
using M = Matrix<R, rows, cols>;
template<I sz>
using MD = DiagonalMatrix<R, sz>;
template<I cols>
using V = Matrix<R, 1, cols>;

// gradient check
#define GC

class cnn {
	static constexpr I is = 28;      // image size
	static constexpr I fs = 5;       // convolution filter size, must be odd
	static constexpr I nf = 5;       // # of conv filters, # of feature maps
	static constexpr I pws = 2;      // pool window size, must be 2
	static constexpr I as = is-fs+1; // convolution activation size
	static constexpr I ps = as/pws;  // pooled activation size
	static constexpr I os = 10;      // output vector size

	template<typename T>
	static T f(const T& z) { return 1./(1.+(-z).array().exp()); }
	template<typename T>
	static T df(const T& A) {
		const auto& o = A.array();
		return o*(1-o);
	}

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
		I e = 0;
		const R u = .02;
		const R lambda = .001;

		Data test;
		Data train;
		Data validation;
		load_data(train, test, validation);

		M<is,is> x;
		M<fs,fs> w[nf];
		for (I i = 0; i < nf; ++i)
			w[i] = M<fs, fs>::Random()/std::sqrt(fs*fs);
		R b[nf]; for (I i = 1; i < nf; ++i) b[i] = 0;
		M<as,as> z;
		M<as,as> a[nf];
		M<os,1>  y; y.setZero();

		M<ps*ps,1>      m[nf];
		M<os,ps*ps>    fw;
		for (I i = 0; i < os; ++i)
			for (I j = 0; j < ps*ps; ++j)
				fw(i,j) = exp(-(pow((i*2.f-os)/float(os), 2) + pow((j*2.f-ps*ps)/float(ps*ps), 2))) / (ps*ps);
		M<os,1>        fb; fb.setZero();
		M<os,1>        fz;
		M<os,1>        fa;

		// jacobians
		M<1,os>        dcdfa;
		MD<os>        dfadfz;
		M<1,os>        dcdfz;

		M<os,os*ps*ps> dfzdfw;
		M<os,os>       dfzdfb; dfzdfb.setIdentity();

		M<1,os*ps*ps>  dcdfw;
		M<1,os>        dcdfb;

		M<os,ps*ps>    dfzdm;
		M<ps*ps,as*as> dmda[nf]; // ones and zeros
		MD<as*as>      dadz[nf]; for (I k = 0; k < nf; ++k) dadz[k].setZero();
		M<ps*ps,as*as> dmdz[nf];
		M<1,as*as>     dcdz[nf];

		M<as*as,fs*fs> dzdw;
		M<as*as,1>     dzdb; dzdb.setOnes();
		M<1,fs*fs>     dcdw[nf];
		R              dcdb[nf];

		// w z a m fw fz fa c

		auto LOG = [&]() {
			// Nodes
			// cout << "x" << endl << x << endl << endl;
			for (I i = 0; i < nf; ++i)
				cout << "w["<<i<<"]" << endl << w[i] << endl << endl;
			// for (I i = 0; i < os; ++i)
			// 	cout << "b[i]" << endl << b[i] << endl << endl;
			// cout << "z" << endl << z << endl << endl;
			// cout << "a" << endl << a << endl << endl;
			// cout << "y" << endl << y << endl << endl;
			//
			// cout << "m" << endl;
			// for (I i = 0; i < ps; ++i)
			// 	cout << m.subMatrix<ps,1>(i*ps,0).transpose() << endl;
			// cout << endl;
			//
			// cout << "fw" << endl << fw << endl << endl;
			// cout << "fb" << endl << fb << endl << endl;
			// cout << "fz" << endl << fz << endl << endl;
			// cout << "fa" << endl << fa << endl << endl;
			//
			// // Gradients
			//
			// cout << "dcdfa:"  << endl << dcdfa  << endl << endl;
			// // cout << "dfadfz:" << endl << dfadfz << endl << endl;
			// cout << "dcdfz:"  << endl << dcdfz  << endl << endl;
			// cout << "dfzdfw:" << endl << dfzdfw << endl << endl;
			// // cout << "dfzdfb:" << endl << dfzdfb << endl << endl;
			// cout << "dcdfw:"  << endl << dcdfw  << endl << endl;
			// // cout << "dcdfb:"  << endl << dcdfb  << endl << endl;
			// cout << "dfzdm:"  << endl << dfzdm  << endl << endl;
			// cout << "dmda:"   << endl << dmda   << endl << endl;
			// cout << "dadz:"   << endl << dadz   << endl << endl;
			// cout << "dcdz:"   << endl << dcdz   << endl << endl;
			// cout << "dzdw:"   << endl << dzdw   << endl << endl;
			// // cout << "dzdb:"   << endl << dzdb   << endl << endl;
			// cout << "dcdw:"   << endl << dcdw   << endl << endl;
			// // cout << "dcdb:"   << endl << dcdb   << endl << endl;
			// for (I i = 0; i < dmda.cols(); ++i)
			// 	cout << dmda.col(i).sum() << " "; // will sum to at most one
			// cout << endl << endl;
			// for (I i = 0; i < dmda.rows(); ++i)
			// 	cout << dmda.row(i).sum() << " "; // will sum to one
			// cout << endl;
		};

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
			for (I k = 0; k < nf; ++k) {
				dmda[k].setZero();
				for (I i = 0; i < ps; ++i) {
					I row,col;
					for (I j = 0; j < ps; ++j) {
						m[k](i*ps+j,0) = a[k].block<pws,pws>(pws*i,pws*j).maxCoeff(&row, &col);
						dmda[k](i*ps+j,(i*pws+row)*as+j*pws+col) = 1;
					}
				}
			}

			// Compute the fully connected network
			fz.setZero();
			for (I i = 0; i < nf; ++i)
				fz += fw * m[i];
			fz += fb;
			fa = f(fz);
		};

		auto grad_check = [&]() {
			const float eps = 0.00001f;
			// double cost = c(fa,y);
			double cost1, cost2;
			double g;

			//dcdb
			b[3] += eps;
			forward();
			cost1 = c(fa,y);
			b[3] -= 2.*eps;
			forward();
			cost2 = c(fa,y);
			b[3] += eps;
			g = (cost1-cost2)/(2*eps);
			cout << "dcdb   : " << dcdb[3] << endl;
			cout << "diffquo: " << g << endl;
		};

		cout.precision(20);
		auto start = std::chrono::steady_clock::now();
		while (++e < 1000) {

			for (I T = 0; T < 13; ++T) {
				I example = rand() % train.labels.size();
				// Copy data from test matrix to network input matrix
				for (I i = 0; i < is; ++i)
					x.row(i) = train.examples.block<is,1>(i*is,example).transpose();
				y.setZero();
				y(train.labels(example)) = 1;

				forward();




				// Compute jacobians



				// FULLY CONNECTED LAYER DERIVATIVE
				dcdfa = dc(fa, y).transpose();
				// dfadfz = df( fa ).asDiagonal();
				dfzdfw.setZero();
				for (I k = 0; k < nf; ++k)
					for (I i = 0; i < os; ++i) // the ith row of fz is sensitive to elements in the ith row of fw
						for (I j = 0; j < ps*ps; ++j) // but not to elements in other rows of fw
							dfzdfw(i,j + i * ps*ps) += m[k](j,0);
				// dfzdfb.setIdentity(); // already computed
				// dcdfz = dcdfa * dfadfz; // dfadfz cancelled by dc
				dcdfz = dcdfa;
				dcdfw = dcdfz * dfzdfw; // This could be reduced to (dcdfz.T)*(m.T) which would produce a dcdfw that has the same size as fw
				dcdfb = dcdfz * dfzdfb;
				dfzdm = fw;




				// MAX POOL LAYER DERIVATIVE
				// dmda[k] computed during forward pass




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
				// dzdb.setOnes(); // already computed
				for (I k = 0; k < nf; ++k) {
					dcdz[k] = dcdfz * dfzdm * dmda[k] * dadz[k];
					dcdw[k] = dcdz[k] * dzdw;
					dcdb[k] = dcdz[k] * dzdb;
				}




				// Compute gradient check
				#ifdef GC
				if (T == 0 && e % 50==0) {
					grad_check();
				}
				#endif



				// UPDATE WEIGHTS
				for (I i = 0; i < os; ++i)
					fw.row(i) += -u * dcdfw.block<1,ps*ps>(0,i*ps*ps);
				fb += -u * dcdfb.transpose();

				for (I k = 0; k < nf; ++k) {
					for (I i = 0; i < fs; ++i)
						w[k].row(i) += -u * dcdw[k].block<1,fs>(0,i*fs);
					b[k] += -u * dcdb[k];
				}

			}

			if (e % 50 == 0) {
				double acc = 0.;
				I itters = test.labels.size();
				for (I example = 0; example < itters; ++example) {
					for (I i = 0; i < is; ++i)
						x.row(i) = test.examples.block<is,1>(i*is,example).transpose();
					forward();
					I indx;
					fa.maxCoeff(&indx);
					if (indx == test.labels(example))
						acc += 1.;
				}
				acc /= R(itters);
				cout << "Epoch " << e << endl;
				cout << "Accuracy: " << acc << endl;
			}
		}
		cout << (std::chrono::steady_clock::now() - start).count() / 1e9 << endl;
		LOG();
		I XX;
		std::cin >> XX;
		cout << XX * 3 << endl;
	}
};

I main() {
	cnn::run();
	return 0;
}
