#include "header.h"
#include <vector>
// #include <png++/png.hpp>

typedef double R;
typedef int I;

template<I rows, I cols>
using M = Matrix<R, rows, cols>;

#define subMatrix block

// Cheap gradient check
// #define GC

// template<I is, I fs, I pws>
class cnn {
	static constexpr I is = 28;      // is = image size
	static constexpr I fs = 5;       // fs = convolution filter size, must be odd
	static constexpr I pws = 2;      // pws = pool window size, must be 2
	static constexpr I as = is-fs+1; // as = convolution activation size
	static constexpr I ps = as/pws;  // ps = pooled activation size
	static constexpr I os = 10;      // os = output vector size

	template<I rows, I cols>
	static M<rows, cols> f(M<rows, cols>& A) { return 1./(1.+(-A).array().exp()); }
	template<I rows, I cols>
	static M<rows, cols> df(M<rows, cols>& A) { return A.array()*(1-A.array()); }

	template<I rows, I cols>
	static R c(M<rows, cols>& A, M<rows, cols>& Y) { return -(Y.array()*A.array().log()+(1-Y.array())*(1-A.array()).log()).sum(); }
	template<I rows, I cols>
	static M<rows, cols> dc(M<rows, cols>& A, M<rows, cols>& Y) { return A-Y; }

	public:
	static void run() {
		I E = 1000;
		I e = 0;
		R u = .04;
		R lambda = .001;

		Data test;
		Data train;
		Data validation;
		load_data(train, test, validation);

		M<is,is> x;
		M<fs,fs> w[os];
		// TODO Initialize to range [-1, 1] ?
		for (I i = 0; i < fs; ++i)
			for (I j = 0; j < fs; ++j)
				w[0](i,j) = exp(-(pow(i-fs/2., 2) + pow(j-fs/2., 2)));
		for (I i = 1; i < os; ++i) w[i] = w[0];

		R b[os];
		for (I i = 1; i < os; ++i) b[i] = 0;
		M<as,as> z;
		M<as,as> a;
		M<os,1>  y; y.setZero();

		M<ps*ps,1>      m;
		M<os,ps*ps>    fw;
		for (I i = 0; i < os; ++i)
			for (I j = 0; j < ps*ps; ++j)
				fw(i,j) = exp(-(pow((i*2.f-os)/float(os), 2) + pow((j*2.f-ps*ps)/float(ps*ps), 2))) / (ps*ps);
		M<os,1>        fb; fb.setZero();
		M<os,1>        fz;
		M<os,1>        fa;

		// jacobians
		M<1,os>        dcdfa;
		M<os,os>       dfadfz; // diagonal
		M<1,os>        dcdfz;

		M<os,os*ps*ps> dfzdfw;
		M<os,os>       dfzdfb; dfzdfb.setIdentity();

		M<1,os*ps*ps>  dcdfw;
		M<1,os>        dcdfb;

		M<os,ps*ps>    dfzdm;
		M<ps*ps,as*as> dmda; // ones and zeros
		M<as*as,as*as> dadz; dadz.setZero(); // diagonal
		M<1,as*as>     dcdz;

		M<as*as,fs*fs> dzdw;
		M<as*as,1>     dzdb; dzdb.setOnes();
		M<1,fs*fs>     dcdw;
		R              dcdb;

		// w z a m fw fz fa c

		auto LOG = [&]() {
			// Nodes
			// cout << "x" << endl << x << endl << endl;
			for (I i = 0; i < os; ++i)
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

		// label for example from loaded img file
		I label = 0;

		auto forward = [&](I my_label) {
			// Compute the convolution
			for (I i = 0; i < as; ++i)
				for (I j = 0; j < as; ++j)
					z(i,j) = (w[my_label].array() * x.subMatrix<fs,fs>(i,j).array()).sum() + b[my_label];
			a = f(z);

			// Compute the pool and also the derivative of the pooling wrt its inputs
			dmda.setZero();
			for (I i = 0; i < ps; ++i) {
				I row,col;
				for (I j = 0; j < ps; ++j) {
					m(i*ps+j,0) = a.subMatrix<pws,pws>(pws*i,pws*j).maxCoeff(&row, &col);
					dmda(i*ps+j,(i*pws+row)*as+j*pws+col) = 1;
				}
			}

			// Compute the fully connected network
			fz = fw * m + fb;
			fa = f(fz);
		};

		auto grad_check = [&]() {
			const float eps = 0.00001;
			// double cost = c(fa,y);
			double cost1, cost2;
			double g;

			// //dcdb
			// b[label] += eps; // M<os,1>
			// forward(label);
			// cost1 = c(fa,y);
			// b[label] -= 2.*eps;
			// forward(label);
			// cost2 = c(fa,y);
			// b[label] += eps;
			// g = (cost1-cost2)/(2*eps);
			// cout << "dcdb: " << dcdb << endl;
			// cout << "  dq: " << g << endl;

			// //dcdw
			// w[label](2,2) += eps; // M<fs,fs>
			// forward(label);
			// cost1 = c(fa,y);
			// // g = (c(fa,y)-cost) / eps;
			// w[label](2,2) -= 2*eps;
			// forward(label);
			// cost2 = c(fa,y);
			// w[label](2,2) += eps;
			// g = (cost1-cost2) / (2*eps);
			// cout << "dcdw(...): " << dcdw(0,2*fs + 2) << endl;
			// cout << "       dq: " << g << endl;

			//dcdfb
			fb(4) += eps;
			forward(label);
			cost1 = c(fa,y);
			fb(4) -= 2.*eps;
			forward(label);
			cost2 = c(fa,y);
			fb(4) += eps;
			g = (cost1 - cost2)/ (2*eps);
			cout << "dcdfb: " << dcdfb(4) << endl; // M<1,os*ps*ps>
			cout << "   dq: " << g << endl;

			// //dcdfw
			// fw(2,10) += eps; // M<os,ps*ps>
			// forward(label);
			// g = (c(fa,y)-cost) / eps;
			// fw(2,10) -= eps;
			// cout << "dcdfw(...): " << dcdfw(0,2*ps*ps + 10) << endl; // M<1,os*ps*ps>
			// cout << "        dq: " << g << endl;

		};

		cout.precision(20);
		while ( e++ < E ) {

			for (I T = 0; T < 13; ++T) {
				I example = rand() % train.labels.size();
				// Copy data from test matrix to network input matrix
				for (I i = 0; i < is; ++i)
					x.row(i) = train.examples.subMatrix<is,1>(i*is,example).transpose();
				y(label) = 0;
				label = train.labels(example);
				y(label) = 1;
				if (y.sum() != 1) {
					cout << "error label vector incorrect" << endl;
					exit(0);
				}

				forward(label);

				// Compute jacobians
				dcdfa = dc( fa, y ).transpose();
				// dfadfz = df( fa ).asDiagonal();
				dfzdfw.setZero();
				for (I i = 0; i < os; ++i) // the ith row of fz is sensitive to elements in the ith row of fw
					for (I j = 0; j < ps*ps; ++j) // but not to elements in other rows of fw
						dfzdfw(i,j + i * ps*ps) = m(j,0);
				// dfzdfb.setIdentity(); // already computed
				// dcdfz = dcdfa * dfadfz; // dfadfz cancelled by dc
				dcdfz = dcdfa;
				dcdfw = dcdfz * dfzdfw;
				dcdfb = dcdfz * dfzdfb;
				dfzdm = fw;
				// dmda already computed
				a = df( a );
				for (I i = 0; i < as; ++i)
					dadz.subMatrix<as,as>(as*i,as*i) = a.row(i).asDiagonal();
				for (I i = 0; i < as*as; ++i) {
					I z_row = i / as;
					I z_col = i % as;
					for (I j = 0; j < fs*fs; ++j) {
						I w_row = j / fs;
						I w_col = j % fs;
						// z(i,j) = (w[label].array() * x.subMatrix<fs,fs>(i,j).array()).sum() + b[label];
						dzdw(i, j) = (x.subMatrix<fs,fs>(z_row, z_col))(w_row, w_col);
					}
				}
				// dzdb.setOnes(); // already computed
				dcdz = dcdfz * dfzdm * dmda * dadz;
				dcdw = dcdz * dzdw;
				dcdb = dcdz * dzdb;

				// Compute cheap gradient check
				#ifdef GC
				if (example == 50) {
					grad_check();
				}
				#endif

				// Update weights
				for (I i = 0; i < os; ++i) {
					// fw.row(i) += -u * dcdfw.subMatrix<1,ps*ps>(0,i*ps*ps) - lambda*fw.row(i);
					fw.row(i) += -u * dcdfw.subMatrix<1,ps*ps>(0,i*ps*ps);
				}
				fb += -u * dcdfb.transpose();
				for (I i = 0; i < fs; ++i) {
					// w[label].subMatrix<1,fs>(i,0) += -u * dcdw.subMatrix<1,fs>(0,i*fs) - lambda*w[label].subMatrix<1,fs>(i,0);
					w[label].row(i) += -u * dcdw.subMatrix<1,fs>(0,i*fs);
				}
				b[label] += -u * dcdb;
			}

			cout << "Epoch " << e << endl;
			if (e % 50 == 0) {
				double acc = 0.;
				I itters = test.labels.size();
				for (I example = 0; example < itters; ++example) {
					for (I i = 0; i < is; ++i)
						x.row(i) = test.examples.subMatrix<is,1>(i*is,example).transpose();
					I my_label = test.labels(example); // dont use label variable
					
					forward(my_label);
					I indx;
					fa.maxCoeff(&indx);
					if (indx == my_label) {
						acc += 1.;
					}
				}
				acc /= R(itters);
				cout << "Accuracy: " << acc << endl;
			}
		}
		LOG();
	}
};

I main() {
	cnn::run();
	return 0;
}
