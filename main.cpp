#include <iostream>
#include <iomanip>
using std::cout;
using std::endl;
#include <cmath>

#define EIGEN_STACK_ALLOCATION_LIMIT 0
#include <eigen3/Eigen/Eigen>
using namespace Eigen;

#include "plot.h"
#include "mnist_loader.h"

typedef float R;
typedef int I;

template<I rows, I cols>
using M = Matrix<R, rows, cols>;

// template<I is, I fs, I pws>
class nn {
	static constexpr I is = 28;      // is = image size
	static constexpr I fs = 5;       // fs = convolution filter size
	static constexpr I pws = 2;      // pws = pool window size
	static constexpr I as = is-fs+1; // as = convolution activation size
	static constexpr I ps = as/pws;  // ps = pooled activation size
	static constexpr I os = 10;      // os = output vector size

	template<I rows, I cols>
	static M<rows,cols> f(M<rows,cols> A) { return 1./(1.+(-A).array().exp()); }
	template<I rows, I cols>
	static M<rows,cols> df(M<rows,cols> A) { return A.array()*(1-A.array()); }

	template<I rows, I cols>
	static R c(M<rows,cols> A, M<rows,cols> Y) { return -(Y.array()*A.array().log()+(1-Y.array())*(1-A.array()).log()).sum(); }
	template<I rows, I cols>
	static M<rows,cols> dc(M<rows,cols> A, M<rows,cols> Y) { return A-Y; }

	public:
	static void run() {
		if(!Plot::init(400))
			return;
		I E = 400;
		I e = 0;
		R u = .1;

		Data test;
		Data train;
		Data validation;
		load_data(train, test, validation);

		M<is,is> x; for (I i = 0; i < is; ++i) x.row(i) = test.examples.block<is,1>(i*is,0).transpose();
		M<fs,fs> w; for (I i = 0; i < fs; ++i) for (I j = 0; j < fs; ++j) w(i,j) = exp(-1/2.*(pow(i-fs/2,2) + pow(j-fs/2,2)));
		R        b; b = 0;
		M<as,as> z;
		M<as,as> a;
		M<os,1>  y; y.setZero(); y(test.labels(0),0) = 1;

		M<ps*ps,1>      m;
		M<os,ps*ps>    fw; fw.setOnes(); fw*=1./fw.cols();
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
		M<as*as,as*as> dadz; // diagonal
		M<1,as*as>     dcdz;

		M<as*as,fs*fs> dzdw;
		M<as*as,1>     dzdb;
		M<1,fs*fs>     dcdw;
		R              dcdb;

		// w z a m fw fz fa c

		cout.precision(2);
		cout << "Initial w : " << w << endl << endl;
		while ( e++ < E ) {
			// Compute the convolution
			for (I i = 0; i < as; ++i)
				for (I j = 0; j < as; ++j)
					z(i,j) = (w.array() * x.block<fs,fs>(i,j).array()).sum() + b;

			a = f( z );

			// Compute the pool and also the derivative of the pooling wrt its inputs
			dmda.setZero();
			for (I i = 0; i < ps; ++i)
				for (I j = 0; j < ps; ++j) {
					I r,c;
					m(i*ps+j,0) = a.block<pws,pws>(pws*i,pws*j).maxCoeff(&r, &c);
					dmda(i*ps+j,r*as+c) = 1;
				}

			fz = fw * m + fb;
			fa = f( fz );
			dcdfa  = dc( fa, y ).transpose();
			// dfadfz = df( fa ).asDiagonal();

			if ( e > E-1 ) {
				cout << "x" << endl << x << endl << endl;
				cout << "w" << endl << w << endl << endl;
				cout << "b" << endl << b << endl << endl;
				cout << "z" << endl << z << endl << endl;
				cout << "a" << endl << a << endl << endl;
				cout << "y" << endl << y << endl << endl;

				cout << "m" << endl;
				for (I i = 0; i < ps; ++i)
					cout << m.block<ps,1>(i*ps,0).transpose() << endl;
				cout << endl;

				cout << "fw" << endl << fw << endl << endl;
				cout << "fb" << endl << fb << endl << endl;
				cout << "fz" << endl << fz << endl << endl;
				cout << "fa" << endl << fa << endl << endl;
			}

			dfzdfw.setZero();
			for (I i = 0; i < os; ++i)
				for (I jj = 0; jj < ps*ps; ++jj)
					dfzdfw(i,jj + i * ps*ps) = m(jj,0);
			// dfzdfb.setIdentity(); // already computed
			// dcdfz = dcdfa * dfadfz; // dfadfz cancelled by dc
			dcdfz = dcdfa;
			dcdfw = dcdfz * dfzdfw;
			dcdfb = dcdfz * dfzdfb;
			dfzdm = fw;
			// dm/da already computed
			a = df( a );
			dadz.setZero();
			for (I i = 0; i < as; ++i)
				dadz.block<as,as>(as*i,as*i) = a.row(i).asDiagonal();
			for (I i = 0; i < as; ++i)
				for (I j = 0; j < fs; ++j)
					dzdw.block<as,fs>(as*i,fs*j) = x.block<as,fs>(j,i);

			dzdb.setOnes();
			dcdz = dcdfz * dfzdm * dmda * dadz;

			dcdw = dcdz * dzdw;
			dcdb = dcdz * dzdb;
			for (I i = 0; i < os; ++i)
				fw.row(i) += -u * dcdfw.block<1,ps*ps>(0,i*ps*ps);
			fb += -u * dcdfb.transpose();
			for (I i = 0; i < fs; ++i)
				w.block<1,fs>(i,0) += -u * dcdw.block<1,fs>(0,i*fs);
			b += -u * dcdb;

			if ( e > E-1 ) {
				cout << "dcdfa:"  << endl << dcdfa  << endl << endl;
				// cout << "dfadfz:" << endl << dfadfz << endl << endl;
				cout << "dcdfz:"  << endl << dcdfz  << endl << endl;
				// cout << "dfzdfw:" << endl << dfzdfw << endl << endl;
				// cout << "dfzdfb:" << endl << dfzdfb << endl << endl;
				cout << "dcdfw:"  << endl << dcdfw  << endl << endl;
				// cout << "dcdfb:"  << endl << dcdfb  << endl << endl;
				cout << "dfzdm:"  << endl << dfzdm  << endl << endl;
				// cout << "dmda:"   << endl << dmda   << endl << endl;
				// cout << "dadz:"   << endl << dadz   << endl << endl;
				cout << "dcdz:"   << endl << dcdz   << endl << endl;
				cout << "dzdw:"   << endl << dzdw   << endl << endl;
				// cout << "dzdb:"   << endl << dzdb   << endl << endl;
				cout << "dcdw:"   << endl << dcdw   << endl << endl;
				// cout << "dcdb:"   << endl << dcdb   << endl << endl;
			}
			Plot::draw(e/R(E), c( fa, y ));
		}
		Plot::display();
	}
};

I main() {
	nn::run();
	return 0;
}
