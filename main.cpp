#include "header.h"
#include <vector>
#include <png++/png.hpp>

typedef float R;
typedef int I;

template<I rows, I cols>
using M = Matrix<R, rows, cols>;

#define subMatrix block

// template<I is, I fs, I pws>
class cnn {
	static constexpr I is = 28;      // is = image size
	static constexpr I fs = 6;       // fs = convolution filter size
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
		if(!Plot::init(200))
			return;
		I E = 1000;
		I e = 0;
		R u = .04;
		R lambda = .0001;

		Data test;
		Data train;
		Data validation;
		load_data(train, test, validation);
		// if test.lable(0) == 4, then the first example is a 4

		png::image< png::rgb_pixel_16 > image("../testDigit.png");
		M<is,is> testDigit;
		I testLabel = test.labels(0);
		for (int i = 0; i < testDigit.rows(); ++i)
			for (int j = 0; j < testDigit.cols(); ++j)
				testDigit(i,j) = image[i][j].red/65535.;

		M<is,is> x;
		M<fs,fs> w[10];
		for (I i = 0; i < fs; ++i)
			for (I j = 0; j < fs; ++j)
				w[0](i,j) = expf(-1/2.f*(powf(i-fs/2,2) + powf(j-fs/2,2)));
		for (I i = 0; i < 10; ++i) w[i] = w[0];
		R        b[10]; b[0] = 0;
		for (I i = 0; i < 10; ++i) b[i] = b[0];
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
		M<as*as,1>     dzdb; dzdb.setOnes();
		M<1,fs*fs>     dcdw;
		R              dcdb;

		// w z a m fw fz fa c

		auto LOG = [&](){
			// Nodes
			// cout << "x" << endl << x << endl << endl;
			for (I i = 0; i < 10; ++i)
				cout << "w["<<i<<"]" << endl << w[i] << endl << endl;
			// for (I i = 0; i < 10; ++i)
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

		cout.precision(2);
		cout << "Initial w : " << w[0] << endl << endl;
		// label for example
		I label = 0;
		while ( e++ < E ) {

			for (I example = 0, n = 0; example < 100; ++example) {
				// Copy data from test matrix to network input matrix
				for (I i = 0; i < is; ++i)
					x.row(i) = test.examples.subMatrix<is,1>(i*is,example).transpose();
				label = test.labels(example);
				y.setZero(); y(label, 0) = 1;

				// Compute the convolution
				for (I i = 0; i < as; ++i)
					for (I j = 0; j < as; ++j)
						z(i,j) = (w[label].array() * x.subMatrix<fs,fs>(i,j).array()).sum() + b[label];
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

				// Compute derivatives
				dcdfa  = dc( fa, y ).transpose();
				// dfadfz = df( fa ).asDiagonal();
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
				// // dmda already computed
				// a = df( a );
				dadz.setZero();
				for (I i = 0; i < as; ++i)
					dadz.subMatrix<as,as>(as*i,as*i) += a.row(i).asDiagonal();
				for (I i = 0; i < as; ++i)
					for (I j = 0; j < fs; ++j)
						dzdw.subMatrix<as,fs>(as*i,fs*j) = x.subMatrix<as,fs>(j,i);
				// dzdb.setOnes(); // already computed
				dcdz = dcdfz * dfzdm * dmda * dadz;
				dcdw = dcdz * dzdw;
				dcdb = dcdz * dzdb;

				// Update weights
				for (I i = 0; i < os; ++i)
					fw.row(i) += -u * dcdfw.subMatrix<1,ps*ps>(0,i*ps*ps) - lambda*fw.row(i);
				fb += -u * dcdfb.transpose();
				for (I i = 0; i < fs; ++i)
					w[label].subMatrix<1,fs>(i,0) += -u * dcdw.subMatrix<1,fs>(0,i*fs) - lambda*w[label].subMatrix<1,fs>(i,0) ;
				b[label] += -u * dcdb;

			}

			// Test given an image created in gimp
			y.setZero(); y(testLabel,0) = 1;
			// Compute the convolution
			for (I i = 0; i < as; ++i)
				for (I j = 0; j < as; ++j)
					z(i,j) = (w[testLabel].array() * testDigit.subMatrix<fs,fs>(i,j).array()).sum() + b[testLabel];
			a = f(z);

			// Compute the pool and also the derivative of the pooling wrt its inputs
			for (I i = 0; i < ps; ++i)
				for (I j = 0; j < ps; ++j)
					m(i*ps+j,0) = a.subMatrix<pws,pws>(pws*i,pws*j).maxCoeff();
			fz = fw * m + fb;
			fa = f(fz);

			Plot::draw(e/R(E), c(fa, y)/10., WHITE);
			cout << "cost on testDigit.png:: " << c(fa, y) << endl;
		}
		LOG();

		cout << "Done" << endl;
		Plot::display();
	}
};

I main() {
	cnn::run();
	return 0;
}
