#ifndef SS_CONVOLUTION_CONV_H
#define SS_CONVOLUTION_CONV_H

#include <vector>
#include <iostream>
#include <complex>
#include <cmath>
#include <valarray>
#include "AudioFile.h"

typedef std::valarray<std::complex<double>> ComplexArray;
using namespace std::complex_literals;


class Conv {

private:
    ComplexArray fourier(const std::valarray<double> &x);

    ComplexArray ifourier2(const ComplexArray &X);

    std::valarray<double> ifourier(const ComplexArray &x);

public:
    std::valarray<double> MyConvolve(const std::valarray<double> &x, const std::valarray<double> &y);

    std::valarray<double> MyConvolve(const AudioFile<double> &x, const AudioFile<double> &y);

};


#endif //SS_CONVOLUTION_CONV_H
