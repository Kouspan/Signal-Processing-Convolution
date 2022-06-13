#include "Conv.h"
#include <valarray>
#include <complex>
#include <cmath>
#include <future>

/**Recursive Fast fourier transform radix-2.
 * Returns the DFT of x in valarray<complex<double>>.
 * Array x has to have its size be a power of 2.**/
ComplexArray Conv::fourier(const std::valarray<double> &x) {
    int n = x.size();
    if (n == 1) {   //if array has only 1 element, convert it to complex array and return it
        ComplexArray y = {x[0]};
        return y;
    } else {
        //initialisation of arrays
        ComplexArray y(n);
        ComplexArray factor(n / 2);
        ComplexArray X_even, X_odd;

        //split array x to 2 arrays. x_even has all even places of x and x_odd all the odd places.
        std::valarray<double> x_even = x[std::slice(0, n / 2, 2)];
        std::valarray<double> x_odd = x[std::slice(1, n / 2, 2)];

        //call recursively fourier with the 2 arrays and pass their DFTs to X_even and X_odd.
        X_even = fourier(x_even);
        X_odd = fourier(x_odd);

        //Calculate only the first n/2 twiddle factors, because they are symmetric. negative exponent
        for (int i = 0; i < n / 2; ++i)
            factor[i] = (std::exp(-2i * ((M_PI * i) / n)));

        //Combination of the 2 arrays with butterfly operations.
        for (int i = 0; i < n / 2; ++i) {
            y[i] = (X_even[i] + factor[i] * X_odd[i]);
            y[i + n / 2] = (X_even[i] - factor[i] * X_odd[i]);
        }
        return y;
    }

}

/**Wrapper function of ifourier2. @Returns the real values of complex array from ifourier2.  **/
std::valarray<double> Conv::ifourier(const ComplexArray &x) {
    auto y = ifourier2(x);
    int n = x.size();
    std::valarray<double> result(n);
    for (int i = 0; i < n; i++)
        result[i] = y[i].real();
    result /= n;
    return result;
}

/**Recursive Inverse Fast Fourier Transform radix-2.
 * @Returns complex array instead of real array for the recursion to be able to work.
 * Exact copy of fourier function with different sign in the exponent of the twiddle factors.**/
ComplexArray Conv::ifourier2(const ComplexArray &X) {
    int n = X.size();
    if (n == 1)
        return X;
    else {
        //initialisation of arrays
        ComplexArray y(n);
        ComplexArray factor(n / 2);
        ComplexArray x_even, x_odd;

        //split array X to 2 arrays. X_even has all even places of X and X_odd all the odd places.
        ComplexArray X_even(X[std::slice(0, n / 2, 2)]);
        ComplexArray X_odd(X[std::slice(1, n / 2, 2)]);

        //call recursively fourier with the 2 arrays and pass their DFTs to x_even and x_odd.
        x_even = ifourier2(X_even);
        x_odd = ifourier2(X_odd);

        //Calculate only the first n/2 twiddle factors, because they are symmetric. positive exponent.
        for (int i = 0; i < n / 2; ++i)
            factor[i] = (std::exp(2i * ((M_PI * i) / n)));


        //Combination of the 2 arrays with butterfly operations.
        for (int i = 0; i < n / 2; ++i) {
            y[i] = (x_even[i] + factor[i] * x_odd[i]);
            y[i + n / 2] = (x_even[i] - factor[i] * x_odd[i]);
        }

        return y;
    }

}

/**@Returns Convolution of arrays x and y.
 * Finds the DFTs of both arrays, multiplies them point-wise and returns the Inverse DFT of the multiplication result.
 * This is possible because a convolution of 2 arrays in the time domain is equal to the multiplication of their
 * DTFs in the frequency domain. Since we have 2 separate DTF to calculate, we can use multithreading for parallel
 * computing.**/
std::valarray<double> Conv::MyConvolve(const std::valarray<double> &x, const std::valarray<double> &y) {
    //initialisation of sizes.
    int x_size(x.size()), y_size(y.size());
    int conv_size = x_size + y_size - 1;

    //find the next number greater than conv_size that is a power of 2.
    int f_size = std::pow(2, std::ceil(log2(conv_size)));

    //zero pad both both arrays to have size be a power 2.
    std::valarray<double> x_padded(f_size), y_padded(f_size);
    for (int i = 0; i < x_size; ++i)
        x_padded[i] = x[i];
    for (int i = 0; i < y_size; ++i)
        y_padded[i] = y[i];

    //compute one of the 2 DTFs in a new thread.
    auto x_fourier = std::async(std::launch::async, &Conv::fourier, Conv{}, x_padded);
    auto y_fourier = fourier(y_padded);

    //multiply the arrays.
    ComplexArray result_fourier = x_fourier.get() * y_fourier;

    //calculate the Inverse DFT of the result.
    auto result_padded = ifourier(result_fourier);

    //convert results from complex array to real array
    std::valarray<double> result(conv_size);

    //crop the trailing places from the real array that were used for DTF calculation.
    result[std::slice(0, conv_size, 1)] = result_padded;
    return result;
}

/**Overloaded function that works with 2 AudioFiles instead of valarrays.
 * Converts the AudioFiles to valarrays and calls MyConvolve(valarray<double>, valarray<double>).**/
std::valarray<double> Conv::MyConvolve(const AudioFile<double> &x, const AudioFile<double> &y) {
    int x_size(x.samples[0].size()), y_size(y.samples[0].size());
    std::valarray<double> val_x(x_size), val_y(y_size);
    for (int i = 0; i < x_size; ++i)
        val_x[i] = x.samples[0][i];
    for (int i = 0; i < y_size; ++i)
        val_y[i] = y.samples[0][i];
    return MyConvolve(val_x, val_y);
}

