#include <iostream>
#include "Conv.h"
#include <iomanip>
#include "chrono"
#include <random>

using namespace std;

/**@Returns a random number from a gaussian distribution **/
double getNormalRandom() {
    static std::mt19937 generator(std::chrono::system_clock::now().time_since_epoch().count());
    static std::normal_distribution<double> distribution(0, 0.1);
    return distribution(generator);
}

/**@Returns a random number from a uniform distribution  **/
double getRandom() {
    static std::mt19937 generator(std::chrono::system_clock::now().time_since_epoch().count());
    static std::uniform_real_distribution<double> distribution(0, 100);
    return distribution(generator);
}

/**Prints a valarray<double> x, 5 elements every row  **/
void printSequence(const valarray<double> &x) {
    for (int i = 0; i < x.size() - (x.size() % 5); i += 5) {
        cout << "Column " << i << "\t through \t" << "Column " << i + 5 << endl;
        cout << x[i] << ", " << x[i + 1] << ", " << x[i + 2] << ", " << x[i + 3] << ", " << x[i + 4]
             << endl;
    }
    if (x.size() % 5 != 0) {
        cout << "Column " << x.size() - (x.size() % 5) << "\t through \t" << "Column " << x.size() << endl;
        for (unsigned long long i = x.size() - (x.size() % 5); i < x.size() - 1; ++i)
            cout << x[i] << ", ";
        cout << x[x.size() - 1] << endl;
    }

}

int main() {
    Conv conv;
    //Initialise audio files.
    AudioFile<double> audio_sample("../sample_audio.wav");
    AudioFile<double> audio_noise("../pink_noise.wav");
    AudioFile<double> audio_new;

    char ans;
    int mode;
    while (true) {
        cout << "1. Convolution of a random sequence x and h = [0.2, 0.2, 0.2, 0.2, 0.2]." << endl;
        cout << "2. Convolution of pink_noise.wav and sample_audio.wav." << endl;
        cout << "3. Convolution of a white noise signal and sample_audio.wav." << endl;
        cout << "0. Exit." << endl;
        cout << "Select Task:";
        cin >> mode;
        switch (mode) {
            case 1: {
                int n;
                do {
                    cout << "Enter length N of sequence x. N>10: ";
                    cin >> n;

                } while (n <= 10);
                valarray<double> x(n);
                valarray<double> h = {0.2, 0.2, 0.2, 0.2, 0.2};
                for (int i = 0; i < n; ++i)
                    x[i] = getRandom();
                cout << "Print sequence? n/y." << endl;
                cin >> ans;
                if (tolower(ans) == 'y')
                    printSequence(x);
                cout << "Calculating convolution... ";
                auto now = chrono::steady_clock::now();
                auto y = conv.MyConvolve(x, h);
                auto end = chrono::steady_clock::now();
                cout << "Done!" << endl;
                cout << "Time elapsed: ";
                cout << chrono::duration_cast<chrono::milliseconds>(end - now).count() << " ms" << endl;
                cout << "Results size: " << y.size() << ". Print results? n/y. " << endl;
                cin >> ans;
                if (tolower(ans) == 'y')
                    printSequence(y);

                break;
            }
            case 2: {
                cout << "Calculating convolution... ";
                //Convolution of sample_audio.wav and pink_noise.wav
                auto now = chrono::steady_clock::now();
                auto new_samples = conv.MyConvolve(audio_sample, audio_noise);
                auto end = chrono::steady_clock::now();
                cout << "Done!" << endl;
                cout << "Time elapsed: ";
                cout << chrono::duration_cast<chrono::milliseconds>(end - now).count() << " ms" << endl;

                //Convert the convolution results from valarray to vector matrix.
                vector<vector<double>> buffer = {vector<double>(new_samples.size())};
                for (int i = 0; i < new_samples.size(); ++i)
                    buffer[0][i] = new_samples[i];

                //Create pinkNoise_sampleAudio.wav with sample rate equal to sample_audio.wav
                audio_new.setAudioBuffer(buffer);
                audio_new.save("../pinkNoise_sampleAudio.wav");
                cout << "File ../pinkNoise_sampleAudio.wav created." << endl;
                audio_new.printSummary();
                break;
            }
            case 3: {
                //Create white noise
                valarray<double> val_white(audio_sample.samples[0].size());
                for (int i = 0; i < val_white.size(); ++i) {
                    val_white[i] = getNormalRandom();
                }

                //Convert to valarray.
                valarray<double> val_sample(audio_sample.samples[0].size());
                for (int i = 0; i < val_sample.size(); ++i)
                    val_sample[i] = audio_sample.samples[0][i];
                cout << "Calculating convolution... ";
                auto now = chrono::steady_clock::now();
                auto new_samples = conv.MyConvolve(val_sample, val_white);
                auto end = chrono::steady_clock::now();
                cout << "Done!" << endl;
                cout << "Time elapsed: ";
                cout << chrono::duration_cast<chrono::milliseconds>(end - now).count() << " ms" << endl;
                //Creating whiteNoise_sampleAudio.wav
                vector<vector<double>> buffer = {vector<double>(new_samples.size())};
                for (int i = 0; i < new_samples.size(); ++i)
                    buffer[0][i] = new_samples[i];
                audio_new.setAudioBuffer(buffer);
                audio_new.save("../whiteNoise_sampleAudio.wav");
                cout << "File ../whiteNoise_sampleAudio.wav created." << endl;
                audio_new.printSummary();
                break;
            }
            default:
                return 0;
        }
        cout << "||~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~||" << endl;
    }

}
