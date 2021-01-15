// This program computes the sum of two vectors of length N
// Adapted from https://github.com/CoffeeBeforeArch/cuda_programming
//
// Author: Robbie Watling

# include <cuda.h>
# include <cuda_runtime_api.h>
# include <iostream>
# include <vector>

using namespace std;

int main() {
    //Initialize an array of size 2^16
    int n = 1 << 16;
    size_t bytes = sizeof(int) * n;

    //Host-side vectors
    vector <int> a;
    vector <int> b;
    vector <int> c;

    //Device arrays
    int* dev_a;
    int* dev_b;
    int* dev_c;

    //Initialize host-side array
    for (int i = 0; i < n; i++) {
        a.push_back(rand() % 100);
        b.push_back(rand() % 100);
    }

    //Allocate memory on the device
    cudaMalloc((void**) &dev_a, bytes);
    cudaMalloc((void**) &dev_b, bytes);
    cudaMalloc((void**) &dev_c, bytes);

    cudaMemcpy(dev_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b.data(), bytes, cudaMemcpyHostToDevice);

    // Threads per CTA (1024)
    int num_threads = 1 << 10;

    // CTAs per Grid
    // We need to launch at LEAST as many threads as we have elements
    // This equation pads an extra CTA to the grid if N cannot evenly be divided
    // by NUM_THREADS (e.g. N = 1025, NUM_THREADS = 1024)
    int num_blocks = (n + num_threads - 1) / num_threads;

    cout << "Number of array elements (n): " << n << endl;
    cout << "Number of threads: " << num_threads << endl;
    cout << "Number of blocks: " << num_blocks << endl;

    return 0;
}