// This program computes the sum of two vectors of length N
// Adapted from https://github.com/CoffeeBeforeArch/cuda_programming
//
// Author: Robbie Watling

# include <cuda.h>
# include <cuda_runtime_api.h>
# include <device_launch_parameters.h>
# include <iostream>
# include <vector>
# include <cassert>
# include <time.h>

using namespace std;

// CUDA kernel for vector addition
// __global__ means this is called from the CPU, and runs on the GPU
__global__ void vector_add(int* a, int* __restrict b, int* c, int n) {
    // Calculate global thread ID
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    // Boundary check
    if (tid < n) c[tid] = a[tid] + b[tid];
}

// Check vector add result
void verify_result(std::vector<int>& a, std::vector<int>& b, std::vector<int>& c) {
    for (int i = 0; i < a.size(); i++) {
        assert(c[i] == a[i] + b[i]);
    }
}

int main() {
    //Initialize an array
    int N = 1 << 6;
    size_t bytes = sizeof(int) * N;

    //Host-side vectors
    vector <int> a;
    vector <int> b;
    vector <int> c;

    //Device arrays
    int* dev_a;
    int* dev_b;
    int* dev_c;

    //Initialize host-side array
    srand(time(0));
    for (int i = 0; i < N; i++) {
        a.push_back(rand() % 100);
        b.push_back(rand() % 100);
        c.push_back(0);
    }

    //Allocate memory on the device
    cudaMalloc((void**) &dev_a, bytes);
    cudaMalloc((void**) &dev_b, bytes);
    cudaMalloc((void**) &dev_c, bytes);

    cudaMemcpy(dev_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b.data(), bytes, cudaMemcpyHostToDevice);

    // Threads per CTA
    int NUM_THREADS = 1 << 3;

    // CTAs per Grid
    // We need to launch at LEAST as many threads as we have elements
    // This equation pads an extra CTA to the grid if N cannot evenly be divided
    // by NUM_THREADS (e.g. N = 1025, NUM_THREADS = 1024)
    int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

    cout << "Number of array elements (n): " << N << endl;
    cout << "Number of threads: " << NUM_THREADS << endl;
    cout << "Number of blocks: " << NUM_BLOCKS << endl;

    // Kernel calls are asynchronous (the CPU program continues execution after
    // call, but no necessarily before the kernel finishes)
    vector_add<<<NUM_BLOCKS, NUM_THREADS>>>(dev_a, dev_b, dev_c, N);

    // Copy sum vector from device to host
    // cudaMemcpy is a synchronous operation, and waits for the prior kernel
    // launch to complete (both go to the default stream in this case).
    // Therefore, this cudaMemcpy acts as both a memcpy and synchronization
    // barrier.
    cudaMemcpy(c.data(), dev_c, bytes, cudaMemcpyDeviceToHost);

    // Check result for errors

    //Print version
    /*
    cout << "a = [ ";
    for (int i = 0; i < a.size(); i++) {
        cout << a[i] << " ";
    }
    cout << "]" << endl;

    cout << "b = [ ";
    for (int i = 0; i < b.size(); i++) {
        cout << b[i] << " ";
    }
    cout << "]" << endl;

    cout << "c = [ ";
    for (int i = 0; i < a.size(); i++) {
        cout << c[i] << " ";
    }
    cout << "]" << endl;
    */

    //Unit test version
    verify_result(a, b, c);

    // Free memory on device
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    cout << "COMPLETED SUCCESSFULLY\n";

    return 0;
}