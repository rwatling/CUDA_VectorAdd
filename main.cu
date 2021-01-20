// This program computes the sum of two vectors of length N
// Adapted from https://github.com/CoffeeBeforeArch/cuda_programming
//
// Author: Robbie Watling

# include "system_includes.h"

using namespace std;

// CUDA kernel for vector addition
// __global__ means this is called from the CPU, and runs on the GPU
__global__ void vector_add(int* a, int* b, int* c, int n) {
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
    //Performance variables
    clock_t time_start;
    clock_t time_end;
    double elapsed;
    ofstream my_file;
    string run_file_name;


    run_file_name = "vector_add_performance.txt";
    my_file.open(run_file_name);

    for (int j = 0; j < 20; j++) {
        //Initialize clock (in ticks)
        time_start = clock();
        
        //Initialize an array
        int N = 1 << 14;
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
        srand(time(NULL));
        for (int i = 0; i < N; i++) {
            a.push_back(rand() % 100);
            b.push_back(rand() % 100);
            c.push_back(0);
        }

        //Allocate memory on the device
        cudaMalloc((void**) &dev_a, bytes);
        cudaMalloc((void**) &dev_b, bytes);
        cudaMalloc((void**) &dev_c, bytes);

        //Zero out the data for more accurate performance measures
        cudaMemset((void*) dev_a, 0, bytes);
        cudaMemset((void*) dev_b, 0, bytes);
        cudaMemset((void*) dev_c, 0, bytes);

        //Copy host arrays to device arrays
        cudaMemcpy(dev_a, a.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b, b.data(), bytes, cudaMemcpyHostToDevice);

        // Threads per CTA (standard is 1024)
        int NUM_THREADS = 1 << 10;

        // CTAs per Grid
        // We need to launch at LEAST as many threads as we have elements
        // This equation pads an extra CTA to the grid if N cannot evenly be divided
        // by NUM_THREADS (e.g. N = 1025, NUM_THREADS = 1024)
        int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

        // Kernel calls are asynchronous (the CPU program continues execution after
        // call, but no necessarily before the kernel finishes)
        vector_add << <NUM_BLOCKS, NUM_THREADS >> > (dev_a, dev_b, dev_c, N);

        // Copy sum vector from device to host
        // cudaMemcpy is a synchronous operation, and waits for the prior kernel
        // launch to complete (both go to the default stream in this case).
        // Therefore, this cudaMemcpy acts as both a memcpy and synchronization
        // barrier.
        cudaMemcpy(c.data(), dev_c, bytes, cudaMemcpyDeviceToHost);

        // Check result for errors
        verify_result(a, b, c);

        // Free memory on device
        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);

        //Stop clock
        time_end = clock();
        elapsed = (double) (time_end - time_start) / CLOCKS_PER_SEC;

        //Inform the user that the run has been successful
        cout << "COMPLETED SUCCESSFULLY\n";

        //Performance information printed to txt file
        my_file << "Number of array elements (n): " << N << endl;
        my_file << "Number of threads: " << NUM_THREADS << endl;
        my_file << "Number of blocks: " << NUM_BLOCKS << endl;
        my_file << "Time elapsed: " << elapsed << " seconds" << endl;
        my_file << "---------------------------------------------" << endl;
    }

    my_file.close();

    return 0;
}