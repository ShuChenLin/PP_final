#include <iostream>
#include <vector>
#include <cmath>
#include <sstream>
#include <fstream>
#include <sys/time.h>

using namespace std;

__device__ bool flag = true;

typedef struct point {
    float x;
    float y;
    int cluster;
} Point;

__host__ __device__ float distance(Point p1, Point p2) {
    return sqrt((p1.x - p2.x) * (p1.x - p2.x) + 
                (p1.y - p2.y) * (p1.y - p2.y));
}

__host__ __device__ float dis_square(Point p1, Point p2) {
    return (p1.x - p2.x) * (p1.x - p2.x) + 
           (p1.y - p2.y) * (p1.y - p2.y);
}

void initcentroid(int k, int d, Point *centroid, Point *data) {
    // k-means++ algorithm
    
    // 1. Choose one center uniformly at random from among the data points.
    int index = rand() % d;
    centroid[0].x = data[index].x;
    centroid[0].y = data[index].y;

    // 2. For each data point x, compute D(x), the distance between x and the nearest center that has already been chosen.
    for (int i = 1; i < k; ++i) {
        float sum = 0.0;
        float *dist = new float[d];
        for (int j = 0; j < d; ++j) {
            float mx = 1e9;
            for (int l = 0; l < i; ++l) {
                mx = min(mx, dis_square(centroid[l], data[j]));
            }
            dist[j] = mx;
            sum += dist[j];
        }

        // 3. Choose one new data point at random as a new center, using a weighted probability distribution where a point x is chosen with probability proportional to D(x)^2.
        float r = (float)rand() / RAND_MAX * sum;
        int cumulative = 0;
        for (int j = 0; j < d; ++j) {
            cumulative += dist[j];
            if (cumulative >= r) {
                centroid[i].x = data[j].x;
                centroid[i].y = data[j].y;
                break;
            }
        }
    }

}

void initCluster(int k, int d, Point *centroid, Point *data) {
    for (int i = 0; i < d; ++i) {
        float min_dist = 1e9;
        for (int j = 0; j < k; ++j) {
            float dist = distance(data[i], centroid[j]);
            if (dist < min_dist) {
                min_dist = dist;
                data[i].cluster = j;
            }
        }
    }
}

void updateCentroid(int k, int d, Point *centroid, Point *data) {
    int *cnt = new int[k];

    for (int i = 0; i < k; ++i) {
        cnt[i] = 0;
        centroid[i].x = 0.0;
        centroid[i].y = 0.0;
    }
    for (int i = 0; i < d; ++i) {
        cnt[data[i].cluster]++;
        centroid[data[i].cluster].x += data[i].x;
        centroid[data[i].cluster].y += data[i].y;
    }
    for (int i = 0; i < k; ++i) {
        centroid[i].x /= cnt[i];
        centroid[i].y /= cnt[i];
    }
}

__global__ void KmeansGPU(int k, int d, Point *centroid, Point *data) {
    int i = blockIdx.x * blockDim.x + threadIdx.y;
    int j = blockIdx.y * blockDim.y + threadIdx.x;

    int index = i * blockDim.x * gridDim.x + j;

    while (flag) {
        flag = false;
        __syncthreads();
        if (index < d) {
            float min_dist = 1e9;
            int min_cluster = data[index].cluster;
            for (int l = 0; l < k; ++l) {
                float dist = distance(data[index], centroid[l]);
                if (dist < min_dist) {
                    min_dist = dist;
                    min_cluster = l;
                }
            }
            if (min_cluster != data[index].cluster) {
                flag = true;
                data[index].cluster = min_cluster;
            }
        }
        __syncthreads();

        if (flag) {
            if (index < k) {
                centroid[index].x = 0.0;
                centroid[index].y = 0.0;
            }
            __syncthreads();

            if (index < d) {
                atomicAdd(&centroid[data[index].cluster].x, data[index].x);
                atomicAdd(&centroid[data[index].cluster].y, data[index].y);
            }
            __syncthreads();

            if (index < k) {
                centroid[index].x /= d;
                centroid[index].y /= d;
            }
            __syncthreads();
        }
    }
}

int main () {
    struct timeval start, end, Begin, Finish;
    double total, memcpy_time;

    gettimeofday(&Begin, NULL);

    ifstream fin("testcases/kdata1500000");
    string line;

    gettimeofday(&start, NULL);

    Point *data = new Point[1500000];

    int d = 0;
    while (getline(fin, line)) {
        stringstream ss(line);
        ss >> data[d].x >> data[d].y;
        d++;
    }

    gettimeofday(&end, NULL);
    cout << "Read data time: " << 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec << "us" << endl;


    // k-means algorithm
    int k = 10;
    Point *centroid = new Point[k];

    gettimeofday(&start, NULL);

    initcentroid(k, d, centroid, data);
    initCluster(k, d, centroid, data);

    gettimeofday(&end, NULL);

    total = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;

    dim3 block(32, 32);
    dim3 grid((d + block.x - 1) / block.x, (d + block.y - 1) / block.y);

    cudaSetDevice(0);
    cudaHostRegister(data, d * sizeof(Point), cudaHostRegisterDefault);
    cudaHostRegister(centroid, k * sizeof(Point), cudaHostRegisterDefault);
    

    Point *d_data, *d_centroid;
    gettimeofday(&start, NULL);
    cudaMalloc((void **)&d_data, d * sizeof(Point));
    cudaMalloc((void **)&d_centroid, k * sizeof(Point));

    cudaMemcpyAsync(d_data, data, d * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_centroid, centroid, k * sizeof(Point), cudaMemcpyHostToDevice);
    gettimeofday(&end, NULL);
    memcpy_time = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    gettimeofday(&start, NULL);

    KmeansGPU<<<grid, block>>>(k, d, d_centroid, d_data);
    gettimeofday(&end, NULL);

    total += 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;

    gettimeofday(&start, NULL);

    cudaMemcpyAsync(data, d_data, d * sizeof(Point), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(centroid, d_centroid, k * sizeof(Point), cudaMemcpyDeviceToHost);

    gettimeofday(&end, NULL);
    memcpy_time += 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;

    //cudaFree(d_data);


    // print result
    // for (int i = 0; i < k; ++i) {
        // cout << "Centroid " << i << ": (" << centroid[i].x << ", " << centroid[i].y << ")" << endl;
    // }

    gettimeofday(&start, NULL);

    // output data into kdata.out
    ofstream fout("kdata.out");
    fout << "x y cluster" << endl;
    for (int i = 0; i < d; ++i) {
        fout << data[i].x << " " << data[i].y << " " << data[i].cluster << endl;
    }

    gettimeofday(&end, NULL);

    cout << "Write data time: " << 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec << "us" << endl;

    gettimeofday(&Finish, NULL);


    total /= 1000000;
    memcpy_time /= 1000000;
    double final_time = 1000000 * (Finish.tv_sec - Begin.tv_sec) + Finish.tv_usec - Begin.tv_usec;
    final_time /= 1000000;
    cout << "Compute time: " << total << "s" << endl;
    cout << "Memcpy time: " << memcpy_time << "s" << endl;
    cout << "Total time: " << final_time << "s" << endl;


    return 0;
}