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
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

__host__ __device__ float dis_square(Point p1, Point p2) {
    return pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2);
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
    // read data, each row is a sample

    struct timeval start, end;

    ifstream fin("kdata");
    string line;

    Point *data = new Point[300000];

    int d = 0;
    while (getline(fin, line)) {
        stringstream ss(line);
        ss >> data[d].x >> data[d].y;
        d++;
    }

    gettimeofday(&start, NULL);

    // k-means algorithm
    int k = 10;
    Point *centroid = new Point[k];

    initcentroid(k, d, centroid, data);
    initCluster(k, d, centroid, data);

    dim3 block(32, 32);
    dim3 grid((d + block.x - 1) / block.x, (d + block.y - 1) / block.y);

    Point *d_data, *d_centroid;
    cudaMalloc((void **)&d_data, d * sizeof(Point));
    cudaMalloc((void **)&d_centroid, k * sizeof(Point));

    cudaMemcpy(d_data, data, d * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroid, centroid, k * sizeof(Point), cudaMemcpyHostToDevice);

    KmeansGPU<<<grid, block>>>(k, d, d_centroid, d_data);

    cudaMemcpy(data, d_data, d * sizeof(Point), cudaMemcpyDeviceToHost);
    cudaMemcpy(centroid, d_centroid, k * sizeof(Point), cudaMemcpyDeviceToHost);

    cudaFree(d_data);

    gettimeofday(&end, NULL);

    // print result
    // for (int i = 0; i < k; ++i) {
        // cout << "Centroid " << i << ": (" << centroid[i].x << ", " << centroid[i].y << ")" << endl;
    // }

    // output data into kdata.out
    ofstream fout("kdata.out");
    fout << "x y cluster" << endl;
    for (int i = 0; i < d; ++i) {
        fout << data[i].x << " " << data[i].y << " " << data[i].cluster << endl;
    }

    double timeuse = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    timeuse /= 1000000;
    cout << "Time: " << timeuse << "s" << endl;


    return 0;
}