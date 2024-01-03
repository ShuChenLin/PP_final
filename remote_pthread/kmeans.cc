#include <iostream>
#include <vector>
#include <cmath>
#include <sstream>
#include <fstream>
#include <sys/time.h>

using namespace std;

typedef struct point {
    float x;
    float y;
    int cluster;
} Point;

float distance(Point p1, Point p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

float dis_square(Point p1, Point p2) {
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

void Kmeans(int k, int d, Point *centroid, Point *data) {
    // 1. Randomly choose k data points as initial centroids.
    initcentroid(k, d, centroid, data);

    initCluster(k, d, centroid, data);

    updateCentroid(k, d, centroid, data);

    // 2. Repeat steps 2 and 3 until convergence or until the end of a fixed number of iterations.
    bool flag = true;
    while (flag) {
        flag = false;
        for (int i = 0; i < d; ++i) {
            float min_dist = 1e9;
            int min_cluster = data[i].cluster;
            for (int j = 0; j < k; ++j) {
                float dist = distance(data[i], centroid[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    min_cluster = j;
                }
            }
            if (min_cluster != data[i].cluster) {
                flag = true;
                data[i].cluster = min_cluster;
            }
        }

        if (flag) updateCentroid(k, d, centroid, data);

    }
}

int main () {
    // read data, each row is a sample

    ifstream fin("testcases/kdata300000");
    string line;

    Point *data = new Point[300000];

    int d = 0;
    while (getline(fin, line)) {
        stringstream ss(line);
        ss >> data[d].x >> data[d].y;
        d++;
    }

    // k-means algorithm
    int k = 10;
    Point *centroid = new Point[k];

    Kmeans(k, d, centroid, data);

    // print result
    for (int i = 0; i < k; ++i) {
        cout << "Centroid " << i << ": (" << centroid[i].x << ", " << centroid[i].y << ")" << endl;
    }

    // output data into kdata.out
    ofstream fout("kdata.out");
    fout << "x y cluster" << endl;
    for (int i = 0; i < d; ++i) {
        fout << data[i].x << " " << data[i].y << " " << data[i].cluster << endl;
    }


    return 0;
}