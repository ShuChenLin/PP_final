#include <iostream>
#include <vector>
#include <cmath>
#include <sstream>
#include <fstream>
#include <sys/time.h>
#include <pthread.h>

using namespace std;

typedef struct point {
    float x;
    float y;
    int cluster;
} Point;

int flag = 0, ncpus;
pthread_barrier_t barrier;
pthread_mutex_t mutex;

typedef struct arg {
    int k;
    int d;
    int data_st, data_ed;
    int centroid_st, centroid_ed;
    Point *centroid;
    Point *data;
} Arg;

float distance(Point p1, Point p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

float dis_square(Point p1, Point p2) {
    return pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2);
}

void* KmeansPthread(void* arg) {
    Arg *args = (Arg*)arg;
    int k = args->k, d = args->d, data_st = args->data_st, data_ed = args->data_ed, centroid_st = args->centroid_st, centroid_ed = args->centroid_ed;
    Point *centroid = args->centroid, *data = args->data;

    // init cluster
    for (int i = data_st; i < data_ed; ++i) {
        float min_dis = 1e9;
        for (int j = 0; j < k; ++j) {
            float dis = distance(data[i], centroid[j]);
            if (dis < min_dis) {
                min_dis = dis;
                data[i].cluster = j;
            }
        }
    }
    pthread_barrier_wait(&barrier);

    if (centroid_st != -1) {
        for (int i = centroid_st; i < centroid_ed; ++i) {
            centroid[i].x = 0.0;
            centroid[i].y = 0.0;
            centroid[i].cluster = 0;
        }
    }

    pthread_barrier_wait(&barrier);

    for (int i = data_st; i < data_ed; ++i) {
        pthread_mutex_lock(&mutex);
        centroid[data[i].cluster].x += data[i].x;
        centroid[data[i].cluster].y += data[i].y;
        centroid[data[i].cluster].cluster++;
        pthread_mutex_unlock(&mutex);
    }

    pthread_barrier_wait(&barrier);

    if (centroid_st != -1) {
        for (int i = centroid_st; i < centroid_ed; ++i) {
            centroid[i].x /= centroid[i].cluster;  
            centroid[i].y /= centroid[i].cluster;
        }
    }

    pthread_barrier_wait(&barrier);

    for (int i = 0; i < k; ++i) {
        cout << "Centroid " << i << ": (" << centroid[i].x << ", " << centroid[i].y << ")" << endl;
    }

    while (!flag) {

        int local_flag = 0;

        for (int i = data_st; i < data_ed; ++i) {
            float min_dis = 1e9;
            int min_cluster = data[i].cluster;
            for (int j = 0; j < k; ++j) {
                float dis = distance(data[i], centroid[j]);
                if (dis < min_dis) {
                    min_dis = dis;
                    min_cluster = j;
                }
            }
            if (min_cluster != data[i].cluster) {
                // use mutex
                local_flag = 1;
                data[i].cluster = min_cluster;
            }
        }

        pthread_barrier_wait(&barrier);

        pthread_mutex_lock(&mutex);
        flag += local_flag;
        pthread_mutex_unlock(&mutex);

        pthread_barrier_wait(&barrier);

        if (centroid_st == 0 && flag < ncpus) {
            flag = 0;
        }

        if (!flag) {
            if (centroid_st != -1) {
                for (int i = centroid_st; i < centroid_ed; ++i) {
                    centroid[i].x = 0.0;
                    centroid[i].y = 0.0;
                    centroid[i].cluster = 0;
                }
            }

            pthread_barrier_wait(&barrier);

            for (int i = data_st; i < data_ed; ++i) {
                pthread_mutex_lock(&mutex);
                centroid[data[i].cluster].x += data[i].x;
                centroid[data[i].cluster].y += data[i].y;
                centroid[data[i].cluster].cluster++;
                pthread_mutex_unlock(&mutex);
            }

            pthread_barrier_wait(&barrier);

            if (centroid_st != -1) {
                for (int i = centroid_st; i < centroid_ed; ++i) {
                    centroid[i].x /= centroid[i].cluster;  
                    centroid[i].y /= centroid[i].cluster;
                }
            } 
        }

        pthread_barrier_wait(&barrier);

    }

    return NULL;
}

void initcentroid(int k, int d, Point *centroid, Point *data) {
    // k-means++ algorithm
    
    // 1. Choose one center uniformly at random from among the data points.
    int index = rand() % d;
    centroid[0].x = data[index].x;
    centroid[0].y = data[index].y;
    centroid[0].cluster = 0;

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
                centroid[i].cluster = 0;
                break;
            }
        }
    }

}

int main () {
    // read data, each row is a sample

    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    ncpus = CPU_COUNT(&cpu_set);

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

    initcentroid(k, d, centroid, data);

    pthread_t *threads = new pthread_t[ncpus];
    pthread_mutex_init(&mutex, NULL);
    pthread_barrier_init(&barrier, NULL, ncpus);
    int size = d / ncpus, remain = d % ncpus;
    int cen_size = k / ncpus, cen_remain = k % ncpus;

    for (int i = 0; i < ncpus; ++i) {
        Arg *arg = new Arg;
        arg->k = k;
        arg->d = d;
        arg->data_st = i * size + min(i, remain);
        arg->data_ed = (i + 1) * size + min(i + 1, remain);
        arg->centroid_st = ((i < k) ? (i * cen_size + min(i, cen_remain)) : -1);
        arg->centroid_ed = (i + 1) * cen_size + min(i + 1, cen_remain);
        arg->centroid = centroid;
        arg->data = data;
        //cout << arg->data_st << " " << arg->data_ed << " " << arg->centroid_st << " " << arg->centroid_ed << endl;
        pthread_create(&threads[i], NULL, KmeansPthread, (void*)arg);
    }

    for (int i = 0; i < ncpus; ++i) {
        pthread_join(threads[i], NULL);
    }

    pthread_mutex_destroy(&mutex);

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