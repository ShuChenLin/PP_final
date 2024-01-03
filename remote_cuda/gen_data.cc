#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>

struct Point {
    double x, y;
};

// Function to generate a random point within a circle
Point random_point_in_circle(double radius) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist_radius(0, radius);
    std::uniform_real_distribution<> dist_angle(0, 2 * M_PI);

    double r = sqrt(dist_radius(gen));  // sqrt for uniform distribution
    double theta = dist_angle(gen);

    return {r * cos(theta), r * sin(theta)};
}

int main() {
    const int n_points = 3000000;  // Total number of points
    const double radius = 5000.0;  // Radius of the circle

    std::vector<Point> points;

    // Generate points uniformly within the circle
    for (int i = 0; i < n_points; ++i) {
        points.push_back(random_point_in_circle(radius));
    }

    // Saving the points to a file
    std::ofstream file("kdata3000000");
    for (const auto& point : points) {
        file << point.x << " " << point.y << "\n";
    }
    file.close();

    return 0;
}
