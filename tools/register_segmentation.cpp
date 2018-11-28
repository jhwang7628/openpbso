#include <iostream>
#include <fstream>
#include "Eigen/Dense"
#include "igl/read_triangle_mesh.h"
int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "**Usage: " << argv[0] << " <obj> <seg_obj> <outfile>\n";
        return 1;
    }
    // use nearest neighbor to find seg face id
    Eigen::MatrixXd V_o, V_m;
    Eigen::MatrixXi F_o, F_m;
    igl::read_triangle_mesh(argv[1], V_o, F_o);
    igl::read_triangle_mesh(argv[2], V_m, F_m);
    std::ofstream stream(argv[3]);
    std::vector<int> map(V_o.rows());
    Eigen::Vector3d p1, p2;
    double bestDist;
    int bestInd;
    std::cout << "matching starts \n";
    for (int ii=0; ii<V_o.rows(); ++ii) {
        if (ii % 50 == 0)
            std::cout << "ii = " << ii << std::endl;
        p1 = V_o.row(ii);
        bestDist = std::numeric_limits<double>::max();
        for (int jj=0; jj<F_m.rows(); ++jj) {
            p2.setZero();
            p2 += V_m.row(F_m(jj,0));
            p2 += V_m.row(F_m(jj,1));
            p2 += V_m.row(F_m(jj,2));
            p2 /= 3.0;
            if ((p1 - p2).squaredNorm() < bestDist) {
                bestInd = jj;
                bestDist = (p1-p2).squaredNorm();
            }
        }
        map[ii] = bestInd;
        stream << bestInd << "\n";
    }
    stream.close();

    return 0;
}
