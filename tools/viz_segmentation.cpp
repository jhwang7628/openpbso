#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <set>
#include "Eigen/Dense"
#include "igl/read_triangle_mesh.h"
#include "igl/opengl/glfw/Viewer.h"
int main(int argc, char **argv) {
    if (argc != 3) {
        std::cout << "**Usage: " << argv[0] << " <obj> <segmentation>\n";
        return 1;
    }
    Eigen::MatrixXd V, C;
    Eigen::MatrixXi F;
    igl::read_triangle_mesh(argv[1], V, F);
    std::vector<int> seg(V.rows());
    std::ifstream stream(argv[2]);
    std::string line;
    for (int ii=0; ii<V.rows(); ++ii) {
        std::getline(stream, line);
        std::istringstream iss(line);
        iss >> seg[ii];
    }
    std::set<int> segset(seg.begin(), seg.end());
    const int N_seg = segset.size();
    Eigen::MatrixXd palette = Eigen::MatrixXd::Random(N_seg, 3);
    C.resize(V.rows(), 3);
    for (int ii=0; ii<V.rows(); ++ii) {
        auto it = segset.find(seg[ii]);
        int id = std::distance(segset.begin(), it);
        for (int jj=0; jj<3; ++jj) {
            C(ii, jj) = abs(palette(id, jj));
        }
    }

    igl::opengl::glfw::Viewer viewer;
    viewer.data().set_mesh(V, F);
    viewer.data().set_colors(C);
    viewer.launch();
}
