#include <fstream>
#include <dirent.h>
#include <memory>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include "igl/read_triangle_mesh.h"
#include "igl/opengl/glfw/Viewer.h"
#include "igl/opengl/glfw/imgui/ImGuiMenu.h"
#include "igl/opengl/glfw/imgui/ImGuiHelpers.h"
#include "igl/per_vertex_normals.h"
#include "igl/colormap.h"
#include "igl/png/writePNG.h"
#include "ModeData.h"
//##############################################################################
bool IsFile(const char *path) {
    struct stat buffer;
    return (stat (path, &buffer) == 0);
}
//##############################################################################
void ListDirFiles(const char *dirname, std::vector<std::string> &names) {
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(dirname)) != NULL) {
        /* print all the files and directories within directory */
        while ((ent = readdir (dir)) != NULL) {
            std::string f = dirname + std::string("/") + std::string(ent->d_name);
            if (IsFile(f.c_str()) && (ent->d_name[0] != '.'))
                names.push_back(f);
        }
        closedir (dir);
    } else {
        /* could not open directory */
        perror ("");
    }
}
//##############################################################################
// Ref: https://github.com/libigl/libigl/issues/862
//##############################################################################
class CapturePlugin : public igl::opengl::glfw::ViewerPlugin {
public:
    CapturePlugin() { plugin_name = "capture"; }

    bool post_draw() override {
        if (!capturing) {
            return false;
        }

        const int width  = viewer->core().viewport(2);
        const int height = viewer->core().viewport(3);

        std::unique_ptr<GLubyte[]> pixels(new GLubyte[width * height * 4]);

        glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.get());

        char buf[512];
        snprintf(buf, 512, "%s-%0.4d.png", pathPrefix.c_str(), captureIdx++);
        //std::string path = pathPrefix + std::to_string(captureIdx++) + ".png";
        std::string path(buf);
        writePNG(path, std::move(pixels), width, height);
        //std::thread{ writePNG, path, std::move(pixels), width, height }.detach();

        return false;
    }

    void startCapture(std::string capturePath) {
        pathPrefix = capturePath;
        captureIdx = 0;
        capturing  = true;
    }

    void stopCapture() { capturing = false; }

    bool isCapturing() const { return capturing; }

private:
    static void writePNG(std::string path, std::unique_ptr<GLubyte[]> pixels, int width, int height) {
        igl::stbi_write_png(path.c_str(), width, height, 4, pixels.get() + width * (height - 1) * 4, -width * 4);
    };

    std::string pathPrefix;
    int     captureIdx;
    bool        capturing = false;
};
//##############################################################################
struct Param {
    std::vector<int> dims = {424, 424, 88};
    double timeStepSize = 1./116360.;
    double density = 1070.;
    std::vector<double> center = {0.024603, 0.042155, -0.025120};
    double cellSize = 0.005106;
    bool show_lines = false;

    float obj_vmin = -1;
    float obj_vmax = 1;
    float plane_vmin = -1;
    float plane_vmax = 1;
} PARAMS;
//##############################################################################
struct Plane {
    Eigen::Matrix<double,-1,-1,Eigen::RowMajor> data;
    Eigen::MatrixXd Z;
    Eigen::MatrixXd C;
    Eigen::Vector3i dims;
    int N;
    double dx;
    Eigen::Vector3d lowCorner;
    Plane() {
        dims << PARAMS.dims[0], PARAMS.dims[1], PARAMS.dims[2];
        N = dims[0]*dims[1];
        dx = PARAMS.cellSize;
        data.resize(dims[0], dims[1]);
        lowCorner <<
            PARAMS.center.at(0) - dims[0]/2.*dx,
            PARAMS.center.at(1) - dims[1]/2.*dx,
            PARAMS.center.at(2) - dims[2]/2.*dx;
        Z.resize(dims[0]*dims[1]*2, 1);
        C.resize(dims[0]*dims[1]*2, 3);
    }
    void Load(const std::string &filename) {
        //data = Eigen::MatrixXd::Random(data.rows(), data.cols());
        std::ifstream stream(filename.c_str(), std::ios::binary);
        Eigen::MatrixXf data_f;
        data_f.resize(data.rows(), data.cols());
        if (stream) {
            stream.read((char*)data_f.data(), sizeof(float)*N);
        }
        data = data_f.cast<double>();
    }
    void SetColor(
        const std::string &filename, igl::opengl::ViewerData &viewerData) {
        Load(filename);
        assert(data.array().size() == N && "data dimension incorrect");
        for (int ii=0; ii<N; ++ii) {
            Z(ii*2+0) = data.array()(ii);
            Z(ii*2+1) = data.array()(ii);
        }
        igl::colormap(igl::COLOR_MAP_TYPE_PARULA, Z, PARAMS.plane_vmin, PARAMS.plane_vmax, C);
        viewerData.set_colors(C);
    }
    int AddToViewer(igl::opengl::glfw::Viewer &viewer) {
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        V.resize((dims[0]+1)*(dims[1]+1), 3);
        F.resize(N*2, 3);
        for (int ii=0; ii<=dims[0]; ++ii) {
            for (int jj=0; jj<=dims[1]; ++jj) {
                V.row(ii*(dims[1]+1)+jj) <<
                    lowCorner[0] + ii*dx,
                    lowCorner[1] + jj*dx,
                    lowCorner[2] + dims[2]/2*dx;
                if (ii!=dims[0] && jj!=dims[1]) {
                    F.row(2*(ii*dims[1]+jj)) <<
                        ii*(dims[1]+1)+jj,
                        (ii+1)*(dims[1]+1)+jj,
                        ii*(dims[1]+1)+(jj+1);
                    F.row(2*(ii*dims[1]+jj)+1) <<
                        (ii+1)*(dims[1]+1)+jj,
                        (ii+1)*(dims[1]+1)+jj+1,
                        ii*(dims[1]+1)+(jj+1);
                }
            }
        }
        int idx = viewer.append_mesh();
        viewer.data(idx).set_mesh(V, F);
        viewer.data(idx).show_lines = PARAMS.show_lines;
        return idx;
    }
};
//##############################################################################
int main(int argc, char **argv) {
    if (argc < 5) {
        std::cerr << "**Usage: " << argv[0] << " <obj_file> <mod_file> <data_dir> <draw_mode_file>\n";
        return 1;
    }
    std::string obj_file(argv[1]);
    Eigen::MatrixXd V, Z, C, U, VN;
    Eigen::MatrixXi F;
    igl::read_triangle_mesh(obj_file, V, F);
    igl::per_vertex_normals(V, F, VN);
    Z.resize(V.rows(), 1);
    C.resize(V.rows(), 3);

    ModeData<double> *modes = new ModeData<double>();
    modes->read(argv[2]);
    U.resize(modes->numDOF()/3, modes->numModes());
    for (int ii=0; ii<modes->numModes(); ++ii) {
        std::vector<double> &U_i = modes->mode(ii);
        for (int jj=0; jj<modes->numDOF()/3; ++jj) {
            U(jj,ii) = U_i.at(jj*3 + 0) * VN(jj, 0)
                     + U_i.at(jj*3 + 1) * VN(jj, 1)
                     + U_i.at(jj*3 + 2) * VN(jj, 2);
        }
        U.col(ii) /= U.col(ii).maxCoeff(); // NOTE: necessary?
    }
    assert(V.rows() == U.rows() && "Number of vertices inconsistent");

    std::vector<std::string> filenames;
    ListDirFiles(argv[3], filenames);
    auto ParseID = [](const std::string &a) {
        const auto p = a.find_last_of("-");
        return std::stoi(a.substr(p+1, a.find_last_of(".")-p-1));
    };
    std::sort(filenames.begin(), filenames.end(), [&](
        const std::string &a, const std::string &b) {
            return ParseID(a) < ParseID(b);
    });
    for (const auto &f : filenames) {
        std::cout << "Read file: " << f << std::endl;
    }
    Plane plane;

    igl::opengl::glfw::Viewer viewer;
    const int obj_idx = viewer.data().id;
    viewer.data().set_mesh(V, F);
    viewer.core().is_animating = false;
    viewer.core().animation_max_fps = 30.;
    viewer.data().show_lines = PARAMS.show_lines;
    const int plane_idx = plane.AddToViewer(viewer);
    double time = 0.0;
    int timeIdx = 0;
    std::vector<int> drawModes;
    {
        std::ifstream stream(argv[4]);
        if (stream) {
            std::string line;
            std::getline(stream, line);
            std::istringstream iss(line);
            int buf;
            while (iss >> buf) {
                drawModes.push_back(buf);
            }
        }
    }
    std::cout << "drawing following modes: ";
    for (auto b : drawModes) {
        std::cout << b << " ";
    }
    std::cout << std::endl;

    viewer.callback_pre_draw =
        [&](igl::opengl::glfw::Viewer &viewer) {
            if (!viewer.core().is_animating)
                return false;
            Z.setZero();
            for (int drawMode : drawModes) {
                const double omega =
                    std::sqrt(modes->omegaSquared(drawMode)/PARAMS.density);
                Z += U.col(drawMode) * cos(omega*time);
            }
            igl::colormap(igl::COLOR_MAP_TYPE_PARULA, Z,
                PARAMS.obj_vmin, PARAMS.obj_vmax, C);
            viewer.data(obj_idx).set_colors(C);

            plane.SetColor(filenames.at(timeIdx%filenames.size()),
                viewer.data(plane_idx));

            timeIdx += 1;
            time = timeIdx * PARAMS.timeStepSize;
            return false;
        };
    viewer.core().background_color << 0,0,0,1;

    CapturePlugin capture;
    viewer.plugins.push_back(&capture);

    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);
    menu.callback_draw_viewer_menu = [&]() {
        menu.draw_viewer_menu();
        ImGui::Begin("Mine");
        ImGui::InputFloat("obj vmin", &PARAMS.obj_vmin);
        ImGui::InputFloat("obj vmax", &PARAMS.obj_vmax);
        ImGui::InputFloat("plane vmin", &PARAMS.plane_vmin);
        ImGui::InputFloat("plane vmax", &PARAMS.plane_vmax);
        if (!capture.isCapturing()) {
            if (ImGui::Button("Start capture", ImVec2(-1, 0))) {
                std::string capturePath = igl::file_dialog_save();
                if (!capturePath.empty()) {
                    capture.startCapture(capturePath);
                }
            }
        } else {
            if (ImGui::Button("Stop capture", ImVec2(-1, 0))) {
                capture.stopCapture();
            }
        }
        ImGui::End();
    };

    viewer.launch();

    return 0;
}
