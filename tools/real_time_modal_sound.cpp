#include <set>
#include <ctime>
#include <thread>
#include <pthread.h>
#include <string>
#include <deque>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include "igl/read_triangle_mesh.h"
#include "igl/unproject_onto_mesh.h"
#include "igl/per_vertex_normals.h"
#include "igl/opengl/glfw/Viewer.h"
#include "igl/opengl/glfw/imgui/ImGuiMenu.h"
#include "igl/opengl/glfw/imgui/ImGuiHelpers.h"
#include "igl/opengl/create_shader_program.h"
#include "igl/opengl/destroy_shader_program.h"
#include "igl/file_dialog_open.h"
#include "igl/colormap.h"
#include "igl/png/readPNG.h"
#include "imgui/imgui.h"
#include "config.h"
#include "ModalMaterial.h"
#include "ModeData.h"
#include "cmd_parser.h"
#include "portaudio.h"
#include "modal_integrator.h"
#include "modal_solver.h"
#include "forces.h"
#include "matcap_shader.h"
//##############################################################################
static bool PA_STREAM_STARTED = false;
//##############################################################################
#define CHECK_PA_LAUNCH(x) \
    { \
        auto pa_err = x; \
        if (pa_err != paNoError) \
            printf("**ERROR** PortAudio error: %s\n", Pa_GetErrorText(pa_err));\
    }
//##############################################################################
cli::Parser *CreateParser(int argc, char **argv) {
    cli::Parser *parser = new cli::Parser(argc, argv);
    parser->set_required<std::string>("m", "mesh",
        "Triangle mesh for the object");
    parser->set_required<std::string>("s", "surf_mode",
        "surface modes file");
    parser->set_required<std::string>("t", "material",
        "modal material file");
    parser->set_required<std::string>("p", "ffat_map",
        "ffat map folder that contains *.fatcube files");
    parser->set_optional<std::string>("tex", "obj_texture_map", FILE_NOT_EXIST,
        "texture map used in matcaps shader program");
    parser->run_and_exit_if_error();
    return parser;
}
//##############################################################################
struct ViewerSettings {
    float volume = 1.0;
    bool useTransfer = true;
    bool useTransferCache = true;
    float bufferHealth[100] = {1.0f};
    int bufferHealthPtr = 0;
    static float renderFaceTime;
    std::deque<std::pair<int,std::chrono::time_point<
        std::chrono::high_resolution_clock>>> activeFaceIds;
    ForceMessage<double> hitForceCache;
    int hitFidCache = 0;
    int hitVidCache = 0;
    bool newHit = false;
    float transferBallNormalization = 0.45;
    float transferBallBufThres = 200.;
    bool sustainedForceActive = false;
    bool useTextures = false;
    bool drawModes = false;
    int drawModeIdx = 0;
    void incrementBufferHealthPtr() {
        bufferHealthPtr = (bufferHealthPtr+1)%100;
    }
    int force_type_int=0;
    int old_force_type_int=0;
    static ForceType forceType;
    struct GaussianForceParameters {
        float timeScale;
    } gaussianForceParameters;
    struct ARForceParameters {
        Eigen::Vector3d past_surf_pos;
        Eigen::Vector3d curr_surf_pos;
        Eigen::Vector3d curr_surf_vel;
        bool past_mouse_init = false;
        double time_step_size = 1./60.;
        float mu_f = 0.142;
        float a_f[2] = {0.783, 0.116};
        float sigma_f = 0.00148;
        AutoregressiveForceParam<double> arprm;
    } arForceParameters;

    // loading new model stuff
    bool loadingNewModel = false;
    std::condition_variable cvLoadingNewModel;
    std::mutex mutexLoadingNewModel;
    bool terminated = false;
    void Reinitialize() {
        activeFaceIds.clear();
        hitForceCache = ForceMessage<double>();
        hitFidCache = 0;
        hitVidCache = 0;
        newHit = false;
        arForceParameters.past_mouse_init = false;
    }
} VIEWER_SETTINGS;
struct ModalViewer {
    bool animate = false;
    Eigen::MatrixXd V0;
    Eigen::MatrixXi F0;
    Eigen::MatrixXd V;
    Eigen::MatrixXd Z;
    Eigen::VectorXd Zv;
    Eigen::MatrixXd C;
    float scale = 1.E-6;
    float scale_exp = -6.0f;
    float time = 0.;
    int ind = -1;
    float zoom = 1.4f;
    void UpdateModeShape(
        const std::unique_ptr<ModeData<double>> &modes,
        const int ind) {
        if (this->ind != ind) {
            const auto &mode = modes->mode(ind);
            Z.resize(mode.size()/3, 3);
            Zv.resize(mode.size()/3);
            for (int ii=0; ii<mode.size()/3; ++ii) {
                Z(ii,0) = mode.at(ii*3    );
                Z(ii,1) = mode.at(ii*3 + 1);
                Z(ii,2) = mode.at(ii*3 + 2);
                Zv(ii) = Z.row(ii).norm();
            }
            this->ind = ind;
        }
    }
} MODAL_VIEWER;
float ViewerSettings::renderFaceTime = 1.5f;
ForceType ViewerSettings::forceType = ForceType::PointForce;
//##############################################################################
Eigen::Matrix<double,3,1> getCameraWorldPosition(
    igl::opengl::ViewerCore &core) {
    Eigen::Matrix<float,4,1> eye;
    eye << core.camera_eye, 1.0f;
    const Eigen::Matrix<double,3,1> camera_pos =
        (core.view.inverse() * eye).cast<double>().head(3);
    return camera_pos;
}
//##############################################################################
bool CurrentMouseSurfPos(
    igl::opengl::glfw::Viewer &viewer,
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    Eigen::Vector3d &pos, Eigen::Vector3d &coord, int &fid, int &vid) {
    const double x = viewer.current_mouse_x;
    const double y = viewer.core().viewport(3) - viewer.current_mouse_y;
    if(igl::unproject_onto_mesh(
        Eigen::Vector2f(x,y), viewer.core().view,
        viewer.core().proj, viewer.core().viewport, V, F, fid, coord)) {
        vid = F(fid, 0);
        float lar = coord[0];
        if (coord[1] > coord[0]) {
            vid = F(fid, 1);
            lar = coord[1];
        }
        if (coord[2] > lar) vid = F(fid, 2);
        pos = V.row(F(fid,0))*coord[0]
            + V.row(F(fid,1))*coord[1]
            + V.row(F(fid,2))*coord[2];
        return true;
    }
    return false;
}
//##############################################################################
struct PaModalData {
    std::unique_ptr<ModalSolver<double>> *solver;
    SoundMessage<double> soundMessage;
};
//##############################################################################
int PaModalCallback(const void *inputBuffer,
                    void *outputBuffer,
                    unsigned long framesPerBuffer,
                    const PaStreamCallbackTimeInfo* timeInfo,
                    PaStreamCallbackFlags statusFlags,
                    void *userData) {
    /* Cast data passed through stream to our structure. */
    PaModalData *data = (PaModalData*)userData;
    float *out = (float*)outputBuffer;
    unsigned int i;
    (void) inputBuffer; /* Prevent unused variable warning. */
    bool success = (*(data->solver))->dequeueSoundMessage(data->soundMessage);
    VIEWER_SETTINGS.bufferHealth[VIEWER_SETTINGS.bufferHealthPtr] =
        (float)success;
    VIEWER_SETTINGS.incrementBufferHealthPtr();
    for( i=0; i<framesPerBuffer; i++ ) {
        *out++ = (float)(data->soundMessage.data(i)/1E10);
        *out++ = (float)(data->soundMessage.data(i)/1E10);
    }
    return 0;
}
//##############################################################################
template<typename T>
void GetModalForceCopy(
    const ForceMessage<T> &cache,
    ForceMessage<T> &force) {
    // perform a deep copy
    force = cache;
    force.forceType = VIEWER_SETTINGS.forceType;
    if (force.forceType == ForceType::PointForce) {
        force.force.reset(new PointForce<T>());
    }
    else if (force.forceType == ForceType::GaussianForce) {
        force.force.reset(new GaussianForce<T>(
                    VIEWER_SETTINGS.gaussianForceParameters.timeScale));
    }
    else if (force.forceType == ForceType::AutoregressiveForce) {
        force.force.reset(new AutoregressiveForce<T>());
    }
    else {
        assert(false && "Force type not defined");
    }
}
//##############################################################################
template<typename T>
void GetModalForceFace(
    const int forceDim,
    const ModeData<T> &modes,
    const Eigen::Vector3i vids,
    const Eigen::Vector3d coords,
    const Eigen::Vector3d &vn, // NOTE: using the same vn for all three vids
    ForceMessage<double> &force) {
    force.data.setZero(forceDim);
    for (int mm=0; mm<forceDim; ++mm) {
        for (int jj=0; jj<3; ++jj) {
            force.data(mm) += vn[0]*modes.mode(mm).at(vids[jj]*3+0)*coords[jj]
                            + vn[1]*modes.mode(mm).at(vids[jj]*3+1)*coords[jj]
                            + vn[2]*modes.mode(mm).at(vids[jj]*3+2)*coords[jj];
        }
    }
    force.forceType = VIEWER_SETTINGS.forceType;
    if (force.forceType == ForceType::PointForce) {
        force.force.reset(new PointForce<T>());
    }
    else if (force.forceType == ForceType::GaussianForce) {
        force.force.reset(new GaussianForce<T>(
                    VIEWER_SETTINGS.gaussianForceParameters.timeScale));
    }
    else if (force.forceType == ForceType::AutoregressiveForce) {
        force.force.reset(new AutoregressiveForce<T>());
    }
    else {
        assert(false && "Force type not defined");
    }
}
//##############################################################################
template<typename T>
void GetModalForceVertex(
    const int forceDim,
    const ModeData<T> &modes,
    const int vid,
    const Eigen::Vector3d &vn,
    ForceMessage<double> &force) {
    force.data.setZero(forceDim);
    for (int mm=0; mm<forceDim; ++mm) {
        force.data(mm) = vn[0]*modes.mode(mm).at(vid*3+0)
            + vn[1]*modes.mode(mm).at(vid*3+1)
            + vn[2]*modes.mode(mm).at(vid*3+2);
    }
    force.forceType = VIEWER_SETTINGS.forceType;
    if (force.forceType == ForceType::PointForce) {
        force.force.reset(new PointForce<T>());
    }
    else if (force.forceType == ForceType::GaussianForce) {
        force.force.reset(new GaussianForce<T>(
                    VIEWER_SETTINGS.gaussianForceParameters.timeScale));
    }
    else if (force.forceType == ForceType::AutoregressiveForce) {
        force.force.reset(new AutoregressiveForce<T>());
    }
    else {
        assert(false && "Force type not defined");
    }
}
//##############################################################################
template<typename T>
ModalMaterial<T> *ReadMaterial(const char *filename) {
    return ModalMaterial<T>::Read(filename);
}
//##############################################################################
template<typename T>
ModeData<T> *ReadModes(const char *filename) {
    ModeData<T> *modes = new ModeData<T>();
    modes->read(filename);
    return modes;
}
//##############################################################################
template<typename T>
ModalSolver<T> *BuildSolver(
    const std::unique_ptr<ModalMaterial<T>> &material,
    const std::unique_ptr<ModeData<T>> &modes,
    const std::string &ffatMapFolder,
    int &N_modesAudible) {
    // read max frequency to cull modes
    std::string maxFreqFile =
        ffatMapFolder + "/freq_threshold.txt";
    std::ifstream stream(maxFreqFile.c_str());
    N_modesAudible = modes->numModes();
    if (stream) {
        std::string line;
        std::getline(stream, line);
        std::istringstream iss(line);
        double maxFreq;
        iss >> maxFreq;
        N_modesAudible = modes->numModesAudible(material->density, maxFreq);
    } else { // set default frequency to 20kHz
        N_modesAudible = modes->numModesAudible(material->density, 20000.);
    }
    // build integrator and then set it for solver
    ModalSolver<T> *solver = new ModalSolver<T>(N_modesAudible);
    std::shared_ptr<ModalIntegrator<double>> integrator(
        ModalIntegrator<double>::Build(
            material->density,
            modes->_omegaSquared,
            material->alpha,
            material->beta,
            1./(double)SAMPLE_RATE,
            N_modesAudible
        )
    );
    solver->setIntegrator(integrator);
    solver->readFFATMaps(ffatMapFolder);
    return solver;
}
//##############################################################################
void LoadNewModel(
    std::string &obj_file,
    std::string &mod_file,
    std::string &mat_file,
    std::string &fat_path,
    Eigen::MatrixXd &V,
    Eigen::MatrixXd &VN,
    Eigen::MatrixXi &F,
    Eigen::MatrixXd &transfer_ball,
    Eigen::MatrixXd &V_ball,
    Eigen::MatrixXd &C_ball,
    Eigen::VectorXd &transferVals,
    Eigen::Vector3d &pos,
    int &obj_id,
    int &sph_id,
    int &mod_id,
    unsigned int &main_view,
    unsigned int &hud_view,
    int &N_modesAudible,
    cli::Parser *parser,
    Eigen::Matrix<unsigned char,-1,-1> &tex_R,
    Eigen::Matrix<unsigned char,-1,-1> &tex_G,
    Eigen::Matrix<unsigned char,-1,-1> &tex_B,
    Eigen::Matrix<unsigned char,-1,-1> &tex_A,
    igl::opengl::glfw::Viewer &viewer,
    std::unique_ptr<ModalMaterial<double>> &material,
    std::unique_ptr<ModeData<double>> &modes,
    std::unique_ptr<ModalSolver<double>> &solver) {
    std::string meta = igl::file_dialog_open();
    if (meta.length() != 0) {
        if (meta.substr(meta.size()-4, 4) == ".png") {
            const size_t last_slash = meta.find_last_of('/');
            if (last_slash == std::string::npos || last_slash == meta.size()-1) {
                return;
            }
            std::string prefix = meta.substr(0, last_slash);
            std::string suffix = meta.substr(last_slash+1, meta.size());
            std::cout << "prefix, suffix = " << prefix << " " << suffix << std::endl;
            prefix = prefix.substr(0, prefix.find_last_of('/'));
            meta = prefix + "/" + suffix.substr(0, suffix.size()-4) + ".meta";
        }
        std::cout << "loading new model " << meta << " ..." << std::flush;
        std::ifstream stream(meta.c_str());
        if (!stream) return;
        std::string obj_file_tmp;
        std::string mod_file_tmp;
        std::string mat_file_tmp;
        std::string fat_path_tmp;
        std::getline(stream, obj_file_tmp);
        std::getline(stream, mod_file_tmp);
        std::getline(stream, mat_file_tmp);
        std::getline(stream, fat_path_tmp);
        if (Gpu_Wavesolver::IsFile(obj_file.c_str()) &&
            Gpu_Wavesolver::IsFile(mod_file.c_str()) &&
            Gpu_Wavesolver::IsFile(mat_file.c_str()) &&
            Gpu_Wavesolver::IsFile(fat_path.c_str())) {
            obj_file = obj_file_tmp;
            mod_file = mod_file_tmp;
            mat_file = mat_file_tmp;
            fat_path = fat_path_tmp;
            std::unique_lock<std::mutex> lockLoad(
                VIEWER_SETTINGS.mutexLoadingNewModel);
            VIEWER_SETTINGS.loadingNewModel = true;
            igl::read_triangle_mesh(obj_file.c_str(), V, F);
            viewer.data(obj_id).clear();
            viewer.append_mesh();
            viewer.core(main_view).align_camera_center(V);
            viewer.core(hud_view).align_camera_center(V);
            obj_id = viewer.data().id;
            viewer.data(obj_id).clear();
            viewer.data(mod_id).clear();
            viewer.data(obj_id).set_mesh(V, F);
            viewer.data(mod_id).set_mesh(V, F);
            MODAL_VIEWER.V0 = V;
            MODAL_VIEWER.F0 = F;
            MODAL_VIEWER.ind = -1;
            if (VIEWER_SETTINGS.useTextures) {
                igl::png::readPNG(
                        parser->get<std::string>("tex"),
                        tex_R,tex_G,tex_B,tex_A);
                viewer.data(obj_id).set_texture(
                        tex_R,tex_G,tex_B,tex_A);
                viewer.data(obj_id).set_face_based(false);
                viewer.data(obj_id).show_lines = false;
                viewer.data(obj_id).show_texture = true;
                viewer.data(obj_id).meshgl.init();
                igl::opengl::destroy_shader_program(
                        viewer.data(obj_id).meshgl.shader_mesh);
                igl::opengl::create_shader_program(
                        mesh_vertex_shader_string,
                        mesh_fragment_shader_string,
                        {},
                        viewer.data(obj_id).meshgl.shader_mesh);
            }
            else {
                viewer.data(obj_id).show_lines = false;
                viewer.data(obj_id).set_face_based(true);
            }
            igl::per_vertex_normals(V, F, VN);
            material.reset(ReadMaterial<double>(mat_file.c_str()));
            modes.reset(ReadModes<double>(mod_file.c_str()));
            solver.reset(
                    BuildSolver(
                        material,
                        modes,
                        fat_path,
                        N_modesAudible
                        )
                    );
            assert(modes->numDOF() == V.rows()*3 && "DOFs mismatch");
            VIEWER_SETTINGS.Reinitialize();
            transfer_ball.setZero(N_modesAudible, V_ball.rows());
            for (int ii=0; ii<V_ball.rows(); ++ii) {
                pos = V_ball.row(ii);
                solver->computeTransfer(pos,
                        transfer_ball.data()+ii*N_modesAudible);
            }
            transfer_ball /= transfer_ball.maxCoeff();
            transferVals = Eigen::VectorXd::Ones(transferVals.size())*0.1;
            igl::colormap(
                    igl::COLOR_MAP_TYPE_JET,transferVals,true,C_ball);
            viewer.data(sph_id).set_colors(C_ball);
            VIEWER_SETTINGS.loadingNewModel = false;
            VIEWER_SETTINGS.cvLoadingNewModel.notify_all();
            std::cout << " OK\n" << std::flush;
        }
    }
}
//##############################################################################
//##############################################################################
int main(int argc, char **argv) {
    auto *parser = CreateParser(argc, argv);
    std::string obj_file(parser->get<std::string>("m"));
    std::string mod_file(parser->get<std::string>("s"));
    std::string mat_file(parser->get<std::string>("t"));
    std::string fat_path(parser->get<std::string>("p"));
    // read geometry
    Eigen::MatrixXd V, C, VN, V_ball, C_ball, transfer_ball;
    Eigen::MatrixXi F, F_ball;
    Eigen::Matrix<unsigned char,-1,-1> tex_R,tex_G,tex_B,tex_A;
    Eigen::VectorXd transferVals;
    Eigen::Vector3d pos;
    igl::read_triangle_mesh(obj_file, V, F);
    igl::per_vertex_normals(V, F, VN);
    // read modal materials and mode shapes
    std::unique_ptr<ModalMaterial<double>> material(ReadMaterial<double>(
        mat_file.c_str()));
    std::unique_ptr<ModeData<double>> modes(
        ReadModes<double>(mod_file.c_str()));
    assert(modes->numDOF() == V.rows()*3 && "DOFs mismatch");
    // build modal integrator and solver/scheduler
    int N_modesAudible;
    std::unique_ptr<ModalSolver<double>> solver(
        BuildSolver(
            material,
            modes,
            fat_path,
            N_modesAudible
        )
    );
    // start a simulation thread and use max priority
    std::thread threadSim([&](){
        while (!VIEWER_SETTINGS.terminated) {
            solver->step();
            std::unique_lock<std::mutex> lockLoad(
                VIEWER_SETTINGS.mutexLoadingNewModel);
            while (VIEWER_SETTINGS.loadingNewModel) {
                VIEWER_SETTINGS.cvLoadingNewModel.wait(lockLoad);
            }
        }
    });
    sched_param sch_params;
    sch_params.sched_priority = sched_get_priority_max(SCHED_FIFO);
    pthread_setschedparam(threadSim.native_handle(), SCHED_FIFO, &sch_params);

    // setup audio callback stuff
    CHECK_PA_LAUNCH(Pa_Initialize());
    PaModalData paData;
    paData.solver = &solver;
    PaStream *stream;
    CHECK_PA_LAUNCH(Pa_OpenDefaultStream(&stream,
        0,                 /* no input channels */
        2,                 /* stereo output */
        paFloat32,         /* 32 bit floating point output */
        SAMPLE_RATE,       /* audio sampling rate */
        FRAMES_PER_BUFFER, /* frames per buffer */
        PaModalCallback,   /* audio callback function */
        &paData ));        /* audio callback data */

    // Plot the mesh
    igl::opengl::glfw::Viewer viewer;
    viewer.core().is_animating = true;
    unsigned int main_view, hud_view, mode_view;
    int obj_id, sph_id, mod_id;
    viewer.callback_init = [&](
            igl::opengl::glfw::Viewer &viewer) {
        // initialize parameters in VIEWER_SETTINGS
        VIEWER_SETTINGS.hitForceCache.data.setZero(N_modesAudible);
        viewer.core().viewport = Eigen::Vector4f(0, 0, 1280, 800);
        viewer.append_core(Eigen::Vector4f(800, 0, 480, 480));
        viewer.append_core(Eigen::Vector4f(880, 200, 600, 600));
        main_view = viewer.core_list[0].id;
        hud_view = viewer.core_list[1].id;
        mode_view = viewer.core_list[2].id;
        viewer.core(hud_view).update_transform_matrices = false;
        viewer.core(mode_view).update_transform_matrices = false;
        obj_id = viewer.data_list[0].id;
        sph_id = viewer.data_list[1].id;
        mod_id = viewer.data_list[2].id;
        viewer.data(mod_id).set_visible(false,main_view);
        viewer.data(mod_id).set_visible(false, hud_view);
        viewer.data(mod_id).set_visible(true, mode_view);
        viewer.data(sph_id).set_visible(false,main_view);
        viewer.data(sph_id).set_visible(false,mode_view);
        viewer.data(sph_id).set_visible(true, hud_view);
        viewer.data(obj_id).set_visible(false, hud_view);
        viewer.data(obj_id).set_visible(false,mode_view);
        viewer.data(obj_id).set_visible(true, main_view);
        return false;
    };
    viewer.callback_post_resize = [&](
            igl::opengl::glfw::Viewer &v, int w, int h) {
        v.core(main_view).viewport = Eigen::Vector4f(0, 0, w, h);
        v.core(hud_view).viewport = Eigen::Vector4f(w-480, 0, 480, 480);
        v.core(mode_view).viewport = Eigen::Vector4f(w-600, h-600, 600, 600);
        return true;
    };
    C = Eigen::MatrixXd::Constant(F.rows(),3,1);
    viewer.callback_mouse_down =
        [&](
            igl::opengl::glfw::Viewer& viewer, int, int modifier)->bool {
            int fid, vid;
            Eigen::Vector3f bc;
            Eigen::Vector3d vn, pos, coord;
            // Cast a ray in the view direction starting from the mouse
            // position
            if (CurrentMouseSurfPos(viewer, V, F, pos, coord, fid, vid)) {
                VIEWER_SETTINGS.arForceParameters.curr_surf_pos = pos;
                if (modifier == GLFW_MOD_SHIFT) {
                    VIEWER_SETTINGS.activeFaceIds.push_back(
                            {fid,std::chrono::high_resolution_clock::now()});
                    vn = VN.row(vid).normalized();
                    ForceMessage<double> force;
                    GetModalForceVertex(N_modesAudible, *modes, vid, vn, force);
                    solver->enqueueForceMessage(force);
                    VIEWER_SETTINGS.hitForceCache = force;
                    VIEWER_SETTINGS.hitFidCache = fid;
                    if (VIEWER_SETTINGS.hitVidCache != vid) {
                        VIEWER_SETTINGS.newHit = true;
                    }
                    VIEWER_SETTINGS.hitVidCache = vid;
                    viewer.data().set_colors(C);
                    return true;
                }
            }
            return false;
        };
    viewer.callback_mouse_up = [&](
        igl::opengl::glfw::Viewer &viewer, int, int)->bool {
        auto &ar_parm = VIEWER_SETTINGS.arForceParameters;
        ar_parm.curr_surf_vel.setZero();
        ForceMessage<double> force;
        GetModalForceVertex(N_modesAudible, *modes, 0,
            ar_parm.curr_surf_vel, force);
        solver->enqueueForceMessageNoFail(force);
        return false;
    };
    viewer.callback_mouse_move = [&](
        igl::opengl::glfw::Viewer &viewer, int, int)->bool {
        if (!viewer.down) {
            VIEWER_SETTINGS.arForceParameters.past_mouse_init = false;
        }
        if (VIEWER_SETTINGS.sustainedForceActive) {
            return true;
        }
        return false;
    };

    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);
    menu.callback_draw_viewer_menu = [&]()
    {
        // viewer menu
        //menu.draw_viewer_menu();
        // simulation menu
        ImGui::SetNextWindowPos(ImVec2(0,0));
        ImGui::SetNextWindowSize(ImVec2(250,0));
        ImGui::Begin("Simulation");
        auto extractLast = [](const std::string str, const std::string delim) {
            return str.substr(str.rfind(delim)+1);
        };
        if (ImGui::Button("Load Model")) {
            LoadNewModel(
                obj_file,
                mod_file,
                mat_file,
                fat_path,
                V,
                VN,
                F,
                transfer_ball,
                V_ball,
                C_ball,
                transferVals,
                pos,
                obj_id,
                sph_id,
                mod_id,
                main_view,
                hud_view,
                N_modesAudible,
                parser,
                tex_R,
                tex_G,
                tex_B,
                tex_A,
                viewer,
                material,
                modes,
                solver);
        }
        if (ImGui::CollapsingHeader("Info", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::TreeNode("Geometry")) {
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.8f,0.8f,0.8f,1.0f));
                ImGui::Columns(2, "mycolumns2", false);
                ImGui::Text("Mesh file"); ImGui::NextColumn();
                ImGui::Text("%s", extractLast(obj_file, "/").c_str()); ImGui::NextColumn();
                ImGui::Text("#Vertices"); ImGui::NextColumn();
                ImGui::Text("%d", (int)V.rows()); ImGui::NextColumn();
                ImGui::Text("#Triangles"); ImGui::NextColumn();
                ImGui::Text("%d", (int)F.rows()); ImGui::NextColumn();
                ImGui::Columns(1);
                ImGui::TreePop();
                ImGui::PopStyleColor();
            }
            if (ImGui::TreeNode("Modal Info")) {
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.8f,0.8f,0.8f,1.0f));
                ImGui::Columns(2, "mycolumns2", false);
                ImGui::Text("Modes file"); ImGui::NextColumn();
                ImGui::Text("%s", extractLast(mod_file, "/").c_str()); ImGui::NextColumn();
                ImGui::Text("#Modes"); ImGui::NextColumn();
                ImGui::Text("%d", modes->numModes()); ImGui::NextColumn();
                ImGui::Text("#Audible modes"); ImGui::NextColumn();
                ImGui::Text("%d", N_modesAudible); ImGui::NextColumn();
                ImGui::Text("Material file"); ImGui::NextColumn();
                ImGui::Text("%s", extractLast(mat_file, "/").c_str()); ImGui::NextColumn();
                ImGui::Text("Density"); ImGui::NextColumn();
                ImGui::Text("%.2f", material->density); ImGui::NextColumn();
                ImGui::Text("Young's"); ImGui::NextColumn();
                ImGui::Text("%.2e", material->youngsModulus); ImGui::NextColumn();
                ImGui::Text("Poisson ratio"); ImGui::NextColumn();
                ImGui::Text("%.2f", material->poissonRatio); ImGui::NextColumn();
                ImGui::Text("alpha"); ImGui::NextColumn();
                ImGui::Text("%.2f", material->alpha); ImGui::NextColumn();
                ImGui::Text("beta"); ImGui::NextColumn();
                ImGui::Text("%.2f", material->beta); ImGui::NextColumn();
                ImGui::Columns(1);
                ImGui::TreePop();
                ImGui::PopStyleColor();
            }
        }
		if (ImGui::CollapsingHeader(
            "Sound Model", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Text("Impact Force Control:");
            ImGui::BeginChild(
                "option",
                ImVec2(220, 220),
                true);
            if (ImGui::Button("Repeat hit")) {
                ForceMessage<double> force;
                GetModalForceCopy(VIEWER_SETTINGS.hitForceCache, force);
                solver->enqueueForceMessage(force);
                VIEWER_SETTINGS.activeFaceIds.push_back(
                    {VIEWER_SETTINGS.hitFidCache,
                    std::chrono::high_resolution_clock::now()});
            }
            ImGui::SameLine();
            if (ImGui::Button("Clear force")) {
                if (!VIEWER_SETTINGS.sustainedForceActive) {
                    ForceMessage<double> force;
                    force.clearAllForces = true;
                    solver->enqueueForceMessageNoFail(force);
                }
            }
            ImGui::RadioButton("Point force", &VIEWER_SETTINGS.force_type_int, 0);
            ImGui::RadioButton("Gaussian force", &VIEWER_SETTINGS.force_type_int, 1);
            ImGui::RadioButton("Autoregressive force", &VIEWER_SETTINGS.force_type_int, 2);
            VIEWER_SETTINGS.forceType = static_cast<ForceType>(VIEWER_SETTINGS.force_type_int);
            if (VIEWER_SETTINGS.force_type_int == 2) {
                if (!VIEWER_SETTINGS.sustainedForceActive) {
                    VIEWER_SETTINGS.sustainedForceActive = true;
                    // send a dummy start signal
                    ForceMessage<double> force;
                    force.sustainedForceStart = true;
                    GetModalForceVertex(
                        N_modesAudible, *modes, 0, Eigen::Vector3d::Zero(), force);
                    solver->enqueueForceMessageNoFail(force);
                }
            } else {
                if (VIEWER_SETTINGS.old_force_type_int == 2 &&
                    VIEWER_SETTINGS.sustainedForceActive) {
                    // send a dummy end signal
                    ForceMessage<double> force;
                    force.sustainedForceEnd = true;
                    GetModalForceVertex(
                        N_modesAudible, *modes, 0, Eigen::Vector3d::Zero(),
                        force);
                    solver->enqueueForceMessageNoFail(force);
                }
                VIEWER_SETTINGS.sustainedForceActive = false;
            }
            VIEWER_SETTINGS.old_force_type_int = VIEWER_SETTINGS.force_type_int;
            // gaussian force parameters
            if (VIEWER_SETTINGS.forceType != ForceType::GaussianForce) {
                ImGui::PushStyleVar(
                    ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
            }
            ImGui::SliderFloat("width",
                &VIEWER_SETTINGS.gaussianForceParameters.timeScale,
                10.0f, 500.0f, "%.1f (us)");
            static int width_samples = 0;
            width_samples = std::max(
                1,
                (int)(
                    VIEWER_SETTINGS.gaussianForceParameters.timeScale
                    /1000000.*SAMPLE_RATE));
            ImGui::SameLine(); ImGui::Text("(%d)", width_samples);
            if (VIEWER_SETTINGS.forceType != ForceType::GaussianForce) {
                ImGui::PopStyleVar();
            }
            if (VIEWER_SETTINGS.forceType != ForceType::AutoregressiveForce) {
                ImGui::PushStyleVar(
                    ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
            }
            auto &ar = VIEWER_SETTINGS.arForceParameters;
            ImGui::DragFloat("mu", &ar.mu_f, 0.1, 0.0, 1);
            ImGui::DragFloat("a1", &ar.a_f[0], 0.1, 0.0, 1.0);
            ImGui::DragFloat("a2", &ar.a_f[1], 0.1, 0.0, 1.0);
            ImGui::DragFloat("sigma", &ar.sigma_f, 0.1, 0.0, 1.0);
            if (ar.arprm.mu != (double)ar.mu_f ||
                ar.arprm.a[0] != (double)ar.a_f[0] ||
                ar.arprm.a[1] != (double)ar.a_f[1] ||
                ar.arprm.sigma != (double)ar.sigma_f) {
                ar.arprm.mu = (double)ar.mu_f;
                ar.arprm.a = {(double)ar.a_f[0], (double)ar.a_f[1]};
                ar.arprm.sigma = (double)ar.sigma_f;
                solver->enqueueArprmMessageNoFail(ar.arprm);
            }
            if (VIEWER_SETTINGS.forceType != ForceType::AutoregressiveForce) {
                ImGui::PopStyleVar();
            }
            ImGui::EndChild();
            ImGui::BeginChild(
                "health",
                ImVec2(220, 75),
                true);
            ImGui::Text("Audio buffer health status:");
            ImGui::PlotLines(
                "",
                &(VIEWER_SETTINGS.bufferHealth[0]),
                IM_ARRAYSIZE(VIEWER_SETTINGS.bufferHealth),
                VIEWER_SETTINGS.bufferHealthPtr,
                nullptr,
                0.0f, 1.0f,
                ImVec2(200, 40));
            ImGui::EndChild();
            const auto transfer = solver->getLatestTransfer();
            const Eigen::Matrix<float,-1,1> hist = transfer.data.cast<float>()/
                transfer.data.array().abs().maxCoeff();
            ImGui::BeginChild(
                "histogram",
                ImVec2(220, 160),
                true);
            ImGui::Checkbox(
                "Enable FFAT transfer", &VIEWER_SETTINGS.useTransfer);
            if (VIEWER_SETTINGS.useTransfer !=
                VIEWER_SETTINGS.useTransferCache) {
                solver->setUseTransfer(VIEWER_SETTINGS.useTransfer);
                solver->computeTransfer(
                    getCameraWorldPosition(viewer.core(main_view)));
                VIEWER_SETTINGS.useTransferCache = VIEWER_SETTINGS.useTransfer;
            }
            ImGui::Text("Transfer values for different modes:");
            ImGui::PlotHistogram("",
                hist.data(),
                solver->getLatestTransfer().data.size(),
                0, nullptr, 0.0f, 1.0f, ImVec2(200, 80));
            ImGui::EndChild();
		}
		if (ImGui::CollapsingHeader(
            "Visualization", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::BeginChild(
                "TransferVis",
                ImVec2(220, 80),
                true);
            auto &mv = MODAL_VIEWER;
            if (ImGui::Button("Draw Modes")) {
                VIEWER_SETTINGS.drawModes = !VIEWER_SETTINGS.drawModes;
            }
            ImGui::DragInt("Mode Idx", &(VIEWER_SETTINGS.drawModeIdx), 1.0f,
                0, N_modesAudible);
            if (ImGui::DragFloat("Scale Exponent", &(mv.scale_exp),0.2,-9,-3)) {
                mv.scale = pow(10, mv.scale_exp);
            }
            ImGui::DragFloat("Zoom", &(mv.zoom),0.1,0.1,10.0,0);
            ImGui::EndChild();
            ImGui::BeginChild(
                "TransferVis",
                ImVec2(220, 80),
                true);
            ImGui::Text("\'Ball\' visualization parameters:");
            ImGui::DragFloat("norma val",
                    &VIEWER_SETTINGS.transferBallNormalization,
                    0.01, 0.001, 10.0);
            ImGui::DragFloat("thres val",
                    &VIEWER_SETTINGS.transferBallBufThres,
                    1., 1., 10000.);
            ImGui::EndChild();
        }
        ImGui::End();
    };

    viewer.data().set_mesh(V, F);
    viewer.data().set_colors(C);
    viewer.data().show_lines = false;
    viewer.data().set_face_based(true);

    // preparing for the HUD
    // FIXME debug: this won't work outside the build directory.
    // should probably hard code the ball or use cmake to communicate the
    // base dir
    igl::read_triangle_mesh("../assets/ball_higher.obj", V_ball, F_ball);
    viewer.append_mesh();
    viewer.data().set_mesh(V_ball, F_ball);
    viewer.data().show_lines = false;
    viewer.data().set_face_based(true);
    viewer.data().set_colors(Eigen::RowVector3d(0,0,1));
    viewer.data().shininess = 500.0f;

    // preparing for the modal viewer draw
    viewer.append_mesh();
    MODAL_VIEWER.V0 = V;
    MODAL_VIEWER.F0 = F;

    viewer.data().set_mesh(V, F);
    viewer.data().show_lines = false;
    viewer.data().set_face_based(true);
    viewer.data().set_colors(Eigen::RowVector3d(0,1,0));

    viewer.selected_data_index = 0;
    // get transfer values on the ball
    transferVals.setOnes(V_ball.rows());
    transferVals *= 0.1;
    igl::colormap(
            igl::COLOR_MAP_TYPE_JET,transferVals,true,C_ball);
    transfer_ball.setZero(N_modesAudible, V_ball.rows());
    for (int ii=0; ii<V_ball.rows(); ++ii) {
        pos = V_ball.row(ii);
        solver->computeTransfer(pos,
            transfer_ball.data()+ii*N_modesAudible);
    }
    transfer_ball /= transfer_ball.maxCoeff();
    viewer.callback_pre_draw =
        [&](igl::opengl::glfw::Viewer &viewer) {
            if (VIEWER_SETTINGS.loadingNewModel) {
                return false;
            }
            auto &queue = VIEWER_SETTINGS.activeFaceIds;
            while (!queue.empty()) {
                const auto front = queue.front();
                const std::chrono::duration<float> diff =
                    std::chrono::high_resolution_clock::now() - front.second;
                if (diff.count() > ViewerSettings::renderFaceTime) {
                    C.row(front.first)<<1,1,1;
                    queue.pop_front();
                } else {
                    break;
                }
            }
            C = Eigen::MatrixXd::Constant(F.rows(),3,1);
            for (auto it=queue.begin(); it!=queue.end(); ++it) {
                const std::chrono::duration<float> diff =
                    std::chrono::high_resolution_clock::now() - it->second;
                if (diff.count() > ViewerSettings::renderFaceTime) {
                    C.row(it->first)<<1,1,1;
                } else {
                    const float blend = std::min(1.0f, std::max(0.0f,
                        (diff.count()) /
                        ViewerSettings::renderFaceTime));
                    C.row(it->first)<<1.f,blend,1.f;
                }
            }
            viewer.data().set_colors(C);
            // set ball color
            if (VIEWER_SETTINGS.newHit) {
                Eigen::VectorXd power;
                int ite = 0;
                while (ite++ < 5) {
                    power = solver->getQBufferNorm();
                    if (power.norm() > VIEWER_SETTINGS.transferBallBufThres) {
                        VIEWER_SETTINGS.newHit = false;
                        break;
                    }
                }
                transferVals = Eigen::VectorXd::Ones(transferVals.size())*0.1;
                for (int ii=0; ii<V_ball.rows(); ++ii) {
                    double &val = transferVals(ii);
                    val = 0.1*std::log10(power.dot(transfer_ball.col(ii)));
                    val /= VIEWER_SETTINGS.transferBallNormalization;
                    val = std::max(0.1, std::min(1.0, val));
                }
                igl::colormap(
                    igl::COLOR_MAP_TYPE_JET,transferVals,true,C_ball);
            }
            viewer.data(1).set_colors(C_ball);
            // handling the matrices myself
            for (int ii=0; ii<2; ++ii) {
                auto &core = viewer.core(main_view);
                auto &core_slave = ii == 0 ?
                    viewer.core(hud_view) : viewer.core(mode_view);
    		    core_slave.view = Eigen::Matrix4f::Identity();
    		    core_slave.proj = Eigen::Matrix4f::Identity();
    		    core_slave.norm = Eigen::Matrix4f::Identity();

    		    float width  = core_slave.viewport(2);
    		    float height = core_slave.viewport(3);

    		    // Set view
                igl::look_at(core.camera_eye, core.camera_center,
                    core.camera_up, core_slave.view);
                if (ii==0) {
    		        core_slave.view = core_slave.view
    		          * (core.trackball_angle * Eigen::Scaling(1.2f)
    		          * Eigen::Translation3f(core.camera_translation
                          + core.camera_base_translation)).matrix();
                } else {
                    float zoom =
                        core.camera_zoom*core.camera_base_zoom*
                        MODAL_VIEWER.zoom;
    		        core_slave.view = core_slave.view
    		          * (core.trackball_angle * Eigen::Scaling(zoom)
    		          * Eigen::Translation3f(core.camera_translation
                          + core.camera_base_translation)).matrix();
                }
    		    core_slave.norm = core_slave.view.inverse().transpose();

    		    // Set projection
    		    if (core.orthographic)
    		    {
    		      float length = (core.camera_eye - core.camera_center).norm();
    		      float h = tan(core.camera_view_angle/360.0 * igl::PI)*length;
                  igl::ortho(-h*width/height, h*width/height, -h, h,
                          core.camera_dnear, core.camera_dfar,core_slave.proj);
    		    }
    		    else
    		    {
                  float fH = tan(core.camera_view_angle / 360.0 * igl::PI) *
                      core.camera_dnear;
    		      float fW = fH * (double)width/(double)height;
                  igl::frustum(-fW, fW, -fH, fH, core.camera_dnear,
                          core.camera_dfar, core_slave.proj);
    		    }
            }
            // update mode viewer
            viewer.data(mod_id).set_visible(
                VIEWER_SETTINGS.drawModes, mode_view);
            std::unique_lock<std::mutex> lockLoad(
                VIEWER_SETTINGS.mutexLoadingNewModel);
            while (VIEWER_SETTINGS.loadingNewModel) {
                VIEWER_SETTINGS.cvLoadingNewModel.wait(lockLoad);
            }
            if (VIEWER_SETTINGS.drawModes) {
                auto &mv = MODAL_VIEWER;
                mv.UpdateModeShape(modes, VIEWER_SETTINGS.drawModeIdx);
                const double tmp = sqrt(modes->omegaSquared(mv.ind)/2000.);
                mv.V = mv.V0 + mv.Z*mv.scale*cos(tmp*mv.time);
                mv.time += 1./(tmp/(2.*3.14159))/24.;
                igl::colormap(igl::COLOR_MAP_TYPE_PARULA,mv.Zv,true,mv.C);
                viewer.data(mod_id).set_mesh(mv.V, mv.F0);
                viewer.data(mod_id).set_colors(mv.C);
            }
            return false;
        };
    viewer.callback_key_pressed = [&](
        igl::opengl::glfw::Viewer &viewer, unsigned int key, int mod)->bool {
        bool used = false;
        if (key=='1') {
            VIEWER_SETTINGS.force_type_int = 0;
            used = true;
        }
        else if (key=='2') {
            VIEWER_SETTINGS.force_type_int = 1;
            used = true;
        }
        else if (key=='3') {
            VIEWER_SETTINGS.force_type_int = 2;
            used = true;
        }
        else if (key==']') {
            VIEWER_SETTINGS.drawModeIdx =
                std::min(VIEWER_SETTINGS.drawModeIdx+1, N_modesAudible-1);
            used = true;
        }
        else if (key=='[') {
            VIEWER_SETTINGS.drawModeIdx =
                std::max(VIEWER_SETTINGS.drawModeIdx-1, 0);
            used = true;
        }
        else if (key=='u' || key=='U') {
            VIEWER_SETTINGS.useTextures = !VIEWER_SETTINGS.useTextures;
            viewer.data(obj_id).show_texture = VIEWER_SETTINGS.useTextures;
            viewer.data(obj_id).set_face_based(!VIEWER_SETTINGS.useTextures);
            used = true;
        }
        else if (key=='r' || key=='R') {
            LoadNewModel(
                obj_file,
                mod_file,
                mat_file,
                fat_path,
                V,
                VN,
                F,
                transfer_ball,
                V_ball,
                C_ball,
                transferVals,
                pos,
                obj_id,
                sph_id,
                mod_id,
                main_view,
                hud_view,
                N_modesAudible,
                parser,
                tex_R,
                tex_G,
                tex_B,
                tex_A,
                viewer,
                material,
                modes,
                solver);
            used = true;
        }
        else if (key=='d' || key=='D') {
            ForceMessage<double> force;
            GetModalForceCopy(VIEWER_SETTINGS.hitForceCache, force);
            solver->enqueueForceMessage(force);
            VIEWER_SETTINGS.activeFaceIds.push_back(
                    {VIEWER_SETTINGS.hitFidCache,
                    std::chrono::high_resolution_clock::now()});
        }
        return used;
    };
    viewer.callback_post_draw = [&](
        igl::opengl::glfw::Viewer &viewer) {
            if (VIEWER_SETTINGS.loadingNewModel) {
                return false;
            }
            // detect sustained force stuff with mouse
            if (VIEWER_SETTINGS.sustainedForceActive &&
                viewer.down) {
                auto &ar_parm = VIEWER_SETTINGS.arForceParameters;
                int fid, vid;
                Eigen::Vector3d pos, coords;
                Eigen::Vector3i vids;
                if (CurrentMouseSurfPos(viewer, V, F, pos, coords, fid, vid)) {
                    vids = F.row(fid);
                    Eigen::Vector3d vn;
                    ar_parm.curr_surf_pos = pos;
                    if (ar_parm.past_mouse_init) {
                        ar_parm.curr_surf_vel =
                            (ar_parm.curr_surf_pos - ar_parm.past_surf_pos) /
                            ar_parm.time_step_size;
                    }
                    else {
                        ar_parm.curr_surf_vel.setZero();
                        ar_parm.past_mouse_init = true;
                    }
                    ar_parm.past_surf_pos = ar_parm.curr_surf_pos;
                    vn = ar_parm.curr_surf_vel;
                    if (vn.norm() > 1E-10)
                        vn/=vn.norm();
                    ForceMessage<double> force;
                    GetModalForceFace(N_modesAudible, *modes, vids, coords, vn,
                        force);
                    solver->enqueueForceMessage(force);
                    VIEWER_SETTINGS.hitVidCache = vid;
                    viewer.data().set_colors(C);
                }
                else {
                    ar_parm.past_mouse_init = false;
                }
            }
            // PA stuff
            if (!PA_STREAM_STARTED) {
                CHECK_PA_LAUNCH(Pa_StartStream(stream));
                PA_STREAM_STARTED = true;
            }
            // update the transfer
            static Eigen::Matrix<double,3,1> last_camera_pos;
            const Eigen::Matrix<double,3,1> camera_pos =
                getCameraWorldPosition(viewer.core(main_view));
            static bool cache_initialize = false;
            if (camera_pos != last_camera_pos || !cache_initialize) {
                solver->computeTransfer(camera_pos);
                last_camera_pos = camera_pos;
                cache_initialize = true;
            }
            return false;
        };

    if (parser->get<std::string>("tex") != FILE_NOT_EXIST) {
        VIEWER_SETTINGS.useTextures = true;
        igl::png::readPNG(
            parser->get<std::string>("tex"),
            tex_R,tex_G,tex_B,tex_A);
        viewer.data(obj_id).set_texture(tex_R,tex_G,tex_B,tex_A);
        viewer.data(obj_id).set_face_based(false);
        viewer.data(obj_id).show_lines = false;
        viewer.data(obj_id).show_texture = true;
        viewer.launch_init(true, false, "RT Modal Sound");
        viewer.data(obj_id).meshgl.init();
        igl::opengl::destroy_shader_program(
            viewer.data(obj_id).meshgl.shader_mesh);
        igl::opengl::create_shader_program(
                mesh_vertex_shader_string,
                mesh_fragment_shader_string,
                {},
                viewer.data(obj_id).meshgl.shader_mesh);
        viewer.launch_rendering(true);
        viewer.launch_shut();
    }
    else {
        viewer.launch(true, false, "RT Modal Sound");
    }
    CHECK_PA_LAUNCH(Pa_StopStream(stream));
    CHECK_PA_LAUNCH(Pa_CloseStream(stream));
    CHECK_PA_LAUNCH(Pa_Terminate());
    VIEWER_SETTINGS.terminated = true;
    //threadSim.join(); // TODO: should join but may get trapped in
    //solver->step() because of the no fail enqueue
}
//##############################################################################
