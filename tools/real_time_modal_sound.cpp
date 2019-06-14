#include <set>
#include <ctime>
#include <thread>
#include <pthread.h>
#include <string>
#include <deque>
#include <chrono>
#include "igl/read_triangle_mesh.h"
#include "igl/unproject_onto_mesh.h"
#include "igl/per_vertex_normals.h"
#include "igl/opengl/glfw/Viewer.h"
#include "igl/opengl/glfw/imgui/ImGuiMenu.h"
#include "igl/opengl/glfw/imgui/ImGuiHelpers.h"
#include "imgui/imgui.h"
#include "config.h"
#include "ModalMaterial.h"
#include "ModeData.h"
#include "cmd_parser.h"
#include "portaudio.h"
#include "modal_integrator.h"
#include "modal_solver.h"
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
    parser->set_optional<std::string>("p", "ffat_map", FILE_NOT_EXIST,
        "ffat map folder that contains *.fatcube files");
    parser->run_and_exit_if_error();
    return parser;
}
//##############################################################################
struct ViewerSettings {
    bool useTransfer = true;
    bool useTransferCache = true;
    float bufferHealth[100] = {1.0f};
    int bufferHealthPtr = 0;
    static float renderFaceTime;
    std::deque<std::pair<int,std::chrono::time_point<
        std::chrono::high_resolution_clock>>> activeFaceIds;
    ForceMessage<double> hitForceCache;
    int hitFidCache = 0;
    void incrementBufferHealthPtr() {
        bufferHealthPtr = (bufferHealthPtr+1)%100;
    }
    static ForceType forceType;
    struct GaussianForceParameters {
        float timeScale;
    } gaussianForceParameters;
} VIEWER_SETTINGS;
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
struct PaModalData {
    ModalSolver<double> *solver;
    SoundMessage<double> soundMessage;
};
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
    bool success = data->solver->dequeueSoundMessage(data->soundMessage);
    VIEWER_SETTINGS.bufferHealth[VIEWER_SETTINGS.bufferHealthPtr] =
        (float)success;
    VIEWER_SETTINGS.incrementBufferHealthPtr();
    for( i=0; i<framesPerBuffer; i++ ) {
        *out++ = (float)(data->soundMessage.data(i)/1E10);
        *out++ = (float)(data->soundMessage.data(i)/1E10);
    }
    return 0;
}
template<typename T>
void StepSolver(ModalSolver<T> &solver) {
    while (true) {
        solver.step();
    }
}
//##############################################################################
template<typename T>
void GetModalForceVertex(
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
}
//##############################################################################
//##############################################################################
int main(int argc, char **argv) {
    auto *parser = CreateParser(argc, argv);

    // read geometry
    Eigen::MatrixXd V, C, VN;
    Eigen::MatrixXi F;
    igl::read_triangle_mesh(parser->get<std::string>("m").c_str(), V, F);
    igl::per_vertex_normals(V, F, VN);

    // read modal materials and mode shapes
    ModalMaterial<double> *material =
        ModalMaterial<double>::Read(parser->get<std::string>("t").c_str());
    ModeData<double> modes;
    modes.read(parser->get<std::string>("s").c_str());
    assert(modes.numDOF() == V.rows()*3 && "DOFs mismatch");

    // build modal integrator and solver/scheduler
    int N_modesAudible = modes.numModes();
    if (parser->get<std::string>("p") != FILE_NOT_EXIST) {
        std::string maxFreqFile =
            parser->get<std::string>("p") + "/freq_threshold.txt";
        std::ifstream stream(maxFreqFile.c_str());
        if (stream) {
            std::string line;
            std::getline(stream, line);
            std::istringstream iss(line);
            double maxFreq;
            iss >> maxFreq;
            N_modesAudible = modes.numModesAudible(material->density, maxFreq);
        }
    }
    std::cout << "Number of audible modes: " << N_modesAudible << std::endl;
    ModalSolver<double> solver(N_modesAudible);
    ModalIntegrator<double> *integrator = ModalIntegrator<double>::Build(
        material->density,
        modes._omegaSquared,
        material->alpha,
        material->beta,
        1./(double)SAMPLE_RATE,
        N_modesAudible);
    solver.setIntegrator(integrator);
    VIEWER_SETTINGS.hitForceCache.data.setZero(N_modesAudible);

    // start a simulation thread and use max priority
    std::thread threadSim([&solver](){
        while (true) {
            solver.step();
        }
    });
    sched_param sch_params;
    sch_params.sched_priority = sched_get_priority_max(SCHED_FIFO);
    pthread_setschedparam(threadSim.native_handle(), SCHED_FIFO, &sch_params);

    // read transfer files if exist
    if (parser->get<std::string>("p") != FILE_NOT_EXIST) {
        solver.readFFATMaps(parser->get<std::string>("p"));
    }

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
    unsigned int main_view, hud_view;
    viewer.callback_init = [&](
            igl::opengl::glfw::Viewer &viewer) {
        viewer.core().viewport = Eigen::Vector4f(0, 0, 1280, 800);
        main_view = viewer.core_list[0].id;
        hud_view = viewer.append_core(Eigen::Vector4f(-50, 0, 400, (int)(400*800./1280.)));
        viewer.core(hud_view).update_transform_matrices = false;
        return false;
    };
    viewer.callback_post_resize = [&](
            igl::opengl::glfw::Viewer &v, int w, int h) {
        v.core(main_view).viewport = Eigen::Vector4f(0, 0, w, h);
        v.core(hud_view).viewport = Eigen::Vector4f(-50, 0, 400, (int)(400*800./1280.));
        return true;
    };
    C = Eigen::MatrixXd::Constant(F.rows(),3,1);
    viewer.callback_mouse_down =
        [&V,&F,&C,&modes,&VN,&solver,&N_modesAudible](
            igl::opengl::glfw::Viewer& viewer, int, int modifier)->bool {
            if (modifier == GLFW_MOD_SHIFT) {
                int fid, vid;
                Eigen::Vector3f bc;
                Eigen::Vector3d vn;
                Eigen::RowVector3d hp;
                // Cast a ray in the view direction starting from the mouse
                // position
                double x = viewer.current_mouse_x;
                double y = viewer.core().viewport(3) - viewer.current_mouse_y;
                if(igl::unproject_onto_mesh(
                    Eigen::Vector2f(x,y), viewer.core().view,
                    viewer.core().proj, viewer.core().viewport, V, F, fid, bc)){
                    // paint hit red
                    //C.row(fid)<<1,0,0;
                    VIEWER_SETTINGS.activeFaceIds.push_back(
                        {fid,std::chrono::high_resolution_clock::now()});
                    vid = F(fid, 0);
                    float lar = bc[0];
                    if (bc[1] > bc[0]) {
                        vid = F(fid, 1);
                        lar = bc[1];
                    }
                    if (bc[2] > lar) vid = F(fid, 2);
                    vn = VN.row(vid).normalized();
                    ForceMessage<double> force;
                    GetModalForceVertex(N_modesAudible, modes, vid, vn, force);
                    solver.enqueueForceMessage(force);
                    VIEWER_SETTINGS.hitForceCache = force;
                    VIEWER_SETTINGS.hitFidCache = fid;
                    hp = V.row(vid);
                    viewer.data().set_colors(C);
                    return true;
                }
            }
            return false;
        };

    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);
    menu.callback_draw_viewer_menu = [&]()
    {
        // viewer menu
        menu.draw_viewer_menu();

        // simulation menu
        ImGui::SetNextWindowPos(ImVec2(0,0));
        ImGui::SetNextWindowSize(ImVec2(250,0));
        ImGui::Begin("Simulation");
        auto extractLast = [](const std::string str, const std::string delim) {
            return str.substr(str.rfind(delim)+1);
        };
        if (ImGui::CollapsingHeader("Info", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::TreeNode("Geometry")) {
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.8f,0.8f,0.8f,1.0f));
                ImGui::Columns(2, "mycolumns2", false);
                ImGui::Text("Mesh file"); ImGui::NextColumn();
                ImGui::Text("%s", extractLast(parser->get<std::string>("m"), "/").c_str()); ImGui::NextColumn();
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
                ImGui::Text("%s", extractLast(parser->get<std::string>("s"), "/").c_str()); ImGui::NextColumn();
                ImGui::Text("#Modes"); ImGui::NextColumn();
                ImGui::Text("%d", modes.numModes()); ImGui::NextColumn();
                ImGui::Text("#Audible modes"); ImGui::NextColumn();
                ImGui::Text("%d", N_modesAudible); ImGui::NextColumn();
                ImGui::Text("Material file"); ImGui::NextColumn();
                ImGui::Text("%s", extractLast(parser->get<std::string>("t"), "/").c_str()); ImGui::NextColumn();
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
                ImVec2(220, 140),
                true);
            if (ImGui::Button("Repeat hit")) {
                ForceMessage<double> force;
                GetModalForceVertex(VIEWER_SETTINGS.hitForceCache, force);
                solver.enqueueForceMessage(force);
                VIEWER_SETTINGS.activeFaceIds.push_back(
                    {VIEWER_SETTINGS.hitFidCache,
                    std::chrono::high_resolution_clock::now()});
            }
            static int force_type = 0;
            ImGui::RadioButton("Point force", &force_type, 0);
            ImGui::RadioButton("Gaussian force", &force_type, 1);
            ImGui::RadioButton("Autoregressive force", &force_type, 2);
            VIEWER_SETTINGS.forceType = static_cast<ForceType>(force_type);
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
            const auto transfer = solver.getLatestTransfer();
            const Eigen::Matrix<float,-1,1> hist = transfer.data.cast<float>()/
                transfer.data.array().abs().maxCoeff();
            ImGui::BeginChild(
                "histogram",
                ImVec2(220, 145),
                true);
            ImGui::Checkbox(
                "Enable FFAT transfer", &VIEWER_SETTINGS.useTransfer);
            if (VIEWER_SETTINGS.useTransfer !=
                VIEWER_SETTINGS.useTransferCache) {
                solver.setUseTransfer(VIEWER_SETTINGS.useTransfer);
                solver.computeTransfer(
                    getCameraWorldPosition(viewer.core(main_view)));
                VIEWER_SETTINGS.useTransferCache = VIEWER_SETTINGS.useTransfer;
            }
            ImGui::Text("Transfer values for different modes:");
            ImGui::PlotHistogram("",
                hist.data(),
                solver.getLatestTransfer().data.size(),
                0, nullptr, 0.0f, 1.0f, ImVec2(200, 80));
            ImGui::EndChild();
            ImGui::Text("this is the next line");
		}
        ImGui::End();
    };

    viewer.data().set_mesh(V, F);
    viewer.data().set_colors(C);
    viewer.data().show_lines = false;
    viewer.data().set_face_based(true);
    // preparing for the HUD
    Eigen::MatrixXd V_ball, C_ball;
    Eigen::MatrixXi F_ball;
    // FIXME debug: this won't work outside the build directory.
    // should probably hard code the ball or use cmake to communicate the
    // base dir
    igl::read_triangle_mesh("../assets/ball.obj", V_ball, F_ball);
    C_ball.setOnes(V_ball.rows(), 3);
    // TODO
    const SoundMessage<double> *sound = solver.peekSoundQueue();
    Eigen::Vector3d pos;
    TransMessage<double> transfer;
    double val;
    const double sound_power = sound->data.norm();
    std::cout << "sound power = " << sound_power << std::endl;
    if (sound) {
        for (int ii=0; ii<V_ball.rows(); ++ii) {
            pos = V_ball.row(ii);
            solver.computeTransfer(pos, transfer);
            val = transfer.data(0) * sound_power;
            C_ball.row(ii) << val, val, val;
        }
    }

    viewer.append_mesh();
    viewer.data().set_mesh(V_ball, F_ball);
    viewer.data().show_lines = false;
    viewer.data().set_face_based(true);
    viewer.data().set_colors(Eigen::RowVector3d(0,0,1));
    viewer.data().shininess = 500.0f;
    viewer.selected_data_index = 0;
    viewer.data(1).set_visible(false, main_view);
    viewer.callback_pre_draw =
        [&](igl::opengl::glfw::Viewer &viewer) {
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
            viewer.data(1).set_colors(C_ball);


            // handling the matrices myself
            auto &core = viewer.core(main_view);
            auto &core_hud = viewer.core(hud_view);
    		core_hud.view = Eigen::Matrix4f::Identity();
    		core_hud.proj = Eigen::Matrix4f::Identity();
    		core_hud.norm = Eigen::Matrix4f::Identity();

    		float width  = core_hud.viewport(2);
    		float height = core_hud.viewport(3);

    		// Set view
            igl::look_at(core.camera_eye, core.camera_center, core.camera_up,
                    core_hud.view);
    		core_hud.view = core_hud.view
    		  * (core.trackball_angle
                      * Eigen::Scaling(0.1f * core.camera_base_zoom)
    		  * Eigen::Translation3f(core.camera_translation
                  + core.camera_base_translation)).matrix();
    		core_hud.norm = core_hud.view.inverse().transpose();

    		// Set projection
    		if (core.orthographic)
    		{
    		  float length = (core.camera_eye - core.camera_center).norm();
    		  float h = tan(core.camera_view_angle/360.0 * igl::PI) * (length);
              igl::ortho(-h*width/height, h*width/height, -h, h, core.camera_dnear, core.camera_dfar,core_hud.proj);
    		}
    		else
    		{
    		  float fH = tan(core.camera_view_angle / 360.0 * igl::PI) * core.camera_dnear;
    		  float fW = fH * (double)width/(double)height;
              igl::frustum(-fW, fW, -fH, fH, core.camera_dnear, core.camera_dfar, core_hud.proj);
    		}






            //viewer.core(hud_view).view = viewer.core(main_view).view;
            //viewer.core(hud_view).proj = viewer.core(main_view).proj;
            //viewer.core(hud_view).norm = viewer.core(main_view).norm;
            return false;
        };
    viewer.callback_post_draw =
        [&](igl::opengl::glfw::Viewer &viewer) {
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
                solver.computeTransfer(camera_pos);
                last_camera_pos = camera_pos;
                cache_initialize = true;
            }
            return false;
        };


    viewer.launch();
    CHECK_PA_LAUNCH(Pa_StopStream(stream));
    CHECK_PA_LAUNCH(Pa_CloseStream(stream));
    CHECK_PA_LAUNCH(Pa_Terminate());
}
//##############################################################################
