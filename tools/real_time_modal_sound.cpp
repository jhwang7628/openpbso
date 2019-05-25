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
} VIEWER_SETTINGS;
float ViewerSettings::renderFaceTime = 1.5f;
//##############################################################################
Eigen::Matrix<double,3,1> getCameraWorldPosition(
    const igl::opengl::glfw::Viewer &viewer) {
    auto &core = viewer.core;
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
    viewer.core.is_animating = true;
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
                double y = viewer.core.viewport(3) - viewer.current_mouse_y;
                if(igl::unproject_onto_mesh(
                    Eigen::Vector2f(x,y), viewer.core.view,
                    viewer.core.proj, viewer.core.viewport, V, F, fid, bc)) {
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
                    ForceMessage<double> &force = solver.getForceMessage();
                    force.data.setZero(N_modesAudible);
                    for (int mm=0; mm<N_modesAudible; ++mm) {
                        force.data(mm) = vn[0]*modes.mode(mm).at(vid*3+0)
                                       + vn[1]*modes.mode(mm).at(vid*3+1)
                                       + vn[2]*modes.mode(mm).at(vid*3+2);
                    }
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
        //menu.draw_viewer_menu();

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
            ImGui::BeginChild(
                "option",
                ImVec2(220, 115),
                true);
            ImGui::Text("Rendering Options");
            ImGui::Checkbox(
                "Enable FFAT transfer", &VIEWER_SETTINGS.useTransfer);
            if (VIEWER_SETTINGS.useTransfer !=
                VIEWER_SETTINGS.useTransferCache) {
                solver.setUseTransfer(VIEWER_SETTINGS.useTransfer);
                solver.computeTransfer(
                    getCameraWorldPosition(viewer));
                VIEWER_SETTINGS.useTransferCache = VIEWER_SETTINGS.useTransfer;
            }
            if (ImGui::Button("Repeat hit")) {
                solver.enqueueForceMessage(VIEWER_SETTINGS.hitForceCache);
                VIEWER_SETTINGS.activeFaceIds.push_back(
                    {VIEWER_SETTINGS.hitFidCache,
                    std::chrono::high_resolution_clock::now()});
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
                ImVec2(220, 115),
                true);
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
            // TODO send only the needed rows
            viewer.data().set_colors(C);
            return false;
        };
    viewer.callback_post_draw =
        [&stream, &solver](igl::opengl::glfw::Viewer &viewer) {
            if (!PA_STREAM_STARTED) {
                CHECK_PA_LAUNCH(Pa_StartStream(stream));
                PA_STREAM_STARTED = true;
            }
            // update the transfer
            static Eigen::Matrix<double,3,1> last_camera_pos;
            const Eigen::Matrix<double,3,1> camera_pos =
                getCameraWorldPosition(viewer);
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
