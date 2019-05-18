#include <set>
#include <ctime>
#include <thread>
#include <pthread.h>
#include <string>
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
    ModalIntegrator<double> *integrator = ModalIntegrator<double>::Build(
        material->density,
        modes._omegaSquared,
        material->alpha,
        material->beta,
        1./(double)SAMPLE_RATE);
    int N_modesAudible = modes.numModes();
    if (parser->get<std::string>("p") != FILE_NOT_EXIST) {
        std::string maxFreqFile =
            parser->get<std::string>("p") + "/freq_threshold.txt";
        std::ifstream stream(maxFreqFile.c_str());
        assert(stream &&
            "freq_threshold.txt does not exist in the ffat folder");
        std::string line;
        std::getline(stream, line);
        std::istringstream iss(line);
        double maxFreq;
        iss >> maxFreq;
        N_modesAudible = modes.numModesAudible(material->density, maxFreq);
    }
    ModalSolver<double> solver(N_modesAudible);
    solver.setIntegrator(integrator);

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
    C = Eigen::MatrixXd::Constant(F.rows(),3,1);
    viewer.callback_mouse_down =
        [&V,&F,&C,&modes,&VN,&solver,&N_modesAudible](
            igl::opengl::glfw::Viewer& viewer, int, int modifier)->bool {
            if (modifier == GLFW_MOD_SHIFT) {
                int fid, vid;
                Eigen::Vector3f bc;
                Eigen::Vector3d vn;
                Eigen::RowVector3d hp;
                // Cast a ray in the view direction starting from the mouse position
                double x = viewer.current_mouse_x;
                double y = viewer.core.viewport(3) - viewer.current_mouse_y;
                if(igl::unproject_onto_mesh(Eigen::Vector2f(x,y), viewer.core.view,
                            viewer.core.proj, viewer.core.viewport, V, F, fid, bc)) {
                    // paint hit red
                    C.row(fid)<<1,0,0;
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
                    hp = V.row(vid);
                    viewer.data().point_size = 5;
                    viewer.data().set_points(hp, Eigen::RowVector3d(0,1,0));
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
        ImGui::SetNextWindowSize(ImVec2(200,500));
        ImGui::Begin("Simulation");
        auto extractLast = [](const std::string str, const std::string delim) {
            return str.substr(str.rfind(delim)+1);
        };
        if (ImGui::CollapsingHeader("Info", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::Button("Unfold All")) {
                std::cout << "unfold all\n";
            }
            ImGui::SameLine();
            if (ImGui::Button("Fold All")) {
                std::cout << "fold all\n";
            }
            if (ImGui::TreeNode("Geometry")) {
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.8f,0.8f,0.8f,1.0f));
                ImGui::Columns(2, "mycolumns2", false);
                ImGui::Text("Mesh"); ImGui::NextColumn();
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
		if (ImGui::CollapsingHeader("Sound Model")) {
            static bool animate = true;
            static float arr[] = { 0.6f, 0.1f, 1.0f, 0.5f, 0.92f, 0.1f, 0.2f };
            static float values[90] = { 0 };
            static int values_offset = 0;
            static double refresh_time = 0.0;
            if (!animate || refresh_time == 0.0f)
                refresh_time = ImGui::GetTime();
            while (refresh_time < ImGui::GetTime()) // Create dummy data at fixed 60 hz rate for the demo
            {
                static float phase = 0.0f;
                values[values_offset] = cosf(phase);
                values_offset = (values_offset+1) % IM_ARRAYSIZE(values);
                phase += 0.10f*values_offset;
                refresh_time += 1.0f/60.0f;
            }
            ImGui::PlotLines("Lines", values, IM_ARRAYSIZE(values), values_offset, "avg 0.0", -1.0f, 1.0f, ImVec2(0,80));
            const auto transfer = solver.getLatestTransfer();
            const Eigen::Matrix<float,-1,1> hist = transfer.data.cast<float>()/
                transfer.data.array().abs().maxCoeff();
            ImGui::PlotHistogram("Transfer",
                hist.data(),
                solver.getLatestTransfer().data.size(),
                0, NULL, 0.0f, 1.0f, ImVec2(0,80));
		}
        ImGui::End();
    };

    viewer.data().set_mesh(V, F);
    viewer.data().set_colors(C);
    viewer.data().show_lines = false;
    viewer.data().set_face_based(true);
    viewer.callback_post_draw =
        [&stream, &solver](igl::opengl::glfw::Viewer &viewer) {
            if (!PA_STREAM_STARTED) {
                CHECK_PA_LAUNCH(Pa_StartStream(stream));
                PA_STREAM_STARTED = true;
            }
            // update the transfer
            auto &core = viewer.core;
            // TODO check if this transformation is correct
            Eigen::Matrix<float,4,1> eye;
            eye << core.camera_eye, 1.0f;
            const Eigen::Matrix<double,3,1> camera_pos =
                (core.view.inverse() * eye).cast<double>().head(3);
            static Eigen::Matrix<double,3,1> last_camera_pos;
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
