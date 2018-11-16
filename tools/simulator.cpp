#include <string>
#include "igl/opengl/glfw/Viewer.h"
#include "igl/read_triangle_mesh.h"
#include "igl/unproject_onto_mesh.h"
#include "igl/per_vertex_normals.h"
#include "ModalMaterial.h"
#include "ModeData.h"
#include "cmd_parser.h"
#include "portaudio.h"
#include "modal_integrator.h"
////////////////////////////////////////////////////////////////////////////////
static bool PA_STREAM_STARTED = false;
static int SAMPLE_RATE = 44100.;
static int FORCE_DURATION = 50;
////////////////////////////////////////////////////////////////////////////////
#define CHECK_PA_LAUNCH(x) \
    { \
        auto pa_err = x; \
        if (pa_err != paNoError) \
            printf("**ERROR** PortAudio error: %s\n", Pa_GetErrorText(pa_err));\
    }
////////////////////////////////////////////////////////////////////////////////
cli::Parser *CreateParser(int argc, char **argv) {
    cli::Parser *parser = new cli::Parser(argc, argv);
    parser->set_required<std::string>("m", "mesh",
        "Triangle mesh for the object");
    parser->set_required<std::string>("s", "surf_mode",
        "surface modes file");
    parser->set_required<std::string>("t", "material",
        "modal material file");
    parser->set_optional<unsigned int>("n", "offline_samples", 0,
        "number of offline samples we take");
    parser->set_optional<std::string>("o", "output_file", "training_set.dat",
        "output file for the offline samples");
    parser->set_optional<float>("l", "offline_samples_sec", 1,
        "number of seconds to run each offline sample for");
    parser->run_and_exit_if_error();
    return parser;
}
////////////////////////////////////////////////////////////////////////////////
typedef struct {
    float left_phase;
    float right_phase;
} PaTestData;

int patestCallback(const void *inputBuffer,
                   void *outputBuffer,
                   unsigned long framesPerBuffer,
                   const PaStreamCallbackTimeInfo* timeInfo,
                   PaStreamCallbackFlags statusFlags,
                   void *userData) {
    /* Cast data passed through stream to our structure. */
    PaTestData *data = (PaTestData*)userData;
    float *out = (float*)outputBuffer;
    unsigned int i;
    (void) inputBuffer; /* Prevent unused variable warning. */

    for( i=0; i<framesPerBuffer; i++ ) {
        *out++ = data->left_phase;  /* left */
        *out++ = data->right_phase;  /* right */
        /* Generate simple sawtooth phaser that ranges between -1.0 and 1.0. */
        data->left_phase += 0.01f;
        /* When signal reaches top, drop back down. */
        if( data->left_phase >= 1.0f ) data->left_phase -= 2.0f;
        /* higher pitch so we can distinguish left and right. */
        data->right_phase += 0.03f;
        if( data->right_phase >= 1.0f ) data->right_phase -= 2.0f;
    }
    return 0;
}
////////////////////////////////////////////////////////////////////////////////
struct PaModalData {
    ModalIntegrator<double> *integrator;
    ModeData *modes;
    Eigen::VectorXd F;
    Eigen::VectorXd Fbuf;
    float volume = 0.0001f;
    int counter = 0;
    double hitStrength = 1.0;
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

    for( i=0; i<framesPerBuffer; i++ ) {
        const Eigen::VectorXd &q = data->integrator->Step(data->F);
        if (data->counter == FORCE_DURATION) // only set once
            data->F.setZero();
        const float qsum = (float)q.sum()*data->volume;
        *out++ = qsum;
        *out++ = qsum;
        ++(data->counter);
    }
    return 0;
}
////////////////////////////////////////////////////////////////////////////////
void TestPortAudio(const ModalMaterial &material, ModeData *modes) {
    CHECK_PA_LAUNCH(Pa_Initialize());
    //PaTestData data;
    PaModalData data;
    //const double dt = 1./sqrt(modes->omegaSquared(modes->numModes()-1)/material.density)/8.;
    data.modes = modes;
    data.integrator = ModalIntegrator<double>::Build(
        material.density,
        modes->_omegaSquared,
        material.alpha,
        material.beta,
        1./(double)SAMPLE_RATE);
    data.F.setOnes(modes->numModes());
    PaStream *stream;
    CHECK_PA_LAUNCH(Pa_OpenDefaultStream(&stream,
                         0,          /* no input channels */
                         2,          /* stereo output */
                         paFloat32,  /* 32 bit floating point output */
                         SAMPLE_RATE,
                         256,        /* frames per buffer, i.e. the number
                                        of sample frames that PortAudio will
                                        request from the callback. Many apps
                                        may want to use
                                        paFramesPerBufferUnspecified, which
                                        tells PortAudio to pick the best,
                                        possibly changing, buffer size.*/
                         PaModalCallback, /* this is your callback function */
                         &data )); /*This is a pointer that will be passed to
                                    your callback*/
    CHECK_PA_LAUNCH(Pa_StartStream(stream));
    Pa_Sleep(5*1000);
    CHECK_PA_LAUNCH(Pa_StopStream(stream));
    CHECK_PA_LAUNCH(Pa_CloseStream(stream));
    CHECK_PA_LAUNCH(Pa_Terminate());
}
////////////////////////////////////////////////////////////////////////////////
void TestModalIntegrator(const ModalMaterial &material, const ModeData &modes) {

    const double dt = 1./sqrt(modes.omegaSquared(modes.numModes()-1)/material.density)/8.;
    ModalIntegrator<double> *integrator = ModalIntegrator<double>::Build(
        material.density,
        modes._omegaSquared,
        material.alpha,
        material.beta,
        dt);

    Eigen::VectorXd Q;
    Q.setOnes(modes.numModes());
    for (int ii=0; ii<5000; ++ii) {
        if (ii==1)
            Q.setZero();
        const Eigen::VectorXd &q = integrator->Step(Q);
        std::cout << "q " << ii << " " << q.transpose() << std::endl;
    }
}
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {

    auto *parser = CreateParser(argc, argv);

    Eigen::MatrixXd V, C, VN;
    Eigen::MatrixXi F;
    igl::read_triangle_mesh(parser->get<std::string>("m").c_str(), V, F);
    igl::per_vertex_normals(V, F, VN);

    ModalMaterial *material =
        ModalMaterial::Read(parser->get<std::string>("t").c_str());

    ModeData modes;
    modes.read(parser->get<std::string>("s").c_str());


    if (parser->get<unsigned int>("n") > 0) {
        // offline sample mode
        std::string outfile = parser->get<std::string>("o");
        float secs = parser->get<float>("l");
        int N_samples = (int)parser->get<unsigned int>("n");
        int N_steps = (int)std::ceil(secs*SAMPLE_RATE);
        PaModalData sim;
        sim.modes = &modes;
        sim.F.setZero(modes.numModes());
        std::ofstream stream(outfile, std::ios::binary);
        stream.write((char*)&N_samples, sizeof(int));
        stream.write((char*)&N_steps, sizeof(int));
        float qsum;
        int vid;
        Eigen::Vector3d vn;
        for (int sample=0; sample<N_samples; ++sample) {
            sim.integrator = ModalIntegrator<double>::Build(
                material->density,
                modes._omegaSquared,
                material->alpha,
                material->beta,
                1./(double)SAMPLE_RATE);

            vid = rand() % (modes.numDOF()/3);
            vn = VN.row(vid).normalized()*sim.hitStrength;
            for (int mm=0; mm<modes.numModes(); ++mm) {
                sim.F(mm) = vn[0]*modes.mode(mm).at(vid*3+0)
                          + vn[1]*modes.mode(mm).at(vid*3+1)
                          + vn[2]*modes.mode(mm).at(vid*3+2);
            }

            stream.write((char*)&vid, sizeof(int));
            std::cout << sample << " => " << vid << " ";
            for (int step=0; step<N_steps; ++step) {
                const Eigen::VectorXd &q = sim.integrator->Step(sim.F);
                if (step == FORCE_DURATION)
                    sim.F.setZero();
                qsum = (float)q.sum()*sim.volume;
                if (step < 5 || step > N_steps-5) {
                    std::cout << qsum << " ";
                }
                else if (step == 10) {
                    std::cout << " ... ";
                }
                stream.write((char*)&qsum, sizeof(float));
            }
            std::cout << std::endl;
            delete sim.integrator;
            sim.integrator = nullptr;
        }
        return 0;
    }




    // setup audio callback stuff
    CHECK_PA_LAUNCH(Pa_Initialize());
    PaModalData paData;
    paData.modes = &modes;
    paData.integrator = ModalIntegrator<double>::Build(
        material->density,
        modes._omegaSquared,
        material->alpha,
        material->beta,
        1./(double)SAMPLE_RATE);
    paData.F.setOnes(modes.numModes());
    paData.Fbuf.setZero(modes.numModes());
    PaStream *stream;
    CHECK_PA_LAUNCH(Pa_OpenDefaultStream(&stream,
                         0,          /* no input channels */
                         2,          /* stereo output */
                         paFloat32,  /* 32 bit floating point output */
                         SAMPLE_RATE,
                         256,        /* frames per buffer, i.e. the number
                                        of sample frames that PortAudio will
                                        request from the callback. Many apps
                                        may want to use
                                        paFramesPerBufferUnspecified, which
                                        tells PortAudio to pick the best,
                                        possibly changing, buffer size.*/
                         PaModalCallback, /* this is your callback function */
                         &paData )); /*This is a pointer that will be passed to
                                    your callback*/

    assert(modes.numDOF() == V.rows()*3 && "DOFs mismatch");

    // Plot the mesh
    igl::opengl::glfw::Viewer viewer;

    C = Eigen::MatrixXd::Constant(F.rows(),3,1);
    viewer.callback_mouse_down =
        [&V,&F,&C,&paData,&modes,&VN](igl::opengl::glfw::Viewer& viewer, int, int modifier)->bool {
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
                    vn = VN.row(vid).normalized()*paData.hitStrength;

                    paData.Fbuf.setZero(modes.numModes());
                    for (int mm=0; mm<modes.numModes(); ++mm) {
                        paData.Fbuf(mm) = vn[0]*modes.mode(mm).at(vid*3+0)
                                        + vn[1]*modes.mode(mm).at(vid*3+1)
                                        + vn[2]*modes.mode(mm).at(vid*3+2);
                    }

                    paData.F = paData.Fbuf;
                    paData.counter = 0;
                    hp = V.row(vid);
                    viewer.data().point_size = 5;
                    viewer.data().set_points(hp, Eigen::RowVector3d(0,1,0));
                    viewer.data().set_colors(C);
                    return true;
                }
            }
            return false;
        };

    viewer.data().set_mesh(V, F);
    viewer.data().set_colors(C);
    viewer.data().show_lines = false;
    viewer.data().set_face_based(true);
    viewer.callback_post_draw =
        [&stream](igl::opengl::glfw::Viewer &viewer) {
            if (!PA_STREAM_STARTED) {
                CHECK_PA_LAUNCH(Pa_StartStream(stream));
                PA_STREAM_STARTED = true;
            }
            return false;
        };
    viewer.launch();
    CHECK_PA_LAUNCH(Pa_StopStream(stream));
    CHECK_PA_LAUNCH(Pa_CloseStream(stream));
    CHECK_PA_LAUNCH(Pa_Terminate());
}
////////////////////////////////////////////////////////////////////////////////
