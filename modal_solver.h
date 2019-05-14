#ifndef MODAL_SOLVER_H
#define MODAL_SOLVER_H
#include <ctime>
#include "Eigen/Dense"
#include "external/readerwriterqueue.h"
#include "modal_integrator.h"
//##############################################################################
template<typename T, int BUF_SIZE=FRAMES_PER_BUFFER>
struct ForceMessage {
    enum Type {
        impulse
    };
    Eigen::Matrix<T,-1,1> data;
};
//##############################################################################
template<typename T, int BUF_SIZE=FRAMES_PER_BUFFER>
struct SoundMessage {
    Eigen::Matrix<T,BUF_SIZE,1> data;
};
//##############################################################################
template<typename T, int BUF_SIZE=FRAMES_PER_BUFFER>
class ModalSolver {
private:
    typedef moodycamel::ReaderWriterQueue<T> Queue;
    moodycamel::ReaderWriterQueue<ForceMessage<T, BUF_SIZE>> _queue_force;
    moodycamel::ReaderWriterQueue<SoundMessage<T, BUF_SIZE>> _queue_sound;
    SoundMessage<T, BUF_SIZE> _mess_sound;
    ForceMessage<T, BUF_SIZE> _mess_force;
    ModalIntegrator<T> *_integrator = nullptr;
    const int _N_modes;
public:
    explicit ModalSolver(const int N_modes)
        : _queue_force(512),
          _queue_sound(2),
          _N_modes(N_modes) {
    }
    inline void setIntegrator(ModalIntegrator<T> *integrator)
    {_integrator = integrator;}
    inline ForceMessage<T, BUF_SIZE> &getForceMessage()
    {return _mess_force;}


    void step();
    bool enqueueForceMessage(const ForceMessage<T, BUF_SIZE> &mess);
    bool dequeueForceMessage(ForceMessage<T, BUF_SIZE> &mess);
    bool enqueueSoundMessage(const SoundMessage<T, BUF_SIZE> &mess);
    bool dequeueSoundMessage(SoundMessage<T, BUF_SIZE> &mess);
};
//##############################################################################
template<typename T, int BUF_SIZE>
void ModalSolver<T, BUF_SIZE>::step(){
    // fetch one force message and process it, then step the ode in buffer
    bool success = dequeueForceMessage(_mess_force);
    if (!success) { // use zero force
        for (int ii=0; ii<BUF_SIZE; ++ii) {
            const Eigen::Matrix<T,-1,1> &q =
                _integrator->Step();
            _mess_sound.data(ii) = q.sum(); // TODO: add transfer/scaling here
        }
    } else {
        assert(_mess_force.data.size() == _N_modes &&
               "dimension of force message incorrect");
        for (int ii=0; ii<BUF_SIZE; ++ii) {
            // TODO: turn force message into force data for each step in buf
            // use force.data as placeholder
            if (ii == 0) {
                const Eigen::Matrix<T,-1,1> &q =
                    _integrator->Step(_mess_force.data);
                _mess_sound.data(ii) = q.sum(); // TODO: add transfer/scaling here
            } else {
                const Eigen::Matrix<T,-1,1> &q =
                    _integrator->Step();
                _mess_sound.data(ii) = q.sum(); // TODO: add transfer/scaling here
            }
        }
    }
    // keep trying to enqueue until successful
    while (true) {
        success = enqueueSoundMessage(_mess_sound);
        if (success) {
            break;
        }
    }
}
//##############################################################################
template<typename T, int BUF_SIZE>
bool ModalSolver<T, BUF_SIZE>::enqueueForceMessage(
    const ForceMessage<T, BUF_SIZE> &mess){
    return _queue_force.try_enqueue(mess);
}
//##############################################################################
template<typename T, int BUF_SIZE>
bool ModalSolver<T, BUF_SIZE>::dequeueForceMessage(
    ForceMessage<T, BUF_SIZE> &mess) {
    return _queue_force.try_dequeue(mess);
}
//##############################################################################
template<typename T, int BUF_SIZE>
bool ModalSolver<T, BUF_SIZE>::enqueueSoundMessage(
    const SoundMessage<T, BUF_SIZE> &mess){
    return _queue_sound.try_enqueue(mess);
}
//##############################################################################
template<typename T, int BUF_SIZE>
bool ModalSolver<T, BUF_SIZE>::dequeueSoundMessage(
    SoundMessage<T, BUF_SIZE> &mess) {
    return _queue_sound.try_dequeue(mess);
}
//##############################################################################
#endif
