#ifndef MODAL_SOLVER_H
#define MODAL_SOLVER_H
#include <ctime>
#include <memory>
#include "Eigen/Dense"
#include "external/readerwriterqueue.h"
#include "modal_integrator.h"
#include "ffat_solver.h"
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
template<typename T>
struct TransMessage {
    bool useCompressed = false;
    Eigen::Matrix<T,-1,1> data;
    explicit TransMessage() = default;
    explicit TransMessage(const int N) {
        std::cout << "here\n";
        data.setOnes(N);
        std::cout << "here2\n";
    }
};
//##############################################################################
template<typename T, int BUF_SIZE=FRAMES_PER_BUFFER>
class ModalSolver {
private:
    typedef moodycamel::ReaderWriterQueue<T> Queue;
    typedef Gpu_Wavesolver::FFAT_Map<T,3> FFAT_Map;
    moodycamel::ReaderWriterQueue<ForceMessage<T, BUF_SIZE>> _queue_force;
    moodycamel::ReaderWriterQueue<SoundMessage<T, BUF_SIZE>> _queue_sound;
    moodycamel::ReaderWriterQueue<TransMessage<T>>           _queue_trans;
    SoundMessage<T, BUF_SIZE> _mess_sound;
    ForceMessage<T, BUF_SIZE> _mess_force;
    TransMessage<T>           _mess_trans; // buffer used to compute and enqueue
    TransMessage<T>           _latest_transfer;
    ModalIntegrator<T> *_integrator = nullptr;
    const int _N_modes;
    std::unique_ptr<std::map<int,FFAT_Map>> _ffat_maps;

public:
    explicit ModalSolver(const int N_modes)
        : _queue_force(512),
          _queue_sound(2),
          _queue_trans(1),
          _mess_trans(N_modes),
          _N_modes(N_modes) {
    }
    inline void setIntegrator(ModalIntegrator<T> *integrator)
    {_integrator = integrator;}
    inline ForceMessage<T, BUF_SIZE> &getForceMessage()
    {return _mess_force;}

    void step();
    void readFFATMaps(const std::string &mapFolderPath);
    bool computeTransfer(const Eigen::Matrix<T,3,1> &pos);
    bool enqueueForceMessage(const ForceMessage<T, BUF_SIZE> &mess);
    bool dequeueForceMessage(ForceMessage<T, BUF_SIZE> &mess);
    bool enqueueSoundMessage(const SoundMessage<T, BUF_SIZE> &mess);
    bool dequeueSoundMessage(SoundMessage<T, BUF_SIZE> &mess);
    bool enqueueTransMessage(const TransMessage<T> &mess);
    bool dequeueTransMessage(TransMessage<T> &mess);
};
//##############################################################################
template<typename T, int BUF_SIZE>
void ModalSolver<T, BUF_SIZE>::step(){
    // fetch one force message and process it, then step the ode in buffer
    bool success = dequeueForceMessage(_mess_force);

    TransMessage<T> trans;
    if (dequeueTransMessage(trans)) {
        _latest_transfer = trans;
        std::cout << "latest transfer = " << _latest_transfer.data << std::endl;
    }

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
void ModalSolver<T, BUF_SIZE>::readFFATMaps(const std::string &mapPath) {
    _ffat_maps = std::unique_ptr<std::map<int,FFAT_Map>>(
        FFAT_Map::LoadAll( mapPath.c_str()));
    std::cout << "loaded all maps" << std::endl;
}
//##############################################################################
template<typename T, int BUF_SIZE>
bool ModalSolver<T, BUF_SIZE>::computeTransfer(
    const Eigen::Matrix<T,3,1> &pos) {
    // if no transfer maps are given, return immediately (and use all ones)
    if (!_ffat_maps)
        return false;
    std::cout << "computing transer at pos = " << pos.transpose() << std::endl; // FIXME debug
    const int N = _mess_trans.data.size();
    std::cout << "N = " << N << std::endl;
    std::cout << "_ffat_maps.size() = " << _ffat_maps->size() << std::endl;
    for (const auto & m : *_ffat_maps) {
        std::cout << "ffat key = " << m.first << std::endl;
    }
    for (int ii=0; ii<N; ++ii) {
        std::cout << "ii = " << ii << std::endl;
        _mess_trans.data(ii) = std::abs(
            _ffat_maps->at(ii).GetMapVal(pos, _mess_trans.useCompressed));
    }
    std::cout << "transfer computed\n";
    // this might fail, if fail then skip this update
    bool success = enqueueTransMessage(_mess_trans);
    std::cout << "successfully encode transfer: " << _mess_trans.data << std::endl;
    return success;
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
template<typename T, int BUF_SIZE>
bool ModalSolver<T, BUF_SIZE>::enqueueTransMessage(
    const TransMessage<T> &mess){
    return _queue_trans.try_enqueue(mess);
}
//##############################################################################
template<typename T, int BUF_SIZE>
bool ModalSolver<T, BUF_SIZE>::dequeueTransMessage(
    TransMessage<T> &mess) {
    return _queue_trans.try_dequeue(mess);
}
//##############################################################################
#endif
