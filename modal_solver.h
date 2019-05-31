#ifndef MODAL_SOLVER_H
#define MODAL_SOLVER_H
#include <ctime>
#include <memory>
#include <mutex>
#include <list>
#include "config.h"
#include "Eigen/Dense"
#include "external/readerwriterqueue.h"
#include "modal_integrator.h"
#include "ffat_solver.h"
#include "ffat_map_serialize.h"
#include "forces.h"
//##############################################################################
template<typename T, int BUF_SIZE=FRAMES_PER_BUFFER>
struct ForceMessage {
    Eigen::Matrix<T,-1,1> data;
    ForceType forceType = ForceType::PointForce;
    std::unique_ptr<Force<T,BUF_SIZE>> force;
    ForceMessage() = default;
    ForceMessage(const ForceMessage<T,BUF_SIZE> &tar)
        : data(tar.data),
          forceType(tar.forceType),
          force(new PointForce<T,BUF_SIZE>()){
        _DeepCopyForce(tar);
    }
    ForceMessage<T,BUF_SIZE> &operator = (const ForceMessage<T,BUF_SIZE> &tar) {
        if (&tar == this)
            return *this;
        // perform a deep copy
        data = tar.data;
        forceType = tar.forceType;
        _DeepCopyForce(tar);
        return *this;
    }
private:
    void _DeepCopyForce(const ForceMessage<T,BUF_SIZE> &tar) {
        if (tar.forceType == ForceType::PointForce) {
            force.reset(new PointForce<T,BUF_SIZE>(
                *(static_cast<PointForce<T,BUF_SIZE>*>(tar.force.get()))));
        } else if (tar.forceType == ForceType::GaussianForce) {
            force.reset(new GaussianForce<T,BUF_SIZE>(
                *(static_cast<GaussianForce<T,BUF_SIZE>*>(tar.force.get()))));
        } else if (tar.forceType == ForceType::AutoregressiveForce) {
            force.reset(new AutoregressiveForce<T,BUF_SIZE>(
                *(static_cast<AutoregressiveForce<T,BUF_SIZE>*>(
                    tar.force.get()))));
        } else {
            std::cout << static_cast<int>((tar.forceType)) << std::endl;
            assert(false && "unrecognized force type");
        }
    }
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
    int N = 0;
    Eigen::Matrix<T,-1,1> data;
    void setToUnit() {
        data.setOnes(N);
        data*=1E7;
    }
    explicit TransMessage() = default;
    explicit TransMessage(const int N_)
        : N(N_) {
        setToUnit();
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
    ForceMessage<T, BUF_SIZE> _mess_force; // buffer used to dequeue
    TransMessage<T>           _mess_trans; // buffer used to compute and enqueue
    TransMessage<T>           _latest_transfer;
    ModalIntegrator<T> *_integrator = nullptr;
    const int _N_modes;
    std::unique_ptr<std::map<int,FFAT_Map>> _ffat_maps;
    std::list<ForceMessage<T, BUF_SIZE>> _activeForces;
    Eigen::Matrix<T,-1,1> _forceSpreadBufferSpace;
    Eigen::Matrix<T,BUF_SIZE,1> _forceSpreadBufferTime;

    std::mutex _useTransferMutex;
    bool _useTransfer;
    bool _useTransferCache;

public:
    explicit ModalSolver(const int N_modes)
        : _queue_force(512),
          _queue_sound(2),
          _queue_trans(1),
          _mess_trans(N_modes),
          _latest_transfer(N_modes),
          _N_modes(N_modes),
          _useTransfer(true),
          _useTransferCache(true) {
        _forceSpreadBufferSpace.resize(_N_modes);
    }
    inline void setIntegrator(ModalIntegrator<T> *integrator) {
        _integrator = integrator;
    }
    inline const TransMessage<T> &getLatestTransfer() {
        return _latest_transfer;
    }
    inline void setUseTransfer(const bool s) {
        _useTransferMutex.lock();
        _useTransfer = s;
        _useTransferMutex.unlock();
    }

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
    if (success) {
        _activeForces.push_back(_mess_force);
    }

    // TODO get spread out force here
    _forceSpreadBufferTime.setZero();
    _forceSpreadBufferSpace.setZero();
    auto it = _activeForces.begin();
    while (it != _activeForces.end()) {
        assert(it->force && "obsolete forces should be removed");
        bool added = it->force->Add(_forceSpreadBufferTime);
        // if force not producing anymore, kill it.
        if (!added) {
            _activeForces.erase(it++);
        } else {
            _forceSpreadBufferSpace += it->data;
            ++it;
        }
    }
    //if (_mess_force.force) {
    //    _mess_force.force->Add(_forceSpreadBuffer);
    //} else {
    //    _forceSpreadBuffer.setZero();
    //}

    TransMessage<T> trans;
    bool useTransfer = _useTransferCache;
    // only set this field if can obtain lock
    if (_useTransferMutex.try_lock()) {
        useTransfer = _useTransfer;
        _useTransferMutex.unlock();
    }
    if (useTransfer) {
        if (dequeueTransMessage(trans)) {
            _latest_transfer = trans;
        }
    } else {
        _latest_transfer.setToUnit();
    }
    _useTransferCache = useTransfer;

    //if (!success) { // use zero force
    //    for (int ii=0; ii<BUF_SIZE; ++ii) {
    //        const Eigen::Matrix<T,-1,1> &q =
    //            _integrator->Step();
    //        _mess_sound.data(ii) =
    //            q.head(_latest_transfer.data.size()).dot(_latest_transfer.data);
    //    }
    //} else {
        assert(_forceSpreadBufferSpace.size() == _N_modes &&
               "dimension of force message incorrect");
        for (int ii=0; ii<BUF_SIZE; ++ii) {
            const Eigen::Matrix<T,-1,1> &q =
                _integrator->Step(
                    _forceSpreadBufferSpace*_forceSpreadBufferTime(ii));
            _mess_sound.data(ii) =
                q.head(_latest_transfer.data.size()).dot(
                    _latest_transfer.data);
        }
    //}
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
        Gpu_Wavesolver::FFAT_Map_Serialize::LoadAll(
            mapPath.c_str())
    );
}
//##############################################################################
template<typename T, int BUF_SIZE>
bool ModalSolver<T, BUF_SIZE>::computeTransfer(
    const Eigen::Matrix<T,3,1> &pos) {
    // if no transfer maps are given, return immediately (and use all ones)
    if (!_ffat_maps)
        return false;
    const int N = _mess_trans.data.size();
    for (int ii=0; ii<N; ++ii) {
        _mess_trans.data(ii) = std::abs(
            _ffat_maps->at(ii).GetMapVal(pos, _mess_trans.useCompressed));
    }
    // this might fail, if fail then skip this update
    bool success = enqueueTransMessage(_mess_trans);
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
