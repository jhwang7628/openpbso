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
template<typename T>
struct DataMessage {
    Eigen::Matrix<T,-1,1> data;
};
//##############################################################################
template<typename T, int BUF_SIZE=FRAMES_PER_BUFFER>
struct ForceMessage {
    Eigen::Matrix<T,-1,1> data;
    ForceType forceType = ForceType::PointForce;
    std::unique_ptr<Force<T,BUF_SIZE>> force;
    // signal used to indicate start/stop of sustained forces
    bool sustainedForceStart = false;
    bool sustainedForceEnd = false;
    bool clearAllForces = false;
    ForceMessage()
        : force(new PointForce<T,BUF_SIZE>()) {
    }
    ForceMessage(const ForceMessage<T,BUF_SIZE> &tar)
        : data(tar.data),
          forceType(tar.forceType),
          force(new PointForce<T,BUF_SIZE>()),
          sustainedForceStart(tar.sustainedForceStart),
          sustainedForceEnd(tar.sustainedForceEnd),
          clearAllForces(tar.clearAllForces) {
        _DeepCopyForce(tar);
    }
    ForceMessage<T,BUF_SIZE> &operator = (const ForceMessage<T,BUF_SIZE> &tar) {
        if (&tar == this)
            return *this;
        // perform a deep copy
        data = tar.data;
        forceType = tar.forceType;
        sustainedForceStart = tar.sustainedForceStart;
        sustainedForceEnd = tar.sustainedForceEnd;
        clearAllForces = tar.clearAllForces;
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
    moodycamel::ReaderWriterQueue<DataMessage<T>>            _queue_qnorm;
    moodycamel::ReaderWriterQueue<AutoregressiveForceParam<T>> _queue_arprm;
    SoundMessage<T, BUF_SIZE> _mess_sound;
    ForceMessage<T, BUF_SIZE> _mess_force; // buffer used to dequeue
    TransMessage<T>           _mess_trans; // buffer used to compute and enqueue
    TransMessage<T>           _latest_transfer;
    DataMessage<T>            _mess_qnorm;
    ModalIntegrator<T> *_integrator = nullptr;
    const int _N_modes;
    std::unique_ptr<std::map<int,FFAT_Map>> _ffat_maps;
    std::list<ForceMessage<T, BUF_SIZE>> _activeForces;
    Eigen::Matrix<T,-1,1> _forceSpreadBufferSpace;
    Eigen::Matrix<T,BUF_SIZE,1> _forceSpreadBufferTime;

    std::mutex _useTransferMutex;
    bool _useTransfer;
    bool _useTransferCache;
    bool _sustainedForces = false;

public:
    explicit ModalSolver(const int N_modes)
        : _queue_force(512),
          _queue_sound(2),
          _queue_trans(1),
          _queue_qnorm(2),
          _queue_arprm(1),
          _mess_trans(N_modes),
          _latest_transfer(N_modes),
          _N_modes(N_modes),
          _useTransfer(true),
          _useTransferCache(true) {
        _forceSpreadBufferSpace.resize(_N_modes);
        _mess_qnorm.data.setZero(_N_modes);
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
    inline Eigen::Matrix<T,-1,1> getQBufferNorm() {
        DataMessage<T> mess;
        if (_queue_qnorm.try_dequeue(mess)) {
            return mess.data;
        }
        return Eigen::Matrix<T,-1,1>::Zero(_N_modes);
    }

    void step();
    void readFFATMaps(const std::string &mapFolderPath);
    bool computeTransfer(const Eigen::Matrix<T,3,1> &pos);
    bool computeTransfer(const Eigen::Matrix<T,3,1> &pos, T *trans);
    bool enqueueForceMessageNoFail(const ForceMessage<T, BUF_SIZE> &mess, const
        int maxIte = -1);
    bool enqueueForceMessage(const ForceMessage<T, BUF_SIZE> &mess);
    bool dequeueForceMessage(ForceMessage<T, BUF_SIZE> &mess);
    bool enqueueSoundMessage(const SoundMessage<T, BUF_SIZE> &mess);
    bool dequeueSoundMessage(SoundMessage<T, BUF_SIZE> &mess);
    bool enqueueTransMessage(const TransMessage<T> &mess);
    bool dequeueTransMessage(TransMessage<T> &mess);
    bool enqueueArprmMessage(const AutoregressiveForceParam<T> &mess);
    bool enqueueArprmMessageNoFail(const AutoregressiveForceParam<T> &mess, const
        int maxIte = -1);
    bool dequeueArprmMessage(AutoregressiveForceParam<T> &mess);
};
//##############################################################################
template<typename T, int BUF_SIZE>
void ModalSolver<T, BUF_SIZE>::step(){
    // fetch one force message and process it, then step the ode in buffer
    bool success = dequeueForceMessage(_mess_force);
    if (success) {
        if (_mess_force.clearAllForces) {
            _activeForces.clear();
            return;
        }
        if (_mess_force.sustainedForceStart) {
            _activeForces.clear();
            _sustainedForces = true;
            _activeForces.push_back(_mess_force);
        }
        if (!_sustainedForces) {
            _activeForces.push_back(_mess_force);
        } else {
            // copy sustained force data over
            _activeForces.begin()->data = _mess_force.data;
        }
        if (_mess_force.sustainedForceEnd) {
            _activeForces.clear();
            _sustainedForces = false;
        }
    }
    _forceSpreadBufferTime.setZero();
    if (!_sustainedForces) {
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
    }
    else {
        assert(_activeForces.size() == 1 &&
            "Should only have 1 concurrent sustained force");
        auto it = _activeForces.begin();
        if (it->forceType == ForceType::AutoregressiveForce) {
            // check for new AR parameters
            AutoregressiveForceParam<T> arprm;
            if (dequeueArprmMessage(arprm)) {
                std::cout << "dequeued:\n";
                std::cout << " a = " << arprm.a[0] << ", " << arprm.a[1] << std::endl;
                std::cout << " sigma = " << arprm.sigma << std::endl;
                std::cout << " mu = " << arprm.mu << std::endl;
                static_cast<AutoregressiveForce<T,BUF_SIZE>*>(
                    it->force.get())->SetParam(arprm);
            }
        }
        it->force->Add(_forceSpreadBufferTime);
        _forceSpreadBufferSpace = it->data;
    }

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

    assert(_forceSpreadBufferSpace.size() == _N_modes &&
            "dimension of force message incorrect");

    // get log power of the sound buffer for each mode
    _mess_qnorm.data.setZero(_N_modes);
    for (int ii=0; ii<BUF_SIZE; ++ii) {
        const Eigen::Matrix<T,-1,1> &q =
            _integrator->Step(
                    _forceSpreadBufferSpace*_forceSpreadBufferTime(ii));
        _mess_sound.data(ii) =
            q.head(_latest_transfer.data.size()).dot(
                    _latest_transfer.data);
        _mess_qnorm.data.array() += q.array().square();
    }
    _mess_qnorm.data.array() = _mess_qnorm.data.array().sqrt();
    _queue_qnorm.try_enqueue(_mess_qnorm); // it's okay to fail this one
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
    auto &trans = _mess_trans;
    const int N = trans.data.size();
    for (int ii=0; ii<N; ++ii) {
        trans.data(ii) = std::abs(
            _ffat_maps->at(ii).GetMapVal(pos, trans.useCompressed));
    }
    const bool success = enqueueTransMessage(_mess_trans);
    return success;
}
//##############################################################################
template<typename T, int BUF_SIZE>
bool ModalSolver<T, BUF_SIZE>::computeTransfer(
    const Eigen::Matrix<T,3,1> &pos,
    T *trans) {
    // if no transfer maps are given, return immediately (and use all ones)
    if (!_ffat_maps)
        return false;
    const int N = _ffat_maps->size();
    for (int ii=0; ii<N; ++ii) {
        trans[ii] = std::abs(
            _ffat_maps->at(ii).GetMapVal(pos, _mess_trans.useCompressed));
    }
    return true;
}
//##############################################################################
template<typename T, int BUF_SIZE>
bool ModalSolver<T, BUF_SIZE>::enqueueForceMessageNoFail(
    const ForceMessage<T, BUF_SIZE> &mess, const int maxIte){
    int ite=0;
    while (maxIte < 0 || ite++ < maxIte) {
        if (_queue_force.try_enqueue(mess)) {
            return true;
        }
    }
    return false;
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
template<typename T, int BUF_SIZE>
bool ModalSolver<T, BUF_SIZE>::enqueueArprmMessage(
    const AutoregressiveForceParam<T> &mess) {
    return _queue_arprm.try_enqueue(mess);
}
//##############################################################################
template<typename T, int BUF_SIZE>
bool ModalSolver<T, BUF_SIZE>::enqueueArprmMessageNoFail(
    const AutoregressiveForceParam<T> &mess, const int maxIte) {
    int ite=0;
    while (maxIte < 0 || ite++ < maxIte) {
        if (_queue_arprm.try_enqueue(mess)) {
            return true;
        }
    }
    return false;
}
//##############################################################################
template<typename T, int BUF_SIZE>
bool ModalSolver<T, BUF_SIZE>::dequeueArprmMessage(
    AutoregressiveForceParam<T> &mess) {
    return _queue_arprm.try_dequeue(mess);
}
//##############################################################################
#endif
