#include <cstring>
#include <initializer_list>
#include <optional>
#include <string>
#include <utility>
#include <pybind11/pybind11.h>

#include "RLEnvironment.cuh"
#include "EnvIO.cuh"

namespace py = pybind11;

// Accept a DLPack capsule or order a producer onto CARL's CUDA stream.
static py::capsule toCapsule(py::object obj, cudaStream_t stream)
{
    if (py::isinstance<py::capsule>(obj))
    {
        return obj.cast<py::capsule>();
    }

    return obj.attr("__dlpack__")(
        py::arg("stream") = reinterpret_cast<intptr_t>(stream)
    ).cast<py::capsule>();
}

// Only delete if still named "dltensor" - consumers rename to prevent double-free
static void capsuleDeleter(PyObject* o)
{
    const char* name = PyCapsule_GetName(o);
    if (name && std::strcmp(name, "dltensor") == 0)
    {
        auto* t = static_cast<DLManagedTensor*>(
            PyCapsule_GetPointer(o, "dltensor"));

        if (t)
        {
            if (t->deleter) t->deleter(t);
            else delete t;
        }
    }
}

// Wrap DLManagedTensor in a capsule, deleter frees wrapper not EnvIO's buffer
static py::object tensorToCapsule(DLManagedTensor* tensor)
{
    tensor->deleter = [](DLManagedTensor* self) { delete self; };
    return py::capsule(tensor, "dltensor", capsuleDeleter);
}

struct TensorView
{
    py::capsule capsule;
    DLTensor tensor;
};

// Holds env + I/O, DLPack tensors to Python are non-owning views over EnvIO buffers
class EnvWrapper
{
    RLEnvironment env;
    EnvIO io;
    int frameskip;

    const int32_t* actionData(const DLManagedTensor* tensor) const
    {
        if (!tensor) throw py::value_error("invalid DLPack tensor");

        const DLTensor& t = tensor->dl_tensor;
        const bool validType = t.dtype.code == kDLInt
            && t.dtype.bits == 32 && t.dtype.lanes == 1;

        if (!validType)
        {
            throw py::type_error("actions must have dtype int32");
        }

        if (t.device.device_type != kDLCUDA)
        {
            throw py::value_error("actions must be on a CUDA device");
        }

        bool dimMismatch = t.ndim != 3
                        || t.shape[0] != env.getNSim()
                        || t.shape[1] != env.getNCars()
                        || t.shape[2] != ACT_PER_CAR;

        if (dimMismatch)
        {
            throw py::value_error("actions must have shape [n_sim, n_cars, 7]");
        }

        bool strideMismatch = t.strides && (
               t.strides[2] != 1
            || t.strides[1] != ACT_PER_CAR
            || t.strides[0] != env.getNCars() * ACT_PER_CAR
        );

        if (strideMismatch)
        {
            throw py::value_error("actions must be contiguous");
        }

        return reinterpret_cast<const int32_t*>(
            static_cast<const char*>(t.data) + t.byte_offset);
    }

    void validateTensor(
        const DLTensor& tensor,
        const char* name,
        std::initializer_list<int64_t> shape,
        uint8_t type,
        uint8_t bits) const
    {
        if (tensor.dtype.code != type
            || tensor.dtype.bits != bits
            || tensor.dtype.lanes != 1)
        {
            throw py::type_error(std::string(name) + " has an invalid dtype");
        }

        if (tensor.device.device_type != kDLCUDA || tensor.device.device_id != 0)
        {
            throw py::value_error(std::string(name) + " must be on cuda:0");
        }

        if (tensor.ndim != static_cast<int32_t>(shape.size()))
        {
            throw py::value_error(std::string(name) + " has an invalid shape");
        }

        int dim = 0;
        int64_t stride = 1;
        for (int64_t expected : shape)
        {
            if (tensor.shape[dim++] != expected)
            {
                throw py::value_error(std::string(name) + " has an invalid shape");
            }
        }
        if (tensor.strides)
        {
            for (int i = tensor.ndim - 1; i >= 0; i--)
            {
                if (tensor.strides[i] != stride)
                {
                    throw py::value_error(std::string(name) + " must be contiguous");
                }
                stride *= tensor.shape[i];
            }
        }
    }

    TensorView tensorData(
        py::object obj,
        const char* name,
        std::initializer_list<int64_t> shape,
        uint8_t type,
        uint8_t bits) const
    {
        py::capsule capsule = toCapsule(obj, env.getStream());
        const DLManagedTensor* managed = capsule.get_pointer<DLManagedTensor>();
        if (!managed) throw py::value_error("invalid DLPack tensor");

        const DLTensor tensor = managed->dl_tensor;
        validateTensor(tensor, name, shape, type, bits);
        return { std::move(capsule), tensor };
    }

    TensorView selectionData(py::object obj, int& nSelected) const
    {
        py::capsule capsule = toCapsule(obj, env.getStream());
        const DLManagedTensor* managed = capsule.get_pointer<DLManagedTensor>();
        if (!managed) throw py::value_error("invalid DLPack tensor");

        const DLTensor tensor = managed->dl_tensor;
        const bool validType = tensor.dtype.code == kDLInt
            && tensor.dtype.bits == 64 && tensor.dtype.lanes == 1;
        if (!validType)
        {
            throw py::type_error("simulation_indices must have dtype int64");
        }
        if (tensor.device.device_type != kDLCUDA || tensor.device.device_id != 0)
        {
            throw py::value_error("simulation_indices must be on cuda:0");
        }
        if (tensor.ndim != 1)
        {
            throw py::value_error("simulation_indices must be one-dimensional");
        }
        if (tensor.strides && tensor.strides[0] != 1)
        {
            throw py::value_error("simulation_indices must be contiguous");
        }
        if (tensor.shape[0] > env.getNSim())
        {
            throw py::value_error("too many simulation_indices");
        }

        nSelected = static_cast<int>(tensor.shape[0]);
        return { std::move(capsule), tensor };
    }

    static const void* data(const DLTensor& tensor)
    {
        return static_cast<const char*>(tensor.data) + tensor.byte_offset;
    }

public:
    EnvWrapper(
        int nSim,
        int nBlue,
        int nOrange,
        int seed,
        int frameskip,
        bool invertOrange,
        bool normalize)
        : env(nSim, nBlue, nOrange, seed)
        , io(nSim, env.getNCars(), env.getStream(), invertOrange, normalize)
        , frameskip(frameskip)
    {
        if (frameskip < 1)
        {
            throw py::value_error("frameskip must be at least 1");
        }

        env.reset();
        io.packObs(env.getDeviceState());
        io.packState(env.getDeviceState());
        io.packTransitionState(env.getDeviceState(), 1);
        io.packRewardsDones(env.getDeviceState());
        io.packRawMatchState(env.getDeviceState());
    }

    py::object step(py::object actions)
    {
        py::capsule capsule = toCapsule(actions, env.getStream());
        const DLManagedTensor* actTensor = capsule.get_pointer<DLManagedTensor>();
        io.setActions(actionData(actTensor));

        for (int tick = 0; tick < frameskip; tick++)
        {
            env.step(io.getActions());
        }
        
        io.packRewardsDones(env.getDeviceState());
        io.packTransitionState(env.getDeviceState(), frameskip);
        io.packTransitionObs(env.getDeviceState());
        env.resetDones(io.getMaxTicks(), io.getNoTouchTimeoutTicks());
        io.packRawMatchState(env.getDeviceState());
        io.packObs(env.getDeviceState());
        io.packState(env.getDeviceState());

        return tensorToCapsule(io.getObsTensor());
    }

    void reset()
    {
        env.reset();
        io.packObs(env.getDeviceState());
        io.packState(env.getDeviceState());
        io.packTransitionState(env.getDeviceState(), 1);
        io.packRewardsDones(env.getDeviceState());
        io.packRawMatchState(env.getDeviceState());
    }


    void setBall(
        py::object position,
        py::object velocity,
        py::object angularVelocity,
        py::object simulationIndices)
    {
        int nSelected = env.getNSim();
        std::optional<TensorView> selection;
        const int64_t* selected = nullptr;
        if (!simulationIndices.is_none())
        {
            selection.emplace(selectionData(simulationIndices, nSelected));
            selected = static_cast<const int64_t*>(data(selection->tensor));
            nSelected = static_cast<int>(selection->tensor.shape[0]);
        }

        const auto pos = tensorData(
            position, "position", { nSelected, 3 }, kDLFloat, 32);

        const auto vel = tensorData(
            velocity, "velocity", { nSelected, 3 }, kDLFloat, 32);

        const auto ang = tensorData(
            angularVelocity, "angular_velocity", { nSelected, 3 }, kDLFloat, 32);

        io.setBall(env.getDeviceState(),
            static_cast<const float*>(data(pos.tensor)),
            static_cast<const float*>(data(vel.tensor)),
            static_cast<const float*>(data(ang.tensor)),
            selected, nSelected);

        io.packObs(env.getDeviceState());
        io.packState(env.getDeviceState());
        io.packTransitionState(env.getDeviceState(), 1);
        CUDA_CHECK(cudaStreamSynchronize(env.getStream()));
    }

    void setCar(
        py::object position,
        py::object rotation,
        py::object velocity,
        py::object angularVelocity,
        py::object demoed,
        py::object boost,
        py::object simulationIndices)
    {
        int nSelected = env.getNSim();
        std::optional<TensorView> selection;
        const int64_t* selected = nullptr;
        if (!simulationIndices.is_none())
        {
            selection.emplace(selectionData(simulationIndices, nSelected));
            selected = static_cast<const int64_t*>(data(selection->tensor));
            nSelected = static_cast<int>(selection->tensor.shape[0]);
        }

        const std::initializer_list<int64_t> vecShape = {
            nSelected, env.getNCars(), 3
        };

        const auto pos = tensorData(position, "position",
            vecShape, kDLFloat, 32);

        const auto rot = tensorData(rotation, "rotation",
            { nSelected, env.getNCars(), 4 }, kDLFloat, 32);

        const auto vel = tensorData(velocity, "velocity",
            vecShape, kDLFloat, 32);

        const auto ang = tensorData(angularVelocity, "angular_velocity",
            vecShape, kDLFloat, 32);

        py::capsule demoCapsule = toCapsule(demoed, env.getStream());
        const DLManagedTensor* demoTensor = demoCapsule.get_pointer<DLManagedTensor>();

        if (!demoTensor)
        {
            throw py::value_error("invalid DLPack tensor");
        }

        const DLTensor demo = demoTensor->dl_tensor;
        const bool byteDemoed = demo.dtype.code == kDLBool && demo.dtype.bits == 8;

        if (!byteDemoed && !(demo.dtype.code == kDLInt && demo.dtype.bits == 32))
        {
            throw py::type_error("demoed must have dtype bool or int32");
        }

        validateTensor(demo, "demoed",
            { nSelected, env.getNCars() }, demo.dtype.code, demo.dtype.bits);

        const float* boostData = nullptr;
        std::optional<TensorView> boostTensor;
        if (!boost.is_none())
        {
            boostTensor.emplace(tensorData(boost, "boost",
                { nSelected, env.getNCars() }, kDLFloat, 32));
            boostData = static_cast<const float*>(data(boostTensor->tensor));
        }

        io.setCar(env.getDeviceState(),
            static_cast<const float*>(data(pos.tensor)),
            static_cast<const float*>(data(rot.tensor)),
            static_cast<const float*>(data(vel.tensor)),
            static_cast<const float*>(data(ang.tensor)),
            data(demo), byteDemoed, boostData, selected, nSelected);

        io.packObs(env.getDeviceState());
        io.packState(env.getDeviceState());
        io.packTransitionState(env.getDeviceState(), 1);
        CUDA_CHECK(cudaStreamSynchronize(env.getStream()));
    }

    void setMatchState(py::object blueScore, py::object orangeScore,
                       py::object episodeTicks, py::object simulationIndices)
    {
        int nSelected = env.getNSim();
        std::optional<TensorView> selection;
        const int64_t* selected = nullptr;
        if (!simulationIndices.is_none())
        {
            selection.emplace(selectionData(simulationIndices, nSelected));
            selected = static_cast<const int64_t*>(data(selection->tensor));
            nSelected = static_cast<int>(selection->tensor.shape[0]);
        }
        const auto blue = tensorData(blueScore, "blue_score",
            { nSelected }, kDLInt, 32);
        const auto orange = tensorData(orangeScore, "orange_score",
            { nSelected }, kDLInt, 32);
        const auto ticks = tensorData(episodeTicks, "episode_ticks",
            { nSelected }, kDLInt, 32);
        io.setMatchState(env.getDeviceState(),
            static_cast<const int32_t*>(data(blue.tensor)),
            static_cast<const int32_t*>(data(orange.tensor)),
            static_cast<const int32_t*>(data(ticks.tensor)), selected, nSelected);
    }

    py::object getObs()
    {
        return tensorToCapsule(io.getObsTensor());
    }

    py::object getTransitionObs()
    {
        return tensorToCapsule(io.getTransitionObsTensor());
    }

    py::object getState()
    {
        return tensorToCapsule(io.getStateTensor());
    }

    py::object getTransitionState()
    {
        return tensorToCapsule(io.getTransitionStateTensor());
    }

    py::object getRewards()
    {
        return tensorToCapsule(io.getRewardsTensor());
    }

    py::object getDones()
    {
        return tensorToCapsule(io.getDonesTensor());
    }

    py::object getScoreDifference()
    {
        return tensorToCapsule(io.getScoreDifferenceTensor());
    }

    py::object getEpisodeTicks()
    {
        return tensorToCapsule(io.getEpisodeTicksTensor());
    }

    py::object getTransitionScoreDifference()
    {
        return tensorToCapsule(io.getTransitionScoreDifferenceTensor());
    }

    py::object getTransitionEpisodeTicks()
    {
        return tensorToCapsule(io.getTransitionEpisodeTicksTensor());
    }

    py::object getOvertime() { return tensorToCapsule(io.getOvertimeTensor()); }
    py::object getTransitionOvertime() { return tensorToCapsule(io.getTransitionOvertimeTensor()); }

    int getObsDim() const { return io.getObsDim(); }
    int getActDim() const { return io.getActDim(); }
    int getNSim()   const { return env.getNSim();  }
    int getNCars()  const { return env.getNCars(); }

    py::list getActionNvec() const
    {
        py::list cars;
        for (int c = 0; c < env.getNCars(); c++)
        {
            py::list nvec;
            for (int n : ACTION_NVECS) nvec.append(n);
            cars.append(nvec);
        }
        return cars;
    }

    void setMaxTicks(int ticks)
    {
        if (ticks < 1)
        {
            throw py::value_error("max_ticks must be at least 1");
        }
        io.setMaxTicks(ticks);
    }

    void setNoTouchTimeoutTicks(int ticks)
    {
        if (ticks < 0)
        {
            throw py::value_error("no_touch_timeout_ticks must be non-negative");
        }
        io.setNoTouchTimeoutTicks(ticks);
    }

    int getFrameskip() const { return frameskip; }

    void setFrameskip(int ticks)
    {
        if (ticks < 1)
        {
            throw py::value_error("frameskip must be at least 1");
        }
        frameskip = ticks;
    }
};

PYBIND11_MODULE(_carl, m)
{
    m.doc() = "CARL: CUDA Rocket League simulation";

    py::class_<EnvWrapper>(m, "Env")
        .def(py::init<int, int, int, int, int, bool, bool>(),
             py::arg("n_sim"), py::arg("n_blue"),
             py::arg("n_orange"), py::arg("seed"),
             py::arg("frameskip") = 1,
             py::arg("invert_orange") = true,
             py::arg("normalize") = false)

        .def("step",  &EnvWrapper::step, py::arg("actions"))
        .def("reset", &EnvWrapper::reset)

        .def("set_ball", &EnvWrapper::setBall,
             py::arg("position"),
             py::arg("velocity"),
             py::arg("angular_velocity"),
             py::arg("simulation_indices") = py::none())
        .def("set_car", &EnvWrapper::setCar,
             py::arg("position"),
             py::arg("rotation"),
             py::arg("velocity"),
             py::arg("angular_velocity"),
             py::arg("demoed"),
             py::arg("boost") = py::none(),
             py::arg("simulation_indices") = py::none())

        .def("get_obs",              &EnvWrapper::getObs)
        .def("get_transition_obs",   &EnvWrapper::getTransitionObs)
        .def("get_state",            &EnvWrapper::getState)
        .def("get_transition_state", &EnvWrapper::getTransitionState)
        .def("get_rewards",          &EnvWrapper::getRewards)
        .def("get_dones",            &EnvWrapper::getDones)
        .def("set_match_state",      &EnvWrapper::setMatchState,
             py::arg("blue_score"), py::arg("orange_score"),
             py::arg("episode_ticks"), py::arg("simulation_indices") = py::none())
        .def("get_score_difference", &EnvWrapper::getScoreDifference)
        .def("get_episode_ticks",    &EnvWrapper::getEpisodeTicks)
        .def("get_transition_score_difference", &EnvWrapper::getTransitionScoreDifference)
        .def("get_transition_episode_ticks", &EnvWrapper::getTransitionEpisodeTicks)
        .def("get_overtime", &EnvWrapper::getOvertime)
        .def("get_transition_overtime", &EnvWrapper::getTransitionOvertime)

        .def_property("max_ticks", nullptr, &EnvWrapper::setMaxTicks)
        .def_property("no_touch_timeout_ticks", nullptr,
                      &EnvWrapper::setNoTouchTimeoutTicks)

        .def_property("frameskip", &EnvWrapper::getFrameskip,
                      &EnvWrapper::setFrameskip)

        .def_property_readonly("obs_dim",     &EnvWrapper::getObsDim)
        .def_property_readonly("act_dim",     &EnvWrapper::getActDim)
        .def_property_readonly("action_nvec", &EnvWrapper::getActionNvec)
        .def_property_readonly("n_sim",       &EnvWrapper::getNSim)
        .def_property_readonly("n_cars",      &EnvWrapper::getNCars);
}
