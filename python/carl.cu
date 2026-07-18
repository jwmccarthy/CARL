#include <cstring>
#include <initializer_list>
#include <string>
#include <pybind11/pybind11.h>

#include "RLEnvironment.cuh"
#include "EnvIO.cuh"

namespace py = pybind11;

// Accept a DLPack capsule or any object with __dlpack__
static py::capsule toCapsule(py::object obj)
{
    if (py::isinstance<py::capsule>(obj))
    {
        return obj.cast<py::capsule>();
    }

    return obj.attr("__dlpack__")().cast<py::capsule>();
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

// Holds env + I/O, DLPack tensors to Python are non-owning views over EnvIO buffers
class EnvWrapper
{
    RLEnvironment env;
    EnvIO io;
    int skipTicks;

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

    DLTensor tensorData(
        py::object obj,
        const char* name,
        std::initializer_list<int64_t> shape,
        uint8_t type,
        uint8_t bits) const
    {
        py::capsule capsule = toCapsule(obj);
        const DLManagedTensor* tensor = capsule.get_pointer<DLManagedTensor>();
        if (!tensor) throw py::value_error("invalid DLPack tensor");

        const DLTensor& t = tensor->dl_tensor;
        if (t.dtype.code != type || t.dtype.bits != bits || t.dtype.lanes != 1)
        {
            throw py::type_error(std::string(name) + " has an invalid dtype");
        }

        if (t.device.device_type != kDLCUDA)
        {
            throw py::value_error(std::string(name) + " must be on a CUDA device");
        }

        if (t.ndim != static_cast<int32_t>(shape.size()))
        {
            throw py::value_error(std::string(name) + " has an invalid shape");
        }

        int dim = 0;
        int64_t stride = 1;
        for (int64_t expected : shape)
        {
            if (t.shape[dim++] != expected)
            {
                throw py::value_error(std::string(name) + " has an invalid shape");
            }
        }
        if (t.strides)
        {
            for (int i = t.ndim - 1; i >= 0; i--)
            {
                if (t.strides[i] != stride)
                {
                    throw py::value_error(std::string(name) + " must be contiguous");
                }
                stride *= t.shape[i];
            }
        }
        return t;
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
        int skipTicks,
        bool invertOrange)
        : env(nSim, nBlue, nOrange, seed)
        , io(nSim, env.getNCars(), env.getStream(), invertOrange)
        , skipTicks(skipTicks)
    {
        if (skipTicks < 1)
        {
            throw py::value_error("skip_ticks must be at least 1");
        }

        env.reset();
        io.packObs(env.getDeviceState());
        io.packRewardsDones(env.getDeviceState());
    }

    py::object step(py::object actions)
    {
        py::capsule capsule = toCapsule(actions);
        const DLManagedTensor* actTensor = capsule.get_pointer<DLManagedTensor>();
        io.setActions(actionData(actTensor));

        for (int tick = 0; tick < skipTicks; tick++)
        {
            env.step(io.getActions());
        }
        
        io.packRewardsDones(env.getDeviceState(), skipTicks);
        env.resetDones(io.getMaxTicks());
        io.packObs(env.getDeviceState());

        return tensorToCapsule(io.getObsTensor());
    }

    void reset()
    {
        env.reset();
        io.packObs(env.getDeviceState());
        io.packRewardsDones(env.getDeviceState());
    }


    void setBall(py::object position, py::object velocity, py::object angularVelocity)
    {
        const auto pos = tensorData(
            position, "position", { env.getNSim(), 3 }, kDLFloat, 32);

        const auto vel = tensorData(
            velocity, "velocity", { env.getNSim(), 3 }, kDLFloat, 32);

        const auto ang = tensorData(
            angularVelocity, "angular_velocity", { env.getNSim(), 3 }, kDLFloat, 32);

        io.setBall(env.getDeviceState(),
            static_cast<const float*>(data(pos)),
            static_cast<const float*>(data(vel)),
            static_cast<const float*>(data(ang)));

        io.packObs(env.getDeviceState());
    }

    void setCar(
        py::object position,
        py::object rotation,
        py::object velocity,
        py::object angularVelocity,
        py::object demoed)
    {
        const std::initializer_list<int64_t> vecShape = {
            env.getNSim(), env.getNCars(), 3
        };

        const auto pos = tensorData(position, "position",
            vecShape, kDLFloat, 32);

        const auto rot = tensorData(rotation, "rotation",
            { env.getNSim(), env.getNCars(), 4 }, kDLFloat, 32);

        const auto vel = tensorData(velocity, "velocity",
            vecShape, kDLFloat, 32);

        const auto ang = tensorData(angularVelocity, "angular_velocity",
            vecShape, kDLFloat, 32);

        py::capsule demoCapsule = toCapsule(demoed);
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

        tensorData(demoed, "demoed", 
            { env.getNSim(), env.getNCars() }, demo.dtype.code, demo.dtype.bits);

        io.setCar(env.getDeviceState(),
            static_cast<const float*>(data(pos)),
            static_cast<const float*>(data(rot)),
            static_cast<const float*>(data(vel)),
            static_cast<const float*>(data(ang)),
            data(demo), byteDemoed);

        io.packObs(env.getDeviceState());
    }

    py::object getObs()     { return tensorToCapsule(io.getObsTensor()); }
    py::object getRewards() { return tensorToCapsule(io.getRewardsTensor()); }
    py::object getTouches() { return tensorToCapsule(io.getTouchesTensor()); }
    py::object getDones()   { return tensorToCapsule(io.getDonesTensor()); }

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

    void setMaxTicks(int ticks) { io.setMaxTicks(ticks); }
    int getSkipTicks() const { return skipTicks; }

    void setSkipTicks(int ticks)
    {
        if (ticks < 1)
        {
            throw py::value_error("skip_ticks must be at least 1");
        }
        skipTicks = ticks;
    }
};

PYBIND11_MODULE(_carl, m)
{
    m.doc() = "CARL: CUDA Rocket League simulation";

    py::class_<EnvWrapper>(m, "Env")
        .def(py::init<int, int, int, int, int, bool>(),
             py::arg("n_sim"), py::arg("n_blue"),
             py::arg("n_orange"), py::arg("seed"),
             py::arg("skip_ticks") = 1,
             py::arg("invert_orange") = true)

        .def("step",  &EnvWrapper::step, py::arg("actions"))
        .def("reset", &EnvWrapper::reset)

        .def("set_ball", &EnvWrapper::setBall,
             py::arg("position"), py::arg("velocity"),
             py::arg("angular_velocity"))
        .def("set_car", &EnvWrapper::setCar,
             py::arg("position"), py::arg("rotation"),
             py::arg("velocity"), py::arg("angular_velocity"),
             py::arg("demoed"))

        .def("get_obs",          &EnvWrapper::getObs)
        .def("get_rewards",      &EnvWrapper::getRewards)
        .def("get_ball_touches", &EnvWrapper::getTouches)
        .def("get_dones",        &EnvWrapper::getDones)

        .def_property("max_ticks", nullptr, &EnvWrapper::setMaxTicks)
        .def_property("skip_ticks", &EnvWrapper::getSkipTicks,
                      &EnvWrapper::setSkipTicks)

        .def_property_readonly("obs_dim",     &EnvWrapper::getObsDim)
        .def_property_readonly("act_dim",     &EnvWrapper::getActDim)
        .def_property_readonly("action_nvec", &EnvWrapper::getActionNvec)
        .def_property_readonly("n_sim",       &EnvWrapper::getNSim)
        .def_property_readonly("n_cars",      &EnvWrapper::getNCars);
}
