#include <cstring>
#include <pybind11/pybind11.h>

#include "RLEnvironment.cuh"
#include "EnvIO.cuh"

namespace py = pybind11;

// Accept a DLPack capsule or any object with __dlpack__ (e.g. torch.Tensor)
static const DLManagedTensor* capsuleToTensor(py::object obj)
{
    if (py::isinstance<py::capsule>(obj))
    {
        auto* t = obj.cast<py::capsule>().get_pointer<DLManagedTensor>();
        if (t) return t;
    }

    auto dlpack = obj.attr("__dlpack__")();
    return dlpack.cast<py::capsule>().get_pointer<DLManagedTensor>();
}

// Only delete if still named 'dltensor' - consumers rename to prevent double-free
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

public:
    EnvWrapper(int nSim, int nBlue, int nOrange, int seed)
        : env(nSim, nBlue, nOrange, seed)
        , io(nSim, env.getNCars(), env.getStream())
    {
        env.reset();
        io.packObs(env.getDeviceState());
        io.packRewardsDones(env.getDeviceState());
    }

    py::object step(py::object actions)
    {
        const DLManagedTensor* actTensor = capsuleToTensor(actions);
        io.setActions((const float*)actTensor->dl_tensor.data);

        io.unpackActions(env.getDeviceState());

        env.step();
        
        io.packObs(env.getDeviceState());
        io.packRewardsDones(env.getDeviceState());

        return tensorToCapsule(io.getObsTensor());
    }

    void reset()
    {
        env.reset();
        io.packObs(env.getDeviceState());
        io.packRewardsDones(env.getDeviceState());
    }

    py::object getObs()     { return tensorToCapsule(io.getObsTensor()); }
    py::object getRewards() { return tensorToCapsule(io.getRewardsTensor()); }
    py::object getDones()   { return tensorToCapsule(io.getDonesTensor()); }

    int getObsDim() const { return io.getObsDim(); }
    int getActDim() const { return io.getActDim(); }
    int getNSim()   const { return env.getNSim();  }
    int getNCars()  const { return env.getNCars(); }
    void setMaxTicks(int ticks) { io.setMaxTicks(ticks); }
};

PYBIND11_MODULE(carl, m)
{
    m.doc() = "CARL: CUDA Rocket League simulation";

    py::class_<EnvWrapper>(m, "Env")
        .def(py::init<int, int, int, int>(),
             py::arg("n_sim"), py::arg("n_blue"),
             py::arg("n_orange"), py::arg("seed"))

        .def("step",        &EnvWrapper::step, py::arg("actions"))
        .def("reset",       &EnvWrapper::reset)
        .def("get_obs",     &EnvWrapper::getObs)
        .def("get_rewards", &EnvWrapper::getRewards)
        .def("get_dones",   &EnvWrapper::getDones)

        .def_property("max_ticks", nullptr, &EnvWrapper::setMaxTicks)
        .def_property_readonly("obs_dim",   &EnvWrapper::getObsDim)
        .def_property_readonly("act_dim",   &EnvWrapper::getActDim)
        .def_property_readonly("n_sim",     &EnvWrapper::getNSim)
        .def_property_readonly("n_cars",    &EnvWrapper::getNCars);
}
