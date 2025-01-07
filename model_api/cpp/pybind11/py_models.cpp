#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <opencv2/core/types.hpp>
#include <openvino/openvino.hpp>

#include "models/classification_model.h"
#include "models/results.h"

namespace py = pybind11;

namespace {
cv::Mat wrap_np_mat(const py::array_t<uint8_t>& input) {
    if (input.ndim() != 3 || input.shape(2) != 3) {
        throw std::runtime_error("Input image should have HWC_8U layout");
    }

    int height = input.shape(0);
    int width = input.shape(1);

    return cv::Mat(height, width, CV_8UC3, (void*)input.data());
}

ov::Any py_object_to_any(const py::object& py_obj, const std::string& property_name) {
    if (py::isinstance<py::str>(py_obj)) {
        return ov::Any(py_obj.cast<std::string>());
    } else if (py::isinstance<py::float_>(py_obj)) {
        return ov::Any(py_obj.cast<double>());
    } else if (py::isinstance<py::int_>(py_obj)) {
        return ov::Any(py_obj.cast<int>());
    } else {
        throw std::runtime_error("Property \"" + property_name + "\" has unsupported type.");
    }
}

}  // namespace

PYBIND11_MODULE(pybind11_model_api, m) {
    m.doc() = "Pybind11 binding for OpenVINO Vision API library";
    py::class_<ResultBase>(m, "ResultBase").def(py::init<>());

    py::class_<ClassificationResult::Classification>(m, "Classification")
        .def(py::init<unsigned int, const std::string, float>())
        .def_readwrite("id", &ClassificationResult::Classification::id)
        .def_readwrite("label", &ClassificationResult::Classification::label)
        .def_readwrite("score", &ClassificationResult::Classification::score);

    py::class_<ClassificationResult, ResultBase>(m, "ClassificationResult")
        .def(py::init<>())
        .def_readonly("topLabels", &ClassificationResult::topLabels)
        .def("__repr__", &ClassificationResult::operator std::string);

    py::class_<ModelBase>(m, "ModelBase")
        .def("load", [](ModelBase& self, const std::string& device, size_t num_infer_requests) {
            auto core = ov::Core();
            self.load(core, device, num_infer_requests);
        });

    py::class_<ImageModel, ModelBase>(m, "ImageModel");

    py::class_<ClassificationModel, ImageModel>(m, "ClassificationModel")
        .def_static(
            "create_model",
            [](const std::string& model_path,
               const std::map<std::string, py::object>& configuration,
               bool preload,
               const std::string& device) {
                auto ov_any_config = ov::AnyMap();
                for (const auto& item : configuration) {
                    ov_any_config[item.first] = py_object_to_any(item.second, item.first);
                }

                return ClassificationModel::create_model(model_path, ov_any_config, preload, device);
            },
            py::arg("model_path"),
            py::arg("configuration") = ov::AnyMap({}),
            py::arg("preload") = true,
            py::arg("device") = "AUTO")

        .def("__call__",
             [](ClassificationModel& self, const py::array_t<uint8_t>& input) {
                 return self.infer(wrap_np_mat(input));
             })
        .def("infer_batch", [](ClassificationModel& self, const std::vector<py::array_t<uint8_t>> inputs) {
            std::vector<ImageInputData> input_mats;
            input_mats.reserve(inputs.size());

            for (const auto& input : inputs) {
                input_mats.push_back(wrap_np_mat(input));
            }

            return self.inferBatch(input_mats);
        });
}
