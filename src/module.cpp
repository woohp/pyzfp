#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>
#if PY_VERSION_HEX >= 0x03000000
#define PyInt_Check PyLong_Check
#define PyInt_AsLong PyLong_AsLong
#endif
#include "zfp.h"
#include <pybind11/numpy.h>
#include <iostream>

using namespace std;
namespace py = pybind11;


enum class CompressionMode
{
    FixedRate,
    FixedPrecision,
    FixedAccuracy
};


py::bytes compress(
    py::array_t<float, py::array::c_style> input, CompressionMode compression_mode, double rate, unsigned int precision, double tolerance)
{
    py::buffer_info info = input.request();
    float* ptr = reinterpret_cast<float*>(info.ptr);

    zfp_field* field;
    if (input.ndim() == 2)
        field = zfp_field_2d(ptr, zfp_type_float, input.shape(1), input.shape(0));
    else if (input.ndim() == 3)
        field = zfp_field_3d(ptr, zfp_type_float, input.shape(2), input.shape(1), input.shape(0));
    else if (input.ndim() == 4)
        field = zfp_field_4d(ptr, zfp_type_float, input.shape(3), input.shape(2), input.shape(1), input.shape(0));
    else
        field = zfp_field_1d(ptr, zfp_type_float, info.size);
    zfp_stream* zfp = zfp_stream_open(nullptr);

    if (compression_mode == CompressionMode::FixedRate)
        zfp_stream_set_rate(zfp, rate, zfp_type_float, 1, 0);
    else if (compression_mode == CompressionMode::FixedPrecision)
        zfp_stream_set_precision(zfp, precision);
    else if (compression_mode == CompressionMode::FixedAccuracy)
        zfp_stream_set_accuracy(zfp, tolerance);
    size_t bufsize = zfp_stream_maximum_size(zfp, field);
    char* buffer = new char[bufsize];
    bitstream* stream = stream_open(buffer, bufsize);
    zfp_stream_set_bit_stream(zfp, stream);
    zfp_stream_rewind(zfp);
    size_t size = zfp_compress(zfp, field);
    if (size == 0)
    {
        delete[] buffer;
        throw std::runtime_error("compression failed");
    }

    py::bytes out(buffer, size);

    zfp_field_free(field);
    zfp_stream_close(zfp);
    stream_close(stream);
    delete[] buffer;

    return out;
}


void decompress(
    py::buffer b,
    py::array_t<float, py::array::c_style> out,
    CompressionMode compression_mode,
    double rate,
    unsigned int precision,
    double tolerance)
{
    py::buffer_info compressed_info = b.request();
    py::buffer_info decompressed_info = out.request();
    float* ptr = reinterpret_cast<float*>(decompressed_info.ptr);

    zfp_field* field;
    if (out.ndim() == 2)
        field = zfp_field_2d(ptr, zfp_type_float, out.shape(1), out.shape(0));
    else if (out.ndim() == 3)
        field = zfp_field_3d(ptr, zfp_type_float, out.shape(2), out.shape(1), out.shape(0));
    else if (out.ndim() == 4)
        field = zfp_field_4d(ptr, zfp_type_float, out.shape(3), out.shape(2), out.shape(1), out.shape(0));
    else
        field = zfp_field_1d(ptr, zfp_type_float, decompressed_info.size);
    zfp_stream* zfp = zfp_stream_open(nullptr);
    if (compression_mode == CompressionMode::FixedRate)
        zfp_stream_set_rate(zfp, rate, zfp_type_float, 1, 0);
    else if (compression_mode == CompressionMode::FixedPrecision)
        zfp_stream_set_precision(zfp, precision);
    else if (compression_mode == CompressionMode::FixedAccuracy)
        zfp_stream_set_accuracy(zfp, tolerance);
    bitstream* stream = stream_open(compressed_info.ptr, compressed_info.size);
    zfp_stream_set_bit_stream(zfp, stream);
    zfp_stream_rewind(zfp);
    if (!zfp_decompress(zfp, field))
        throw std::runtime_error("decompression failed");

    zfp_field_free(field);
    zfp_stream_close(zfp);
    stream_close(stream);
}


bool import_numpy()
{
    // wacky NumPy API: import_array1() is a macro which actually calls return with the given value
    import_array1(false);
    return true;
}


PYBIND11_MODULE(zfp, m)
{
    using namespace pybind11::literals;

    import_numpy();

    m.doc() = "Zfp compression";

    py::enum_<CompressionMode>(m, "CompressionMode")
        .value("FixedRate", CompressionMode::FixedRate)
        .value("FixedPrecision", CompressionMode::FixedPrecision)
        .value("FixedAccuracy", CompressionMode::FixedAccuracy);

    const auto compress_docstring = R"(Compress a float array using zfp)";
    m.def(
        "compress",
        &compress,
        compress_docstring,
        "input"_a,
        "compression_mode"_a,
        "rate"_a = 8.0,
        "precision"_a = 8,
        "tolerance"_a = 0.001);

    const auto decompress_docstring = R"(Decompress some bytes into a float array)";
    m.def(
        "decompress",
        &decompress,
        decompress_docstring,
        "compressed_bytes"_a,
        "out"_a,
        "compression_mode"_a,
        "rate"_a = 8.0,
        "precision"_a = 8,
        "tolerance"_a = 0.001);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#endif
}
