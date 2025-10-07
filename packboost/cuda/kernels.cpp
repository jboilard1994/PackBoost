#include <torch/extension.h>

namespace {

void check_cuda_tensor(const torch::Tensor& tensor,
                       torch::ScalarType dtype,
                       const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.dtype() == dtype,
                name, " must have dtype ", dtype, ", got ", tensor.dtype());
}

void check_two_dim(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.dim() == 2,
                name, " must be a 2D tensor, got ", tensor.dim(), " dims");
}

torch::Tensor encode_cuts_binding(torch::Tensor X);
torch::Tensor et_sample_1b_binding(torch::Tensor X,
                                   torch::Tensor XS,
                                   torch::Tensor Fsch,
                                   int round);

} // namespace

// Forward declarations of CUDA implementations.
torch::Tensor encode_cuts(torch::Tensor X);
torch::Tensor et_sample_1b(torch::Tensor X,
                           torch::Tensor XS,
                           torch::Tensor Fsch,
                           int round);

namespace {

torch::Tensor encode_cuts_binding(torch::Tensor X) {
    check_two_dim(X, "X");
    check_cuda_tensor(X, torch::kInt8, "X");

    auto X_contig = X.contiguous();
    return encode_cuts(X_contig);
}

torch::Tensor et_sample_1b_binding(torch::Tensor X,
                                   torch::Tensor XS,
                                   torch::Tensor Fsch,
                                   int round) {
    check_two_dim(X, "X");
    check_two_dim(XS, "XS");
    check_two_dim(Fsch, "Fsch");

    check_cuda_tensor(X, torch::kUInt32, "X");
    check_cuda_tensor(XS, torch::kUInt32, "XS");
    check_cuda_tensor(Fsch, torch::kUInt16, "Fsch");

    TORCH_CHECK(round >= 0 && round < Fsch.size(0),
                "round index ", round, " is out of bounds for Fsch with size ", Fsch.size(0));
    TORCH_CHECK(Fsch.size(1) % 32 == 0,
                "Fsch second dimension must be a multiple of 32, got ", Fsch.size(1));
    TORCH_CHECK(XS.size(0) * 32 == Fsch.size(1),
                "XS first dimension (", XS.size(0),
                ") must match Fsch second dimension / 32 (", Fsch.size(1) / 32, ")");
    TORCH_CHECK(XS.size(1) == X.size(1) * 32,
                "XS second dimension must equal X.size(1) * 32");

    auto X_contig = X.contiguous();
    auto XS_contig = XS.contiguous();
    auto Fsch_contig = Fsch.contiguous();

    return et_sample_1b(X_contig, XS_contig, Fsch_contig, round);
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("encode_cuts", &encode_cuts_binding,
          "Encode cuts packed bitplanes",
          pybind11::arg("X"));

    m.def("et_sample_1b", &et_sample_1b_binding,
          "Extract and rotate feature samples",
          pybind11::arg("X"),
          pybind11::arg("XS"),
          pybind11::arg("Fsch"),
          pybind11::arg("round"));
}
