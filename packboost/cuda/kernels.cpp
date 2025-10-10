#include <torch/extension.h>
#include <tuple>


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

std::tuple<torch::Tensor, torch::Tensor>
prep_vars(torch::Tensor L, 
            torch::Tensor Y,  
            torch::Tensor P);

torch::Tensor h0_sm(
    torch::Tensor G,        // [N],    int16, cuda
    torch::Tensor LE,       // [nfolds, N], int16/int32/int64 (signed storage)
    int max_depth
);

torch::Tensor h0_sm_butterfly(
    torch::Tensor G,        // [N], int16, CUDA
    torch::Tensor LE,       // [nfolds, N], (u)int16/32/64, CUDA
    int max_depth
);

void repack_trees_for_features_cuda(
    const torch::Tensor& FST,     // uint8  [nsets, nfeatsets, max_depth]
    const torch::Tensor& LE,      // u16/u32/u64 [nfolds, N]
    torch::Tensor& LF,            // u16/u32/u64 [nfeatsets, N] (output, in-place fill)
    int64_t tree_set,      // which set (round)
);

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
    
    m.def("prep_vars", &prep_vars, "L, Y, P -> LE, G");
    m.def("h0_sm", &h0_sm, "H0 (featureless, unweighted) Murky-parity launcher");
    m.def("h0_sm_butterfly", &h0_sm_butterfly, "H0 butterfly reduce-scatter launcher");
    m.def("repack_trees_for_features",
        &repack_trees_for_features_cuda,
        "Repack trees for features (Murky parity, CUDA)");

}
