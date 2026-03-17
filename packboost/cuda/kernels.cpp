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

} // namespace

// Forward declarations of CUDA implementations.
torch::Tensor encode_cuts(torch::Tensor X);

torch::Tensor h0_sm(
    torch::Tensor G,        // [N],    int16, cuda
    torch::Tensor L_old,    // [nfolds, Dm, N], uint8 branch bits
    int max_depth
);

torch::Tensor h0_sm_butterfly(
    torch::Tensor G,        // [N], int16, CUDA
    torch::Tensor L_old,    // [nfolds, Dm, N], uint8 branch bits, CUDA
    int max_depth
);

void repack_trees_for_features_cuda(
    const torch::Tensor& FST,     // uint8  [nsets, nfeatsets, max_depth]
    const torch::Tensor& L_old,   // uint8 [nfolds, Dm, N]
    torch::Tensor& LF,            // uint8 [nfeatsets, Dm, N] (output)
    int64_t tree_set      // which set (round)
);

torch::Tensor h_sm(
    torch::Tensor XS,
    torch::Tensor Y,
    torch::Tensor LF,
    int max_depth);

void cut_cuda_launcher(
    torch::Tensor F,      // uint16 (stores uint16)
    torch::Tensor FST,    // uint8
    torch::Tensor H,      // int64
    torch::Tensor H0,     // int64
    torch::Tensor V,      // int32
    torch::Tensor I,      // int16 (stores uint16)
    int tree_set,
    double L2,
    double lr,
    int qgrad_bits,
    int max_depth,
    double min_child_weight,
    double min_split_gain);


static void launch_advpred(
    torch::Tensor P,  // int32 [N]
    torch::Tensor X,  // int32 (bitwise uint32) [R, M]
    torch::Tensor L_old,
    torch::Tensor L_new,
    torch::Tensor V,  // int32 [rounds, K0, 2*nodes]
    torch::Tensor I,  // int16 [rounds, K0, nodes] (uint16 payload)
    int tree_set);

void advance_and_predict_launcher(
    torch::Tensor P, torch::Tensor X, torch::Tensor L_old, torch::Tensor L_new,
    torch::Tensor V, torch::Tensor I,
    int tree_set);

torch::Tensor h0_des_butterfly(
    torch::Tensor G,          // [N], int16, CUDA
    torch::Tensor L_old,      // [nfolds, Dm, N], uint8 branch bits, CUDA
    torch::Tensor era_ends,   // [E], int32, CUDA
    int max_depth
);

torch::Tensor h_des(
    torch::Tensor XS,        // [nfeatsets, cols_32M] uint32/int32
    torch::Tensor Y,         // [N]            int16
    torch::Tensor LF,        // [nfeatsets, Dm, N] uint8 branch bits
    torch::Tensor era_ends,  // [E]            int32 (exclusive ends)
    int max_depth);

void cut_des_cuda_launcher(
    torch::Tensor F,   // [rounds, 32*K1] uint16
    torch::Tensor FST, // [rounds, K1, D] uint8
    torch::Tensor H,   // [K1, E, nodes, 2, 32] int64
    torch::Tensor H0,  // [K0, nodes, E, 2]     int64
    torch::Tensor V,   // [rounds, K0, 2*nodes] int32
    torch::Tensor I,   // [rounds, K0, nodes]   uint16
    int tree_set,
    double L2,
    double lr,
    int qgrad_bits,
    int max_depth,
    double min_child_weight,
    double min_split_gain);

namespace {

torch::Tensor encode_cuts_binding(torch::Tensor X) {
    check_two_dim(X, "X");
    check_cuda_tensor(X, torch::kInt8, "X");

    auto X_contig = X.contiguous();
    return encode_cuts(X_contig);
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("encode_cuts", &encode_cuts_binding,
          "Encode cuts packed bitplanes",
          pybind11::arg("X"));
    
    m.def("h0_sm", &h0_sm, "H0 (featureless, unweighted) Murky-parity launcher");
    m.def("h0_sm_butterfly", &h0_sm_butterfly, "H0 butterfly reduce-scatter launcher");

    m.def("h0_des", &h0_des_butterfly, "H0 butterfly reduce-scatter launcher");

    m.def("repack_trees_for_features",
        &repack_trees_for_features_cuda,
        "Repack trees for features (Murky parity, CUDA)");
    m.def("h_sm", &h_sm, "PackBoost H (unweighted histogram, Murky parity; returns H)");
    
    m.def("h_des", &h_des, "PackBoost H (unweighted histogram, Murky parity; returns H)");
    
    m.def("cut_cuda", &cut_cuda_launcher, "PackBoost cut_cuda (CUDA)");

    m.def("cut_cuda_des", &cut_des_cuda_launcher, "PackBoost cut_cuda (CUDA)");
    
    m.def("advance_and_predict", &advance_and_predict_launcher,
        "PackBoost advance_and_predict (CUDA)");


}
