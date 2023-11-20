#include "jlcxx/jlcxx.hpp"
#include "jlcxx/stl.hpp"
#include "ghostbasil/optimization/basil.hpp"
#include "ghostbasil/matrix/block_matrix.hpp"
#include "ghostbasil/matrix/block_group_ghost_matrix.hpp"

static auto init_eigen_vec(jlcxx::ArrayRef<double, 1> vec) {
    Eigen::VectorXd eigen_vec(vec.size());
    size_t k = 0;
    for (size_t i = 0; i < vec.size(); i++) {
        eigen_vec(i) = vec[k++];
    }
    return eigen_vec;
}

static auto init_eigen_mat(jlcxx::ArrayRef<double, 2> mat, int_t rows, int_t cols) {
    Eigen::MatrixXd eigen_mat_(rows, cols);
    size_t k = 0;
    for (size_t i = 0; i < cols; i++) {
        for (size_t j = 0; j < rows; j++) {
            eigen_mat_(j, i) = mat[k++];
        }
    }
    return eigen_mat_;
}

// this is BlockMatrix type in ghostbasil but only accepts 1 (dense) block
class OneBlockMatrixWrap
{
    jlcxx::ArrayRef<double, 2> orig_mat_;
    Eigen::MatrixXd eigen_mat_;
    std::vector<Eigen::Map<Eigen::MatrixXd>> mat_list_;
    ghostbasil::BlockMatrix<Eigen::Map<Eigen::MatrixXd>> block_mat_;
    const Eigen::Array<size_t, 2, 1> dim_;

    static auto init_mat_list(Eigen::MatrixXd mat, int_t rows, int_t cols) {
        std::vector<Eigen::Map<Eigen::MatrixXd>> mat_list_;
        mat_list_.emplace_back(Eigen::Map<Eigen::MatrixXd>(mat.data(), rows, cols));
        return mat_list_;
    }

public: 
    OneBlockMatrixWrap(jlcxx::ArrayRef<double, 2> mat, int_t rows, int_t cols)
        : orig_mat_(mat),
        eigen_mat_(init_eigen_mat(mat, rows, cols)),
        mat_list_(init_mat_list(eigen_mat_, rows, cols)),
        block_mat_(mat_list_),
        dim_(block_mat_.rows(), block_mat_.cols())
    {}

    // GHOSTBASIL_STRONG_INLINE
    const auto& internal() const { return block_mat_; }

    // For export only
    const Eigen::Array<size_t, 2, 1> dim_exp() const { return dim_; }
};

// adapted from https://github.com/JamesYang007/ghostbasil/blob/master/R/inst/include/rcpp_block_group_ghost_matrix.hpp#L7
class BlockGroupGhostMatrixWrap
{
public:
    using bmat_t = ghostbasil::BlockMatrix<Eigen::Map<Eigen::MatrixXd>>;

private:
    jlcxx::ArrayRef<double, 2> orig_mat_S_;
    jlcxx::ArrayRef<double, 2> orig_mat_D_;
    Eigen::MatrixXd eigen_mat_S_;
    Eigen::MatrixXd eigen_mat_D_;
    const Eigen::Map<Eigen::MatrixXd> orig_S_;
    const OneBlockMatrixWrap orig_D_;
    ghostbasil::BlockGroupGhostMatrix<Eigen::MatrixXd, bmat_t> gmat_;
    const Eigen::Array<size_t, 2, 1> dim_;

public:
    BlockGroupGhostMatrixWrap(
        jlcxx::ArrayRef<double, 2> S, // standard matrix from Julia
        jlcxx::ArrayRef<double, 2> D, // this will be converted to a BlockMatrix
        int_t m, // m is number of knockoffs
        size_t rows, // size(D, 1)
        size_t cols  // size(D, 2)
    )
        : orig_mat_S_(S),
          orig_mat_D_(D),
          eigen_mat_S_(init_eigen_mat(S, rows, cols)),
          eigen_mat_D_(init_eigen_mat(D, rows, cols)),
          orig_S_(Eigen::Map<Eigen::MatrixXd>(eigen_mat_S_.data(), rows, cols)),
          orig_D_(orig_mat_D_, rows, cols),
          gmat_(orig_S_, 
                orig_D_.internal(), 
                m + 1), // in ghostbasil, the original variables counts as 1 knockoff
          dim_(gmat_.rows(), gmat_.cols())
    {}

    // GHOSTBASIL_STRONG_INLINE
    const auto& internal() const { return gmat_; }

    // For export only
    const Eigen::Array<size_t, 2, 1> dim_exp() const { return dim_; }
};

// Same function as `basil__` at 
// https://github.com/JamesYang007/ghostbasil/blob/master/R/src/rcpp_basil.cpp#L88
// Except: (a) this returns only the beta that corresponds to the last lambda value
// and (b) there is a lot less "checking" (so its behavior is unpredictable
// when user early terminates)
template <class AType>
std::vector<double> basil__(
        const AType& A, 
        const Eigen::Map<Eigen::VectorXd> r,
        double alpha,
        const Eigen::Map<Eigen::VectorXd> penalty,
        const Eigen::Map<Eigen::VectorXd> user_lmdas,
        size_t max_n_lambdas,
        size_t n_lambdas_iter,
        bool use_strong_rule,
        bool do_early_exit,
        size_t delta_strong_size,
        size_t max_strong_size,
        size_t max_n_cds,
        double thr,
        double min_ratio,
        size_t n_threads)
{
    using namespace ghostbasil::lasso;

    std::vector<Eigen::SparseVector<double>> betas;
    std::vector<double> lmdas;
    std::vector<double> rsqs;

    // slight optimization: reserve spaces ahead of time
    const size_t capacity = std::max(
        max_n_lambdas, static_cast<size_t>(user_lmdas.size())
    );
    betas.reserve(capacity);
    lmdas.reserve(capacity);
    rsqs.reserve(capacity);
    std::string error;

    try {
        basil(
                A, r, alpha, penalty, user_lmdas, max_n_lambdas, n_lambdas_iter,
                use_strong_rule, do_early_exit, delta_strong_size, max_strong_size, max_n_cds, thr, 
                min_ratio, n_threads,
                betas, lmdas, rsqs);
    }
    catch (const std::exception& e) {
        error = e.what();
    }

    // return the beta corresponding to the last lambda value
    Eigen::SparseVector<double> last_beta = betas.back();
    std::vector<double> dense_last_beta(last_beta.size(), 0.0);
    for (Eigen::SparseVector<double>::InnerIterator it(last_beta); it; ++it) {
        dense_last_beta[it.index()] = it.value();
    }
    return dense_last_beta;
}

// Same as `basil_block_group_ghost__` at
// https://github.com/JamesYang007/ghostbasil/blob/master/R/src/rcpp_basil.cpp#L291
// But we don't create a `BlockGroupGhostMatrix` in Julia. Instead, we
// pass `C` and `S` as Julia matrices into C++, convert `S` to a `BlockMatrix`
// with a single block, convert `C` to a Eigen matrix, and then pass `C` and `S`
// directly into `basil__`
std::vector<double> basil_block_group_ghost__(
        const jlcxx::ArrayRef<double, 2> C, // dense matrix
        const jlcxx::ArrayRef<double, 2> S, // becomes a BlockMatrix
        const jlcxx::ArrayRef<double, 1> r,
        const jlcxx::ArrayRef<double, 1> user_lmdas,
        int_t m, // number of knockoffs
        size_t p, // dimension of S and C (both square matrices)
        size_t max_n_lambdas,
        size_t n_lambdas_iter,
        bool use_strong_rule,
        bool do_early_exit,
        size_t delta_strong_size,
        size_t max_strong_size,
        size_t max_n_cds,
        double thr,
        double min_ratio,
        size_t n_threads)
{
    // convert C and S to instance of BlockGroupGhostMatrix
    auto gmw = BlockGroupGhostMatrixWrap(C, S, m, p, p);
    const auto& gm = gmw.internal();

    // test internal BlockGroupGhostMatrix is correct
    std::cout << "testing A has S and D matrices internally" << '\n';
    const auto& C2 = gm.get_S(); // first input to BlockGroupGhostMatrix (dense)
    std::cout << "C2.size() = " << C2.size() << '\n';
    std::cout << "C2.coeff(0, 0) = " << C2.coeff(0, 0) << '\n';
    std::cout << "C2.coeff(0, 1) = " << C2.coeff(0, 1) << '\n';
    std::cout << "C2.coeff(1, 1) = " << C2.coeff(1, 1) << '\n';
    const auto& S2 = gm.get_D(); // second input to BlockGroupGhostMatrix (a BlockMatrix)
    std::cout << "S2.size() = " << S2.size() << '\n';
    std::cout << "S2.coeff(0, 0) = " << S2.coeff(0, 0) << '\n';
    std::cout << "S2.coeff(0, 1) = " << S2.coeff(0, 1) << '\n';
    std::cout << "S2.coeff(1, 1) = " << S2.coeff(1, 1) << '\n';

    // test overall BlockGroupGhostMatrix is correct
    std::cout << "testing BlockGroupGhostMatrix" << '\n';
    std::cout << "gm.rows() = " << gm.rows() << '\n';
    std::cout << "gm.cols() = " << gm.cols() << '\n';
    std::cout << "gm.size() = " << gm.size() << '\n';
    std::cout << "gm.coeff(0, 0) = " << gm.coeff(0, 0) << '\n'; // should be 1.01
    std::cout << "gm.coeff(0, 1) = " << gm.coeff(0, 1) << '\n'; // should be 0.782109
    std::cout << "gm.coeff(1, 1) = " << gm.coeff(1, 1) << '\n'; // should be 1.01
    std::cout << "gm.coeff(p, p) = " << gm.coeff(p, p) << '\n'; // should be 1.01
    std::cout << "gm.coeff(p, p+1) = " << gm.coeff(p, p+1) << '\n'; // should be 0.782109
    std::cout << "gm.coeff(p+1, p+1) = " << gm.coeff(p+1, p+1) << '\n'; // should be 1.01


    // default alpha = 1.0 and penalty = vector of 1s
    double alpha = 1.0;
    Eigen::VectorXd ones = Eigen::VectorXd::Ones(p);
    Eigen::Map<Eigen::VectorXd> penalty(ones.data(), ones.size());

    // convert r and user_lmdas to Eigen::Map<Eigen::VectorXd>
    Eigen::VectorXd lmdas = init_eigen_vec(user_lmdas);
    Eigen::VectorXd r1 = init_eigen_vec(r);
    Eigen::Map<Eigen::VectorXd> lambdas(lmdas.data(), lmdas.size());
    Eigen::Map<Eigen::VectorXd> r2(r1.data(), r1.size());

    std::vector<double> result = basil__(
            gm, r2, alpha, penalty, lambdas, max_n_lambdas,
            n_lambdas_iter, use_strong_rule, do_early_exit, delta_strong_size,
            max_strong_size, max_n_cds, thr, min_ratio, n_threads);

    std::cout << "after convergence, testing BlockGroupGhostMatrix again" << '\n';
    std::cout << "C2.coeff(0, 0) = " << C2.coeff(0, 0) << '\n';
    std::cout << "C2.coeff(0, 1) = " << C2.coeff(0, 1) << '\n';
    std::cout << "C2.coeff(1, 1) = " << C2.coeff(1, 1) << '\n';
    std::cout << "S2.coeff(0, 0) = " << S2.coeff(0, 0) << '\n';
    std::cout << "S2.coeff(0, 1) = " << S2.coeff(0, 1) << '\n';
    std::cout << "S2.coeff(1, 1) = " << S2.coeff(1, 1) << '\n';
    std::cout << "gm.coeff(0, 0) = " << gm.coeff(0, 0) << '\n'; // should be 1.01
    std::cout << "gm.coeff(0, 1) = " << gm.coeff(0, 1) << '\n'; // should be 0.782109
    std::cout << "gm.coeff(1, 1) = " << gm.coeff(1, 1) << '\n'; // should be 1.01
    std::cout << "gm.coeff(p, p) = " << gm.coeff(p, p) << '\n'; // should be 1.01
    std::cout << "gm.coeff(p, p+1) = " << gm.coeff(p, p+1) << '\n'; // should be 0.782109
    std::cout << "gm.coeff(p+1, p+1) = " << gm.coeff(p+1, p+1) << '\n'; // should be 1.01

    return result;
}

JLCXX_MODULE define_julia_module(jlcxx::Module& mod)
{
    mod.method("block_group_ghostbasil", &basil_block_group_ghost__);
}
