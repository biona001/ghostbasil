#include <gtest/gtest.h>
#include <testutil/ghost_matrix_util.hpp>

namespace ghostbasil {
namespace {

using namespace ghost_matrix_util;

static constexpr double tol = 1e-14; 

// ========================================================
// TEST Constructor
// ========================================================

TEST(GhostMatrixTest, ctor_empty_list)
{
    mat_list_t ml;
    vec_list_t vl;
    EXPECT_THROW(gmat_t m(ml, vl, 1), std::runtime_error); 
}

TEST(GhostMatrixTest, ctor_diff_list_size)
{
    mat_t B(1,1);
    vec_t D(1);
    mat_list_t ml(2, B);
    vec_list_t vl(3, D);
    EXPECT_THROW(gmat_t m(ml, vl, 1), std::runtime_error); 
}

TEST(GhostMatrixTest, ctor_no_knockoff)
{
    mat_t B(1,1);
    vec_t D(1);
    mat_list_t ml(2, B);
    vec_list_t vl(2, D);
    EXPECT_THROW(gmat_t m(ml, vl, 0), std::runtime_error); 
}

TEST(GhostMatrixTest, ctor_diff_B_D_size)
{
    mat_t B(2,2);
    vec_t D(1);
    mat_list_t ml(2, B);
    vec_list_t vl(2, D);
    EXPECT_THROW(gmat_t m(ml, vl, 0), std::runtime_error); 
}

TEST(GhostMatrixTest, ctor_B_not_square)
{
    mat_t B(2,3);
    vec_t D(2);
    mat_list_t ml(2, B);
    vec_list_t vl(2, D);
    EXPECT_THROW(gmat_t m(ml, vl, 0), std::runtime_error); 
}

TEST(GhostMatrixTest, ctor_valid)
{
    mat_t B(2,2);
    vec_t D(2);
    mat_list_t ml(2, B);
    vec_list_t vl(2, D);
    EXPECT_NO_THROW(gmat_t m(ml, vl, 1)); 
}

// ========================================================
// TEST col_dot
// ========================================================

struct GhostMatrixColDotFixture
    : testing::Test,
      testing::WithParamInterface<
        std::tuple<size_t, size_t, size_t, size_t> >
{};

template <class VecType>
static inline void test_col_dot(
        const gmat_t& gmat,
        const mat_t& dense,
        const VecType& v)
{
    for (size_t i = 0; i < dense.cols(); ++i) {
        auto actual = gmat.col_dot(i, v);
        auto expected = v.dot(dense.col(i));
        EXPECT_NEAR(actual, expected, tol);
    }
}

TEST_P(GhostMatrixColDotFixture, col_dot)
{
    auto param = GetParam();
    size_t seed = std::get<0>(param);
    size_t L = std::get<1>(param);
    size_t p = std::get<2>(param);
    size_t n_knockoffs = std::get<3>(param);

    auto out = generate_data(seed, L, p, n_knockoffs);
    auto& ml = std::get<0>(out);
    auto& vl = std::get<1>(out);
    auto& v = std::get<2>(out);
    auto& vs = std::get<3>(out);
    auto& dense = std::get<4>(out);

    gmat_t gmat(ml, vl, n_knockoffs);

    test_col_dot(gmat, dense, v);
    test_col_dot(gmat, dense, vs);
}

INSTANTIATE_TEST_SUITE_P(
        GhostMatrixColDotSuite,
        GhostMatrixColDotFixture,
        testing::Values(
            std::make_tuple(0, 1, 4, 1),
            std::make_tuple(124, 2, 3, 2),
            std::make_tuple(321, 5, 1, 1),
            std::make_tuple(9382, 3, 9, 2))
    );

// ========================================================
// TEST quad_form
// ========================================================

struct GhostMatrixQuadFormFixture
    : testing::Test,
      testing::WithParamInterface<
        std::tuple<size_t, size_t, size_t, size_t> >
{};

template <class VecType>
static inline void test_quad_form(
        const gmat_t& gmat,
        const mat_t& dense,
        const VecType& v)
{
    auto actual = gmat.quad_form(v);
    auto expected = v.dot(dense * v);
    EXPECT_NEAR(actual, expected, tol);
}

TEST_P(GhostMatrixQuadFormFixture, quad_form)
{
    auto param = GetParam();
    size_t seed = std::get<0>(param);
    size_t L = std::get<1>(param);
    size_t p = std::get<2>(param);
    size_t n_knockoffs = std::get<3>(param);

    auto out = generate_data(seed, L, p, n_knockoffs);
    auto& ml = std::get<0>(out);
    auto& vl = std::get<1>(out);
    auto& v = std::get<2>(out);
    auto& vs = std::get<3>(out);
    auto& dense = std::get<4>(out);

    gmat_t gmat(ml, vl, n_knockoffs);

    test_quad_form(gmat, dense, v);
    test_quad_form(gmat, dense, vs);
}

INSTANTIATE_TEST_SUITE_P(
        GhostMatrixQuadFormSuite,
        GhostMatrixQuadFormFixture,
        testing::Values(
            std::make_tuple(0, 1, 4, 1),
            std::make_tuple(124, 2, 3, 2),
            std::make_tuple(321, 5, 1, 1),
            std::make_tuple(9382, 3, 9, 2))
    );

// ========================================================
// TEST inv_quad_form
// ========================================================

struct GhostMatrixInvQuadFormFixture
    : testing::Test,
      testing::WithParamInterface<
        std::tuple<size_t, size_t, size_t, size_t, double> >
{};

template <class VecType>
static inline void test_inv_quad_form(
        const gmat_t& gmat,
        const mat_t& dense,
        double s,
        const VecType& v)
{
    auto actual = gmat.inv_quad_form(s, v);
    mat_t T = (1-s) * dense;
    T.diagonal().array() += s;

    auto v_norm_sq = v.squaredNorm();
    vec_t Tv = T * v;
    auto Tv_norm_sq = Tv.squaredNorm();
    auto vTTv = v.dot(Tv);
    double expected = 0;
    if (v_norm_sq > 0) {
        if (vTTv <= 0) expected = std::numeric_limits<double>::infinity();
        else {
            auto v_norm_sq_div_vTTv = v_norm_sq / vTTv;
            auto v_norm_sq_div_vTTv_pow3 = 
                v_norm_sq_div_vTTv * v_norm_sq_div_vTTv * v_norm_sq_div_vTTv;
            expected = v_norm_sq_div_vTTv_pow3 * Tv_norm_sq;
        }
    }
    if (expected == std::numeric_limits<double>::infinity()) {
        EXPECT_EQ(actual, expected);
    } else {
        EXPECT_NEAR(actual, expected, tol * expected);
    }
}

TEST_P(GhostMatrixInvQuadFormFixture, inv_quad_form)
{
    auto param = GetParam();
    size_t seed = std::get<0>(param);
    size_t L = std::get<1>(param);
    size_t p = std::get<2>(param);
    size_t n_knockoffs = std::get<3>(param);
    double s = std::get<4>(param);

    auto out = generate_data(seed, L, p, n_knockoffs);
    auto& ml = std::get<0>(out);
    auto& vl = std::get<1>(out);
    auto& v = std::get<2>(out);
    auto& vs = std::get<3>(out);
    auto& dense = std::get<4>(out);

    gmat_t gmat(ml, vl, n_knockoffs);

    test_inv_quad_form(gmat, dense, s, v);
    test_inv_quad_form(gmat, dense, s, vs);
}

INSTANTIATE_TEST_SUITE_P(
        GhostMatrixInvQuadFormSuite,
        GhostMatrixInvQuadFormFixture,
        testing::Values(
            std::make_tuple(0, 1, 2, 1, 0.5),
            std::make_tuple(124, 1, 3, 1, 0.3),
            std::make_tuple(321, 1, 4, 1, 0.9),
            std::make_tuple(9382, 1, 5, 1, 0.01),
            std::make_tuple(9382, 1, 5, 1, 0),
            std::make_tuple(3213, 1, 5, 1, 0),
            std::make_tuple(341111, 3, 9, 2, 0.2)
            )
    );

} 
} // namespace ghostbasil
