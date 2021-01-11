//  Copyright 2020 Tyurmina Alexandra
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <cmath>
#include "./minimize.h"

double f_test1(double* x) {
    return cos((*x) - 1) * cos((*x) * 8 + 1) + 31;
}

double f_test2(double* x) {
    return (*x) + 19 * (*x);
}

TEST(global_GlobalOpt, TEST0_seq_min) {
    GlobalOpt opt(0, 2.2, f_test1, 1e-5);

    double result = opt.GlobalSearchSeq(800);
    double correct_result = 1.0529;
    ASSERT_NEAR(result, correct_result, 1e-3);
}

TEST(global_GlobalOpt, TEST1_par_min) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    GlobalOpt opt(0.3, 2.2, f_test2, 1e-5);

    double result = opt.GlobalSearchPar(800);
    double correct_result = 0.3;
    if (rank == 0) {
        ASSERT_NEAR(result, correct_result, 1e-4);
    }
}

TEST(global_GlobalOpt, TEST2_parallel_min_for_f) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    GlobalOpt opt(0, 1, f_test2, 1e-5);

    double result = opt.GlobalSearchPar(800);
    double correct_result = 0;
    if (rank == 0) {
        ASSERT_NEAR(result, correct_result, 1e-5);
    }
}

TEST(global_GlobalOpt, TEST3_a_greater_then_b_par) {
    GlobalOpt opt(2.2, 0, f_test1, 1e-5);
    EXPECT_ANY_THROW(opt.GlobalSearchPar(800));
}

TEST(global_GlobalOpt, TEST4_a_greater_then_b_seq) {
    GlobalOpt opt(2.2, 0, f_test1, 1e-5);
    EXPECT_ANY_THROW(opt.GlobalSearchSeq(800));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);

    ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
    ::testing::TestEventListeners& listeners =
        ::testing::UnitTest::GetInstance()->listeners();

    listeners.Release(listeners.default_result_printer());
    listeners.Release(listeners.default_xml_generator());

    listeners.Append(new GTestMPIListener::MPIMinimalistPrinter);
    return RUN_ALL_TESTS();
}
