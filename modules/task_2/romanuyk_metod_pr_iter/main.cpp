// Copyright 2020 Romanuyk Sergey
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <vector>
#include <ctime>
#include "../../../modules/task_2/romanuyk_metod_pr_iter/metod_pr_it.h"

TEST(Parallel_Operations_MPI, Test1) {
    int proc_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    const double eps = 0.00001;
    std::vector<double> a;
    std::vector<double> y;
    if (proc_rank == 0) {
        a = genMatrix(10);
        y = genRandY(10);
    }
    double t_b1, t_e1, t_b2, t_e2;
    if (proc_rank == 0) {
        t_b1 = MPI_Wtime();
    }
    std::vector<double> x1 = genParallelIter(a, y, eps, 1000, 10);
    if (proc_rank == 0) {
        t_e1 = MPI_Wtime();
        std::cout << "Parallel time: " << t_e1 - t_b1 << std::endl;
    }
    if (proc_rank == 0) {
        t_b2 = MPI_Wtime();
        std::vector<double> x2 = Iteration(a, y, eps, 1000, 10);
        t_e2 = MPI_Wtime();
        std::cout << "Sequentional time: " << t_e2 - t_b2 << std::endl;
        double disp1 = discrepancyNorm(x1, a, y);
        double disp2 = discrepancyNorm(x2, a, y);
        ASSERT_EQ(true, disp1 < 1);
        ASSERT_EQ(true, disp2 < 1);
    }
}

TEST(Parallel_Operations_MPI, Test2) {
    int proc_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    const double eps = 0.00001;
    std::vector<double> a;
    std::vector<double> y;
    if (proc_rank == 0) {
        a = genMatrix(20);
        y = genRandY(20);
    }
    double t_b1, t_e1, t_b2, t_e2;
    if (proc_rank == 0) {
        t_b1 = MPI_Wtime();
    }
    std::vector<double> x1 = genParallelIter(a, y, eps, 1000, 20);
    if (proc_rank == 0) {
        t_e1 = MPI_Wtime();
        std::cout << "Parallel time: " << t_e1 - t_b1 << std::endl;
    }
    if (proc_rank == 0) {
        t_b2 = MPI_Wtime();
        std::vector<double> x2 = Iteration(a, y, eps, 1000, 20);
        t_e2 = MPI_Wtime();
        std::cout << "Sequentional time: " << t_e2 - t_b2 << std::endl;
        double disp1 = discrepancyNorm(x1, a, y);
        double disp2 = discrepancyNorm(x2, a, y);
        ASSERT_EQ(true, disp1 < 1);
        ASSERT_EQ(true, disp2 < 1);
    }
}

TEST(Parallel_Operations_MPI, Test3) {
    int proc_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    const double eps = 0.00001;
    std::vector<double> a;
    std::vector<double> y;
    if (proc_rank == 0) {
        a = genMatrix(100);
        y = genRandY(100);
    }
    double t_b1, t_e1, t_b2, t_e2;
    if (proc_rank == 0) {
        t_b1 = MPI_Wtime();
    }
    std::vector<double> x1 = genParallelIter(a, y, eps, 1000, 100);
    if (proc_rank == 0) {
        t_e1 = MPI_Wtime();
        std::cout << "Parallel time: " << t_e1 - t_b1 << std::endl;
    }
    if (proc_rank == 0) {
        t_b2 = MPI_Wtime();
        std::vector<double> x2 = Iteration(a, y, eps, 1000, 100);
        t_e2 = MPI_Wtime();
        std::cout << "Sequentional time: " << t_e2 - t_b2 << std::endl;
        double disp1 = discrepancyNorm(x1, a, y);
        double disp2 = discrepancyNorm(x2, a, y);
        ASSERT_EQ(true, disp1 < 1);
        ASSERT_EQ(true, disp2 < 1);
    }
}

TEST(Parallel_Operations_MPI, Test4) {
    int proc_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    const double eps = 0.00001;
    std::vector<double> a;
    std::vector<double> y;
    if (proc_rank == 0) {
        a = genMatrix(40);
        y = genRandY(40);
    }
    double t_b1, t_e1, t_b2, t_e2;
    if (proc_rank == 0) {
        t_b1 = MPI_Wtime();
    }
    std::vector<double> x1 = genParallelIter(a, y, eps, 1000, 40);
    if (proc_rank == 0) {
        t_e1 = MPI_Wtime();
        std::cout << "Parallel time: " << t_e1 - t_b1 << std::endl;
    }
    if (proc_rank == 0) {
        t_b2 = MPI_Wtime();
        std::vector<double> x2 = Iteration(a, y, eps, 1000, 40);
        t_e2 = MPI_Wtime();
        std::cout << "Sequentional time: " << t_e2 - t_b2 << std::endl;
        double disp1 = discrepancyNorm(x1, a, y);
        double disp2 = discrepancyNorm(x2, a, y);
        ASSERT_EQ(true, disp1 < 1);
        ASSERT_EQ(true, disp2 < 1);
    }
}

TEST(Parallel_Operations_MPI, Test5) {
    int proc_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    const double eps = 0.00001;
    std::vector<double> a;
    std::vector<double> y;
    if (proc_rank == 0) {
        a = genMatrix(30);
        y = genRandY(30);
    }
    double t_b1, t_e1, t_b2, t_e2;
    if (proc_rank == 0) {
        t_b1 = MPI_Wtime();
    }
    std::vector<double> x1 = genParallelIter(a, y, eps, 1000, 30);
    if (proc_rank == 0) {
        t_e1 = MPI_Wtime();
        std::cout << "Parallel time: " << t_e1 - t_b1 << std::endl;
    }
    if (proc_rank == 0) {
        t_b2 = MPI_Wtime();
        std::vector<double> x2 = Iteration(a, y, eps, 1000, 30);
        t_e2 = MPI_Wtime();
        std::cout << "Sequentional time: " << t_e2 - t_b2 << std::endl;
        double disp1 = discrepancyNorm(x1, a, y);
        double disp2 = discrepancyNorm(x2, a, y);
        ASSERT_EQ(true, disp1 < 1);
        ASSERT_EQ(true, disp2 < 1);
    }
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
