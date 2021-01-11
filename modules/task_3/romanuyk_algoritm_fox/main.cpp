// Copyright 2020 Romanuyk Sergey
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <vector>
#include <ctime>
#include <algorithm>
#include <cmath>
#include"../../../modules/task_3/romanuyk_algoritm_fox/algoritm_fox.h"

TEST(Parallel_Operations_MPI, Test1) {
    int procrank;
    int procnum;
    double t_b1, t_e1, t_b2, t_e2;
    MPI_Comm_size(MPI_COMM_WORLD, &procnum);
    MPI_Comm_rank(MPI_COMM_WORLD, &procrank);
    std::vector<double> a;
    std::vector<double> b;
    int size = 16;
    int GridSize = static_cast<int>(sqrt(procnum));
    if (GridSize * GridSize == procnum) {
        int delta = size % GridSize;
        if (delta)
            size += GridSize - (size % GridSize);
        if (procrank == 0) {
            a = genMatrix(size);
            b = genMatrix(size);
        }
        t_b1 = MPI_Wtime();
        std::vector<double> res1 = MultiplyMatrixParallel(a, b, size);
        if (procrank == 0) {
            t_e1 = MPI_Wtime();
            t_b2 = MPI_Wtime();
            std::vector<double> res2 = SequentinalMultiMatrix(a, b, size);
            t_e2 = MPI_Wtime();
            std::cout << "Parallel time: " << t_e1 - t_b1 << std::endl;
            std::cout << "Sequentional time: " << t_e2 - t_b2 << std::endl;
            ASSERT_TRUE(assertMatrix(res1, res2));
        }
    } else {
        if (procrank == 0) {
            ASSERT_FALSE(0);
        }
    }
}

TEST(Parallel_Operations_MPI, Test2) {
    int procrank;
    int procnum;
    double t_b1, t_e1, t_b2, t_e2;
    MPI_Comm_size(MPI_COMM_WORLD, &procnum);
    MPI_Comm_rank(MPI_COMM_WORLD, &procrank);
    std::vector<double> a;
    std::vector<double> b;
    int size = 28;
    int GridSize = static_cast<int>(sqrt(procnum));
    if (GridSize * GridSize == procnum) {
        int delta = size % GridSize;
        if (delta)
            size += GridSize - (size % GridSize);
        if (procrank == 0) {
            a = genMatrix(size);
            b = genMatrix(size);
        }
        t_b1 = MPI_Wtime();
        std::vector<double> res1 = MultiplyMatrixParallel(a, b, size);
        if (procrank == 0) {
            t_e1 = MPI_Wtime();
            t_b2 = MPI_Wtime();
            std::vector<double> res2 = SequentinalMultiMatrix(a, b, size);
            t_e2 = MPI_Wtime();
            std::cout << "Parallel time: " << t_e1 - t_b1 << std::endl;
            std::cout << "Sequentional time: " << t_e2 - t_b2 << std::endl;
            ASSERT_TRUE(assertMatrix(res1, res2));
        }
    } else {
        if (procrank == 0) {
            ASSERT_FALSE(0);
        }
    }
}

TEST(Parallel_Operations_MPI, Test3) {
    int procrank;
    int procnum;
    double t_b1, t_e1, t_b2, t_e2;
    MPI_Comm_size(MPI_COMM_WORLD, &procnum);
    MPI_Comm_rank(MPI_COMM_WORLD, &procrank);
    std::vector<double> a;
    std::vector<double> b;
    int size = 64;
    int GridSize = static_cast<int>(sqrt(procnum));
    if (GridSize * GridSize == procnum) {
        int delta = size % GridSize;
        if (delta)
            size += GridSize - (size % GridSize);
        if (procrank == 0) {
            a = genMatrix(size);
            b = genMatrix(size);
        }
        t_b1 = MPI_Wtime();
        std::vector<double> res1 = MultiplyMatrixParallel(a, b, size);
        if (procrank == 0) {
            t_e1 = MPI_Wtime();
            t_b2 = MPI_Wtime();
            std::vector<double> res2 = SequentinalMultiMatrix(a, b, size);
            t_e2 = MPI_Wtime();
            std::cout << "Parallel time: " << t_e1 - t_b1 << std::endl;
            std::cout << "Sequentional time: " << t_e2 - t_b2 << std::endl;
            ASSERT_TRUE(assertMatrix(res1, res2));
        }
    } else {
        if (procrank == 0) {
            ASSERT_FALSE(0);
        }
    }
}

TEST(Parallel_Operations_MPI, Test4) {
    int procrank;
    int procnum;
    double t_b1, t_e1, t_b2, t_e2;
    MPI_Comm_size(MPI_COMM_WORLD, &procnum);
    MPI_Comm_rank(MPI_COMM_WORLD, &procrank);
    std::vector<double> a;
    std::vector<double> b;
    int size = 49;
    int GridSize = static_cast<int>(sqrt(procnum));
    if (GridSize * GridSize == procnum) {
        int delta = size % GridSize;
        if (delta)
            size += GridSize - (size % GridSize);
        if (procrank == 0) {
            a = genMatrix(size);
            b = genMatrix(size);
        }
        t_b1 = MPI_Wtime();
        std::vector<double> res1 = MultiplyMatrixParallel(a, b, size);
        if (procrank == 0) {
            t_e1 = MPI_Wtime();
            t_b2 = MPI_Wtime();
            std::vector<double> res2 = SequentinalMultiMatrix(a, b, size);
            t_e2 = MPI_Wtime();
            std::cout << "Parallel time: " << t_e1 - t_b1 << std::endl;
            std::cout << "Sequentional time: " << t_e2 - t_b2 << std::endl;
            ASSERT_TRUE(assertMatrix(res1, res2));
        }
    } else {
        if (procrank == 0) {
            ASSERT_FALSE(0);
        }
    }
}

TEST(Parallel_Operations_MPI, Test5) {
    int procrank;
    int procnum;
    double t_b1, t_e1, t_b2, t_e2;
    MPI_Comm_size(MPI_COMM_WORLD, &procnum);
    MPI_Comm_rank(MPI_COMM_WORLD, &procrank);
    std::vector<double> a;
    std::vector<double> b;
    int size = 81;
    int GridSize = static_cast<int>(sqrt(procnum));
    if (GridSize * GridSize == procnum) {
        int delta = size % GridSize;
        if (delta)
            size += GridSize - (size % GridSize);
        if (procrank == 0) {
            a = genMatrix(size);
            b = genMatrix(size);
        }
        t_b1 = MPI_Wtime();
        std::vector<double> res1 = MultiplyMatrixParallel(a, b, size);
        if (procrank == 0) {
            t_e1 = MPI_Wtime();
            t_b2 = MPI_Wtime();
            std::vector<double> res2 = SequentinalMultiMatrix(a, b, size);
            t_e2 = MPI_Wtime();
            std::cout << "Parallel time: " << t_e1 - t_b1 << std::endl;
            std::cout << "Sequentional time: " << t_e2 - t_b2 << std::endl;
            ASSERT_TRUE(assertMatrix(res1, res2));
        }
    } else {
        if (procrank == 0) {
            ASSERT_FALSE(0);
        }
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
