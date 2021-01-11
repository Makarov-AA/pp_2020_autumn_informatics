// Copyright 2020 Kirillov Konstantin
#include <mpi.h>
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <vector>
#include<iostream>
#include "./contrast_enhancement.h"
TEST(Parallel_Operations_MPI, Test_Matrix_4x2) {
    int procRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    int rows = 4;
    int cols = 2;
    double alpha = 1.5;
    int beta = 1;
    Matrix global_mat(rows, std::vector<int>(cols));
    Matrix local_mat(rows, std::vector<int>(cols));
    global_mat[0][0] = 178;
    global_mat[0][1] = 41;
    global_mat[1][0] = 21;
    global_mat[1][1] = 17;
    global_mat[2][0] = 82;
    global_mat[2][1] = 193;
    global_mat[3][0] = 56;
    global_mat[3][1] = 212;
    if (procRank == 0) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                local_mat[i][j] = global_mat[i][j];
                local_mat[i][j] = getSequentialContrast(local_mat, i, j, alpha, beta);
            }
        }
    }
    global_mat = getParallelContrast(global_mat, rows, cols, alpha, beta);
    if (procRank == 0) {
        for (int  i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                ASSERT_EQ(local_mat[i][j], global_mat[i][j]);
            }
        }
    }
}
TEST(Parallel_Operations_MPI, Test_Matrix_4x3) {
    int procRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    int rows = 4;
    int cols = 3;
    double alpha = 1.5;
    int beta = 1;
    Matrix global_mat(rows, std::vector<int>(cols));
    Matrix local_mat(rows, std::vector<int>(cols));
    global_mat[0][0] = 178;
    global_mat[0][1] = 41;
    global_mat[0][2] = 243;
    global_mat[1][0] = 21;
    global_mat[1][1] = 17;
    global_mat[1][2] = 42;
    global_mat[2][0] = 157;
    global_mat[2][1] = 53;
    global_mat[2][2] = 229;
    global_mat[3][0] = 87;
    global_mat[3][1] = 176;
    global_mat[3][2] = 112;
    if (procRank == 0) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                local_mat[i][j] = global_mat[i][j];
                local_mat[i][j] = getSequentialContrast(local_mat, i, j, alpha, beta);
            }
        }
    }
    global_mat = getParallelContrast(global_mat, rows, cols, alpha, beta);
    if (procRank == 0) {
        for (int  i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                ASSERT_EQ(local_mat[i][j], global_mat[i][j]);
            }
        }
    }
}
TEST(Parallel_Operations_MPI, Test_Matrix_4x4) {
    int procRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    int rows = 4;
    int cols = 4;
    double alpha = 1.5;
    int beta = 1;
    Matrix global_mat(rows, std::vector<int>(cols));
    Matrix local_mat(rows, std::vector<int>(cols));
    global_mat[0][0] = 178;
    global_mat[0][1] = 41;
    global_mat[0][2] = 243;
    global_mat[0][3] = 65;
    global_mat[1][0] = 21;
    global_mat[1][1] = 17;
    global_mat[1][2] = 42;
    global_mat[1][3] = 231;
    global_mat[2][0] = 157;
    global_mat[2][1] = 53;
    global_mat[2][2] = 229;
    global_mat[2][3] = 165;
    global_mat[3][0] = 123;
    global_mat[3][1] = 42;
    global_mat[3][2] = 89;
    global_mat[3][3] = 146;
    if (procRank == 0) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                local_mat[i][j] = global_mat[i][j];
                local_mat[i][j] = getSequentialContrast(local_mat, i, j, alpha, beta);
            }
        }
    }
    global_mat = getParallelContrast(global_mat, rows, cols, alpha, beta);
    if (procRank == 0) {
        for (int  i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                ASSERT_EQ(local_mat[i][j], global_mat[i][j]);
            }
        }
    }
}
TEST(Parallel_Operations_MPI, Test_Matrix_5x4) {
    int procRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    int rows = 5;
    int cols = 4;
    double alpha = 1.5;
    int beta = 1;
    Matrix global_mat(rows, std::vector<int>(cols));
    Matrix local_mat(rows, std::vector<int>(cols));
    global_mat[0][0] = 178;
    global_mat[0][1] = 41;
    global_mat[0][2] = 243;
    global_mat[0][3] = 65;
    global_mat[1][0] = 21;
    global_mat[1][1] = 17;
    global_mat[1][2] = 42;
    global_mat[1][3] = 231;
    global_mat[2][0] = 157;
    global_mat[2][1] = 53;
    global_mat[2][2] = 229;
    global_mat[2][3] = 165;
    global_mat[3][0] = 123;
    global_mat[3][1] = 42;
    global_mat[3][2] = 89;
    global_mat[3][3] = 146;
    global_mat[4][0] = 13;
    global_mat[4][1] = 41;
    global_mat[4][2] = 178;
    global_mat[4][3] = 76;
    if (procRank == 0) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                local_mat[i][j] = global_mat[i][j];
                local_mat[i][j] = getSequentialContrast(local_mat, i, j, alpha, beta);
            }
        }
    }
    global_mat = getParallelContrast(global_mat, rows, cols, alpha, beta);
    if (procRank == 0) {
        for (int  i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                ASSERT_EQ(local_mat[i][j], global_mat[i][j]);
            }
        }
    }
}
TEST(Parallel_Operations_MPI, Test_Matrix_4x5) {
    int procRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    int rows = 4;
    int cols = 5;
    double alpha = 1.5;
    int beta = 1;
    Matrix global_mat(rows, std::vector<int>(cols));
    Matrix local_mat(rows, std::vector<int>(cols));
    global_mat[0][0] = 178;
    global_mat[0][1] = 41;
    global_mat[0][2] = 243;
    global_mat[0][3] = 65;
    global_mat[0][4] = 16;
    global_mat[1][0] = 21;
    global_mat[1][1] = 17;
    global_mat[1][2] = 42;
    global_mat[1][3] = 212;
    global_mat[1][4] = 174;
    global_mat[2][0] = 157;
    global_mat[2][1] = 53;
    global_mat[2][2] = 229;
    global_mat[2][3] = 165;
    global_mat[2][4] = 199;
    global_mat[3][0] = 123;
    global_mat[3][1] = 42;
    global_mat[3][2] = 89;
    global_mat[3][3] = 146;
    global_mat[3][4] = 12;
    if (procRank == 0) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                local_mat[i][j] = global_mat[i][j];
                local_mat[i][j] = getSequentialContrast(local_mat, i, j, alpha, beta);
            }
        }
    }
    global_mat = getParallelContrast(global_mat, rows, cols, alpha, beta);
    if (procRank == 0) {
        for (int  i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                ASSERT_EQ(local_mat[i][j], global_mat[i][j]);
            }
        }
    }
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);

    ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();

    listeners.Release(listeners.default_result_printer());
    listeners.Release(listeners.default_xml_generator());

    listeners.Append(new GTestMPIListener::MPIMinimalistPrinter);
    return RUN_ALL_TESTS();
}
