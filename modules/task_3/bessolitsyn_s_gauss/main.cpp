// Copyright 2020 Bessolitsyn Sergey
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include "./bessolitsyn_s_gauss.h"

TEST(Parallel_Operations_MPI, Test_50) {
    MPI_Barrier(MPI_COMM_WORLD);
    int p_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &p_rank);
    int n = 50;
    double t;
    std::vector<double> input_image, s_output_image, p_output_image;
    if (p_rank == 0) {
        input_image.resize(n*n);
        for (int i = 0; i < n * n; i++)
            input_image[i] = i % 10;
        t = MPI_Wtime();
    }
    p_output_image = filter_par(input_image, n, n);
    if (p_rank == 0) {
        std::cout << "Parallel filter:" << MPI_Wtime() - t << std::endl;
        t = MPI_Wtime();
        s_output_image = filter_seq(input_image, n, n);
        std::cout << "Sequential filter:" << MPI_Wtime() - t << std::endl;
        ASSERT_NO_THROW();
    }
}

TEST(Parallel_Operations_MPI, Test_250) {
    MPI_Barrier(MPI_COMM_WORLD);
    int p_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &p_rank);
    int n = 250;
    double t;
    std::vector<double> input_image, s_output_image, p_output_image;
    if (p_rank == 0) {
        input_image.resize(n*n);
        for (int i = 0; i < n * n; i++)
            input_image[i] = i % 10;
        t = MPI_Wtime();
    }
    p_output_image = filter_par(input_image, n, n);
    if (p_rank == 0) {
        std::cout << "Parallel filter:" << MPI_Wtime() - t << std::endl;
        t = MPI_Wtime();
        s_output_image = filter_seq(input_image, n, n);
        std::cout << "Sequential filter:" << MPI_Wtime() - t << std::endl;
        ASSERT_NO_THROW();
    }
}

TEST(Parallel_Operations_MPI, Test_500) {
    MPI_Barrier(MPI_COMM_WORLD);
    int p_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &p_rank);
    int n = 500;
    double t;
    std::vector<double> input_image, s_output_image, p_output_image;
    if (p_rank == 0) {
        input_image.resize(n*n);
        for (int i = 0; i < n * n; i++)
            input_image[i] = i % 10;
        t = MPI_Wtime();
    }
    p_output_image = filter_par(input_image, n, n);
    if (p_rank == 0) {
        std::cout << "Parallel filter:" << MPI_Wtime() - t << std::endl;
        t = MPI_Wtime();
        s_output_image = filter_seq(input_image, n, n);
        std::cout << "Sequential filter:" << MPI_Wtime() - t << std::endl;
        ASSERT_NO_THROW();
    }
}

TEST(Parallel_Operations_MPI, Test_2500) {
    MPI_Barrier(MPI_COMM_WORLD);
    int p_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &p_rank);
    int n = 2500;
    double t;
    std::vector<double> input_image, s_output_image, p_output_image;
    if (p_rank == 0) {
        input_image.resize(n*n);
        for (int i = 0; i < n * n; i++)
            input_image[i] = i % 10;
        t = MPI_Wtime();
    }
    p_output_image = filter_par(input_image, n, n);
    if (p_rank == 0) {
        std::cout << "Parallel filter:" << MPI_Wtime() - t << std::endl;
        t = MPI_Wtime();
        s_output_image = filter_seq(input_image, n, n);
        std::cout << "Sequential filter:" << MPI_Wtime() - t << std::endl;
        ASSERT_NO_THROW();
    }
}

TEST(Parallel_Operations_MPI, Test_5000) {
    MPI_Barrier(MPI_COMM_WORLD);
    int p_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &p_rank);
    int n = 5000;
    double t;
    std::vector<double> input_image, s_output_image, p_output_image;
    if (p_rank == 0) {
        input_image.resize(n*n);
        for (int i = 0; i < n * n; i++)
            input_image[i] = i % 10;
        t = MPI_Wtime();
    }
    p_output_image = filter_par(input_image, n, n);
    if (p_rank == 0) {
        std::cout << "Parallel filter:" << MPI_Wtime() - t << std::endl;
        t = MPI_Wtime();
        s_output_image = filter_seq(input_image, n, n);
        std::cout << "Sequential filter:" << MPI_Wtime() - t << std::endl;
        ASSERT_NO_THROW();
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
