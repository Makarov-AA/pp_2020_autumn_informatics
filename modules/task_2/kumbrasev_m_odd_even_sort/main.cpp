// Copyright 2020 Kumbrasev Mark
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include "./odd_even_sort.h"

TEST(Parallel_Operations_MPI, Test_1000) {
    int my_rank, comm_sz;
    int presize = 1000;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    presize = presize * comm_sz;
    std::vector<int> arr = create_vector(presize);
    int local_size = presize / comm_sz;
    std::vector<int> local_arr(local_size);
    bubbleSort(arr.data(), arr.size());
    MPI_Scatter(arr.data(), local_size, MPI_INT, local_arr.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);
    odd_even_sort(local_arr, local_size);
    for (int proc_itr = 1; proc_itr <= comm_sz; proc_itr++) {
        if ((my_rank + proc_itr) % 2 == 0) {
            if (my_rank < comm_sz - 1) {
                PHASE(my_rank, my_rank + 1, local_arr, local_size, MPI_COMM_WORLD);
            }
        } else { if (my_rank > 0) {
            PHASE(my_rank - 1, my_rank, local_arr, local_size, MPI_COMM_WORLD);
            }
        }
    }
    std::vector<int> final_arr(presize);
    MPI_Gather(local_arr.data(), local_size, MPI_INT, final_arr.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);
    if (my_rank == 0) {
        ASSERT_EQ(arr, final_arr);
    }
}

TEST(Parallel_Operations_MPI, Test_100) {
    int my_rank, comm_sz;
    int presize = 100;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    presize = presize * comm_sz;
    std::vector<int> arr = create_vector(presize);
    int local_size = presize / comm_sz;
    std::vector<int> local_arr(local_size);
    bubbleSort(arr.data(), arr.size());
    MPI_Scatter(arr.data(), local_size, MPI_INT, local_arr.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);
    odd_even_sort(local_arr, local_size);
    for (int proc_itr = 1; proc_itr <= comm_sz; proc_itr++) {
        if ((my_rank + proc_itr) % 2 == 0) {
            if (my_rank < comm_sz - 1) {
                PHASE(my_rank, my_rank + 1, local_arr, local_size, MPI_COMM_WORLD);
            }
        } else { if (my_rank > 0) {
            PHASE(my_rank - 1, my_rank, local_arr, local_size, MPI_COMM_WORLD);
            }
        }
    }
    std::vector<int> final_arr(presize);
    MPI_Gather(local_arr.data(), local_size, MPI_INT, final_arr.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);
    if (my_rank == 0) {
        ASSERT_EQ(arr, final_arr);
    }
}

TEST(Parallel_Operations_MPI, Test_250) {
    int my_rank, comm_sz;
    int presize = 250;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    presize = presize * comm_sz;
    std::vector<int> arr = create_vector(presize);
    int local_size = presize / comm_sz;
    std::vector<int> local_arr(local_size);
    bubbleSort(arr.data(), arr.size());
    MPI_Scatter(arr.data(), local_size, MPI_INT, local_arr.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);
    odd_even_sort(local_arr, local_size);
    for (int proc_itr = 1; proc_itr <= comm_sz; proc_itr++) {
        if ((my_rank + proc_itr) % 2 == 0) {
            if (my_rank < comm_sz - 1) {
                PHASE(my_rank, my_rank + 1, local_arr, local_size, MPI_COMM_WORLD);
            }
        } else { if (my_rank > 0) {
            PHASE(my_rank - 1, my_rank, local_arr, local_size, MPI_COMM_WORLD);
            }
        }
    }
    std::vector<int> final_arr(presize);
    MPI_Gather(local_arr.data(), local_size, MPI_INT, final_arr.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);
    if (my_rank == 0) {
        ASSERT_EQ(arr, final_arr);
    }
}

TEST(Parallel_Operations_MPI, Test_456) {
    int my_rank, comm_sz;
    int presize = 456;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    presize = presize * comm_sz;
    std::vector<int> arr = create_vector(presize);
    int local_size = presize / comm_sz;
    std::vector<int> local_arr(local_size);
    bubbleSort(arr.data(), arr.size());
    MPI_Scatter(arr.data(), local_size, MPI_INT, local_arr.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);
    odd_even_sort(local_arr, local_size);
    for (int proc_itr = 1; proc_itr <= comm_sz; proc_itr++) {
        if ((my_rank + proc_itr) % 2 == 0) {
            if (my_rank < comm_sz - 1) {
                PHASE(my_rank, my_rank + 1, local_arr, local_size, MPI_COMM_WORLD);
            }
        } else { if (my_rank > 0) {
            PHASE(my_rank - 1, my_rank, local_arr, local_size, MPI_COMM_WORLD);
            }
        }
    }
    std::vector<int> final_arr(presize);
    MPI_Gather(local_arr.data(), local_size, MPI_INT, final_arr.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);
    if (my_rank == 0) {
        ASSERT_EQ(arr, final_arr);
    }
}

TEST(Parallel_Operations_MPI, Test_10000) {
    int my_rank, comm_sz;
    int presize = 10000;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    presize = presize * comm_sz;
    std::vector<int> arr = create_vector(presize);
    int local_size = presize / comm_sz;
    std::vector<int> local_arr(local_size);
    bubbleSort(arr.data(), arr.size());
    MPI_Scatter(arr.data(), local_size, MPI_INT, local_arr.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);
    odd_even_sort(local_arr, local_size);
    for (int proc_itr = 1; proc_itr <= comm_sz; proc_itr++) {
        if ((my_rank + proc_itr) % 2 == 0) {
            if (my_rank < comm_sz - 1) {
                PHASE(my_rank, my_rank + 1, local_arr, local_size, MPI_COMM_WORLD);
            }
        } else { if (my_rank > 0) {
            PHASE(my_rank - 1, my_rank, local_arr, local_size, MPI_COMM_WORLD);
            }
        }
    }
    std::vector<int> final_arr(presize);
    MPI_Gather(local_arr.data(), local_size, MPI_INT, final_arr.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);
    if (my_rank == 0) {
        ASSERT_EQ(arr, final_arr);
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
