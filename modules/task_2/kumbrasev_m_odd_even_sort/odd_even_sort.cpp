// Copyright 2020 Kumbrasev Mark
#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <random>
#include <vector>
#include <utility>
#include "../../../modules/task_2/kumbrasev_m_odd_even_sort/odd_even_sort.h"

std::vector<int> create_vector(int size) {
    std::vector<int> result(size);
    std::random_device rd;
    std::mt19937 mersenne(rd());
    std::uniform_real_distribution<> urd(0, 1000);
    for (int i = 0; i < size; i++) {
        result[i] = static_cast<int>(urd(mersenne));
    }
    return result;
}

void bubbleSort(int *arr, int size) {
     for (int i = 0; i < size - 1; i++) {
        for (int j = 0; j < size - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

void odd_even_sort(std::vector<int>arr, int size) {
    int phase, itr;

    for (phase = 0; phase < size; phase++) {
        if (phase % 2 == 0) {
            for (itr = 1; itr < size; itr += 2) {
                if (arr[itr - 1] > arr[itr]) {
                    std::swap(arr[itr - 1], arr[itr]);
                }
            }
        } else { for (itr = 1; itr < size - 1; itr += 2) {
                if (arr[itr] > arr[itr + 1]) {
                    std::swap(arr[itr], arr[itr + 1]);
                }
            }
        }
    }
}

void PHASE(int SEND_RANK, int RCV_RANK, std::vector<int> arr, int size, MPI_Comm COMM) {
    int current_rank;
    MPI_Comm_rank(COMM, &current_rank);

    std::vector<int> temp_arr(size);

    std::vector<int> aux_arr(size * 2);


    if (current_rank == SEND_RANK) {
        MPI_Send(arr.data(), size, MPI_INT, RCV_RANK, 0, COMM);
        MPI_Recv(arr.data(), size, MPI_INT, RCV_RANK, 1, COMM, MPI_STATUS_IGNORE);
    } else {
        MPI_Recv(temp_arr.data(), size, MPI_INT, SEND_RANK, 0, COMM, MPI_STATUS_IGNORE);
        int itr = size;
        for (int i = 0; i < size; i++) {
            temp_arr[i] = aux_arr[i];
            arr[i] = aux_arr[itr];
            itr++;
        }

        MPI_Send(temp_arr.data(), size, MPI_INT, SEND_RANK, 1, COMM);
    }
}
