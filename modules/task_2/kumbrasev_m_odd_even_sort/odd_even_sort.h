// Copyright 2020 Kumbrasev Mark
#ifndef MODULES_TASK_2_KUMBRASEV_M_ODD_EVEN_SORT_ODD_EVEN_SORT_H_
#define MODULES_TASK_2_KUMBRASEV_M_ODD_EVEN_SORT_ODD_EVEN_SORT_H_

#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <random>
#include <vector>

std::vector<int> create_vector(int size);
void bubbleSort(int *arr, int size);
void odd_even_sort(std::vector<int>arr, int size);

void PHASE(int SEND_RANK, int RCV_RANK, std::vector<int> arr, int size, MPI_Comm COMM);

#endif  // MODULES_TASK_2_KUMBRASEV_M_ODD_EVEN_SORT_ODD_EVEN_SORT_H_
