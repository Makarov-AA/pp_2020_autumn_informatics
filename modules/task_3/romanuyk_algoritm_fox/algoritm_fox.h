// Copyright 2020 Romanuyk Sergey
#ifndef MODULES_TASK_3_ROMANUYK_ALGORITM_FOX_ALGORITM_FOX_H_
#define MODULES_TASK_3_ROMANUYK_ALGORITM_FOX_ALGORITM_FOX_H_

#include <iostream>
#include <vector>

bool assertMatrix(const std::vector<double>& A, const std::vector<double>& B);
std::vector<double> genMatrix(int n);
std::vector<double> SequentinalMultiMatrix(const std::vector<double>& A, const std::vector<double>& B, int n);

void createGrid(int GridSize, int procrank, int* GridCoords);
void MultiplyMatrixforParallel(double* A, double* B, double* C, int BlockSize);
std::vector<double> MultiplyMatrixParallel(const std::vector<double>& A, const std::vector<double>& B, int size);

#endif  // MODULES_TASK_3_ROMANUYK_ALGORITM_FOX_ALGORITM_FOX_H_
