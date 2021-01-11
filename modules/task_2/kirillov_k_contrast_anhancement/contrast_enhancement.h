// Copyright 2020 Kirillov Konstantin
#ifndef MODULES_TASK_2_KIRILLOV_K_CONTRAST_ANHANCEMENT_CONTRAST_ENHANCEMENT_H_
#define MODULES_TASK_2_KIRILLOV_K_CONTRAST_ANHANCEMENT_CONTRAST_ENHANCEMENT_H_
#include <vector>
using Matrix = std::vector<std::vector<int>>;
Matrix getRandomMatrix(int rows, int cols);
int check(int tmp);
int getSequentialContrast(Matrix mat, int rows, int cols, double alpha, int beta);
Matrix getParallelContrast(Matrix mat, int rows, int cols, double alpha, int beta);

#endif  // MODULES_TASK_2_KIRILLOV_K_CONTRAST_ANHANCEMENT_CONTRAST_ENHANCEMENT_H_
