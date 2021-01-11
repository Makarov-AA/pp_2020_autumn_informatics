// Copyright 2020 Romanuyk Sergey
#ifndef MODULES_TASK_2_ROMANUYK_METOD_PR_ITER_METOD_PR_IT_H_
#define MODULES_TASK_2_ROMANUYK_METOD_PR_ITER_METOD_PR_IT_H_

#include <vector>

std::vector<double> genRandY(int size);
std::vector<double> genMatrix(int size);

std::vector<double> Iteration(std::vector<double> a, std::vector<double> y, double eps, int maxiter, int size);
std::vector<double> genParallelIter(std::vector<double> a, std::vector<double> y, double eps, int maxiter, int size);

double discrepancyNorm(const std::vector<double>& x, const std::vector<double>& a, const std::vector<double>& y);

#endif  // MODULES_TASK_2_ROMANUYK_METOD_PR_ITER_METOD_PR_IT_H_
