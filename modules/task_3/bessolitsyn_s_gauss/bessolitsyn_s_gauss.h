// Copyright 2020 Bessolitsyn Sergey
#ifndef MODULES_TASK_3_BESSOLITSYN_S_GAUSS_BESSOLITSYN_S_GAUSS_H_
#define MODULES_TASK_3_BESSOLITSYN_S_GAUSS_BESSOLITSYN_S_GAUSS_H_

#include <vector>

std::vector<double> filter_seq(std::vector<double> input_image, int w, int h);
std::vector<double> filter_par(std::vector<double> input_image, int w, int h);

#endif  // MODULES_TASK_3_BESSOLITSYN_S_GAUSS_BESSOLITSYN_S_GAUSS_H_
