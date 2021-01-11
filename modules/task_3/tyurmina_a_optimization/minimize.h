//  Copyright 2020 Tyurmina Alexandra
#ifndef MODULES_TASK_3_TYURMINA_A_OPTIMIZATION_MINIMIZE_H_
#define MODULES_TASK_3_TYURMINA_A_OPTIMIZATION_MINIMIZE_H_

#include <functional>
#include <vector>

class GlobalOpt {
    double a1; double b1;
    double epsilon;
    std::function<double(double*)> test_func;

 public:
    // XY search
    double GlobalSearchSeq(int iterationsTops);
    double GlobalSearchPar(int iterationsTops);

    GlobalOpt(double a1, double b1, std::function<double(double*)> test_func, double eps);


 protected:
    double get_func_val(double x);
    double get_m(double r, double M);
    double get_M(int i, const std::vector<double>& x);
    double get_R(int i, double m, const std::vector<double>& x);
    double get_new_x(int t, double m, const std::vector<double>& x);
};

#endif  // MODULES_TASK_3_TYURMINA_A_OPTIMIZATION_MINIMIZE_H_
