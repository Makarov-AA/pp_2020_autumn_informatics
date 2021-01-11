  // Copyright 2020 Tyurmina Alexandra

#include <mpi.h>
#include <utility>
#include <list>
#include <algorithm>
#include <iostream>
#include <vector>
#include <cmath>
#include "../../../modules/task_3/tyurmina_a_optimization/minimize.h"

double GlobalOpt::get_func_val(double x) {
    return test_func(&x);
}

double GlobalOpt::get_M(int i, const std::vector<double>& x) {
    double diff1 = get_func_val(x[i]) - get_func_val(x[i - 1]);
    double diff2 = x[i] - x[i - 1];

    return std::abs(diff1) / diff2;
}

double GlobalOpt::get_m(double r, double M) {
    if (M == 0) {
        return 1;
    } else {
        return  M * r;
    }
}

GlobalOpt::GlobalOpt(double _a1, double _b1, std::function<double(double*)> _test_func,
    double _eps) {
    a1 = _a1; b1 = _b1;
    test_func = _test_func;
    epsilon = _eps;
}

double GlobalOpt::get_R(int i, double m, const std::vector<double>& x) {
    double diff = get_func_val(x[i]) - get_func_val(x[i - 1]);
    double summ = get_func_val(x[i]) + get_func_val(x[i - 1]);
    double koef = m * (x[i] - x[i - 1]);
    double val = std::pow(diff, 2) / koef - 2 * summ + koef;

    return val;
}

double GlobalOpt::get_new_x(int t, double m, const std::vector<double>& x) {
    double diff = get_func_val(x[t]) - get_func_val(x[t - 1]);
    double summ = x[t] + x[t - 1];
    double val = (summ - diff / m) / 2;

    return val;
}

double GlobalOpt::GlobalSearchSeq(int iterationsTops) {
    std::vector<double> x_val(iterationsTops + 1, 0);
    int iterationsNum = 1;
    int t = 0;

    double r = 2;
    double R = 0;
    double temp = 0;
    double M = get_M(1, x_val);
    double m = get_m(r, M);

    x_val[0] = a1;
    x_val[1] = b1;
    x_val[2] = get_new_x(1, m, x_val);

    ++iterationsNum;

    if (a1 > b1) {
        throw "b is a right border (must be b > a)";
    }
    while (iterationsNum < iterationsTops) {
        sort(x_val.begin(), x_val.begin() + iterationsNum + 1);
        M = get_M(1, x_val);

        for (int i = 2; i <= iterationsNum; ++i) {
            M = std::max(M, get_M(i, x_val));
        }

        m = get_m(r, M);
        R = get_R(1, m, x_val);
        t = 1;

        for (int i = 2; i <= iterationsNum; ++i) {
            temp = get_R(i, m, x_val);
            if (R < temp) {
                R = temp;
                t = i;
            }
        }

        x_val[iterationsNum + 1] = get_new_x(t, m, x_val);
        ++iterationsNum;

        if (x_val[t] - x_val[t - 1] <= epsilon) {
            break;
        }
    }
    return x_val[t];
}

double GlobalOpt::GlobalSearchPar(int iterationsTops) {
    int prNum, prRank;

    MPI_Comm_size(MPI_COMM_WORLD, &prNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &prRank);

    std::vector<double> result(prNum);

    if (a1 > b1) {
        throw "b is a right border (must be b > a)";
    }
    if (prNum > 1) {
        double step = (b1 - a1) / prNum;
        double a = a1 + step * prRank;
        double b = a + step;

        if (a != b) {
            double temp = 0;
            GlobalOpt opt(a, b, test_func, epsilon);
            double* local = new double[prNum];
            *local = opt.GlobalSearchSeq(iterationsTops);
            MPI_Gather(&local[0], 1, MPI_DOUBLE, &result[0], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            double root = get_func_val(result[0]);
            if (prRank == 0) {
                for (int i = 1; i < prNum; ++i) {
                    temp = get_func_val(result[i]);
                    if (temp < root) {
                        root = temp;
                        std::swap(result[i], result[0]);
                    }
                }
            }
        }
    } else {
        return GlobalSearchSeq(iterationsTops);
    }
    return result[0];
}
