// Copyright 2020 Taktaev Artem
#include <mpi.h>
#include <cmath>
#include <vector>
#include <functional>
#include <algorithm>
#include "../../../modules/task_3/taktaev_a_strongin_method/strongin_method.h"

Strongin::Strongin(double _a, double _b, double _r, int _prec, std::function<double(double)> _f) {
    if (_a > _b) throw "Must be a < b";
    if (_r <= 1) throw "Must be r > 1";
    if (_prec > -2) throw "More precisely please";
    a = _a;
    b = _b;
    r = _r;
    prec = _prec;
    f = _f;

    n = 2;
    x.resize(n);
    x[0] = a;
    x[1] = b;
    z.resize(n);
    z[0] = f(a);
    z[1] = f(b);
}

void Strongin::addNode(double xx) {
    if ((xx <= a) || (xx >= b)) throw "New node out of range";
    n++;
    x.resize(n);
    z.resize(n);
    int i = 0;
    while (xx > x[i]) i++;
    for (int j = i + 1; j < n; j++) {
        x[j] = x[j - 1];
        z[j] = z[j - 1];
    }
    x[i] = xx;
    z[i] = f(xx);
}

double Strongin::calculateM() {
    double max = (std::abs(z[1] - z[0])) / (x[1] - x[0]);
    for (int i = 2; i < n; i++) {
        if ((std::abs(z[i] - z[i - 1])) / (x[i] - x[i - 1]) > max) {
            max = (std::abs(z[i] - z[i - 1])) / (x[i] - x[i - 1]);
        }
    }
    if (max == 0) return 1.0;
    return r * max;
}

std::vector<double> Strongin::calculateR(double m) {
    std::vector<double> r_vec;
    for (int i = 1; i < n; i++) {
        double r = m * (x[i] - x[i - 1]) + (z[i] - z[i - 1]) * (z[i] - z[i - 1]) / (m * (x[i] - x[i - 1]))
                   - 2 * (z[i] + z[i - 1]);
        r_vec.push_back(r);
    }
    return r_vec;
}

double Strongin::seqStronginSearch() {
    double delta = b - a;
    double eps = pow(10, prec);
    int t = 1;
    double xx;
    while (delta > eps) {
        double m = calculateM();
        std::vector<double> r_vec = calculateR(m);
        double r_max = r_vec[0];
        t = 1;
        for (int i = 2; i < n; i++) {
            if (r_vec[i - 1] > r_max) {
                r_max = r_vec[i - 1];
                t = i;
            }
        }
        xx = (x[t - 1] + x[t]) / 2 - (z[t] - z[t - 1]) / (2 * m);
        addNode(xx);
        delta = x[t] - x[t - 1];
    }
    return xx;
}

struct Arteboss {
    double R;
    int T;

    Arteboss() {
        R = 0;
        T = 1;
    }
};

double Strongin::parStronginSearch() {
    int proc_num, proc_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
    int t = 0;
    double delta = b - a;
    double eps = pow(10, prec);
    double m = 0;
    double _m = 0;
    double xx;

    while (delta > eps) {
        for (int i = 1; i < n; i++) {
            if (proc_rank == (i - 1) % proc_num) {
                if ((std::abs(z[i] - z[i - 1])) / (x[i] - x[i - 1]) > _m) {
                     _m = (std::abs(z[i] - z[i - 1])) / (x[i] - x[i - 1]);
                }
            }
        }
        MPI_Allreduce(&_m, &m, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        m = (m == 0 ? 1 : r * m);
        Arteboss r_max;
        Arteboss r1;
        for (int i = 1; i < n; i++) {
            if (proc_rank == (i - 1) % proc_num) {
                if (m * (x[i] - x[i - 1]) + (z[i] - z[i - 1]) * (z[i] - z[i - 1]) / (m * (x[i] - x[i - 1]))
                         - 2 * (z[i] + z[i - 1]) > r1.R) {
                    r1.R = m * (x[i] - x[i - 1]) + (z[i] - z[i - 1]) * (z[i] - z[i - 1]) / (m * (x[i] - x[i - 1]))
                         - 2 * (z[i] + z[i - 1]);
                    r1.T = i;
                }
            }
        }
        MPI_Allreduce(&r1, &r_max, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
        t = r_max.T;
        xx = (x[t - 1] + x[t]) / 2 - (z[t] - z[t - 1]) / (2 * m);
        addNode(xx);
        delta = x[t] - x[t - 1];
    }
    return xx;
}
