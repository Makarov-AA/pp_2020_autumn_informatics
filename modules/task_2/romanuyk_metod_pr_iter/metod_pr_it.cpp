// Copyright 2020 Romanuyk Sergey
#include <mpi.h>
#include <vector>
#include <random>
#include <ctime>
#include <algorithm>
#include <iostream>
#include "../../../modules/task_2/romanuyk_metod_pr_iter/metod_pr_it.h"

std::vector<double> Iteration(std::vector<double> a, std::vector<double> y, double eps, int maxiter, int size) {
    std::vector<double> Xn(size);
    std::vector<double> res(size);
    for (int i = 0; i < size; ++i) {
        res[i] = y[i] / a[i * size + i];
        y[i] = y[i] / a[i * size + i];
        for (int j = 0; j < size; j++) {
            if (j != i)
                a[i * size + j] = -(a[i * size + j] / a[i * size + i]);
        }
        a[i * size + i] = 0.0;
    }
    int iter = 0;
    double max = 0.0;
    do {
        for (int i = 0; i < size; i++) {
            Xn[i] = y[i];
            for (int j = 0; j < size; j++) {
                Xn[i] += a[i * size + j] * res[j];
            }
        }
        max = 0.0;
        for (int i = 0; i < size; i++) {
            if (fabs(Xn[i] - res[i]) > max) {
                max = fabs(Xn[i] - res[i]);
            }
            res[i] = Xn[i];
        }
        iter++;
    } while (max >= eps && iter < maxiter);
    return Xn;
}

std::vector<double> genMatrix(int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> urd(-50, 50);
    std::vector<double> matrix(size * size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i * size + j] = urd(gen);
        }
        matrix[i * size + i] = size + 2;
    }
    return matrix;
}

std::vector<double> genRandY(int size) {
    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));
    std::vector<double> y(size);
    for (int i = 0; i < size; i++) {
        y[i] = gen() % 100;
    }
    return y;
}

std::vector<double> genParallelIter(std::vector<double> a, std::vector<double> y, double eps, int maxiter, int size) {
    int proc_size;
    int proc_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &proc_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    const int delta = size / proc_size;
    const int ost = size % proc_size;
    std::vector<double> Xn(size);
    int* iteration = new int[proc_size * 4];
    int* displs_y = iteration;
    int* displs_a = iteration + proc_size;
    int *sendcounts_y = iteration + proc_size + proc_size;
    int *sendcounts_a = iteration + proc_size + proc_size + proc_size;
    displs_a[0] = 0;
    displs_y[0] = 0;
    sendcounts_y[0] = delta + static_cast<int>(0 < ost);
    sendcounts_a[0] = (delta + static_cast<int>(0 < ost)) * size;
    for (int i = 1; i < proc_size; i++) {
        displs_y[i] = displs_y[i - 1] + delta + static_cast<int>(i <= ost);
        displs_a[i] = displs_a[i - 1] + (delta + static_cast<int>(i <= ost)) * size;
        sendcounts_y[i] = delta + static_cast<int>(i < ost);
        sendcounts_a[i] = (delta + static_cast<int>(i < ost)) * size;
    }
    int procsize = delta + static_cast<int>(proc_rank < ost);
    double* a_proc = new double[procsize * size];
    double* y_proc = new double[procsize];
    double* x_proc = new double[procsize];
    double* res = new double[size];
    MPI_Scatterv(a.data(), sendcounts_a, displs_a, MPI_DOUBLE, a_proc, procsize * size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (proc_rank == 0) {
        for (int i = 0; i < size; i++) {
            Xn[i] = y[i] / a[i * size + i];
            res[i] = y[i] / a[i * size + i];
            y[i] = y[i] / a[i * size + i];
        }
    }

    MPI_Scatterv(y.data(), sendcounts_y, displs_y, MPI_DOUBLE, y_proc, procsize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(res, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < procsize; i++) {
        for (int j = 0; j < size; j++) {
            if (j != i)
                a_proc[i * size + j] = -(a_proc[i * size + j] / a_proc[i * size + i]);
        }
        a_proc[i * size + i] = 0.0;
    }
    for (int i = 0; i < size; i++) {
        Xn[i] = res[i];
    }
    double max = 0.0;
    int iter = 0;
    do {
        for (int i = 0; i < procsize; i++) {
            x_proc[i] = y_proc[i];
            for (int j = 0; j < size; j++) {
                x_proc[i] = x_proc[i] + (a_proc[i * size + j] * res[j]);
            }
        }

        MPI_Allgatherv(x_proc, procsize, MPI_DOUBLE, Xn.data(), sendcounts_y, displs_y, MPI_DOUBLE, MPI_COMM_WORLD);

        max = 0.0;
        double local = 0.0;
        for (int i = 0; i < procsize; i++) {
            if (fabs(x_proc[i] - res[displs_y[proc_rank] + i]) > local) {
                local = fabs(x_proc[i] - res[displs_y[proc_rank] + i]);
            }
        }
        MPI_Allreduce(&local, &max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        for (int i = 0; i < size; i++) {
            res[i] = Xn[i];
        }
        iter++;
    } while (max >= eps && iter < maxiter);
    return Xn;
}

double discrepancyNorm(const std::vector<double>& x, const std::vector<double>& a, const std::vector<double>& y) {
    const int size = x.size();
    double dis = 0.0;
    for (int i = 0; i < size; i++) {
        double res = 0.0;
        for (int j = 0; j < size; j++) {
            res += a[i * size + j] * x[j];
        }
        if (fabs(res - y[i]) > dis) {
            dis = fabs(res - y[i]);
        }
    }
    return dis;
}
