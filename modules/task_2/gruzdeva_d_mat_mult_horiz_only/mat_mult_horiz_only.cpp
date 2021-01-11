// Copyright 2020 Gruzdeva Diana

#include "../../../modules/task_2/gruzdeva_d_mat_mult_horiz_only/mat_mult_horiz_only.h"
#include <random>
#include <vector>
#include <ctime>
#include <iostream>

std::vector<double> getRandomMatrix(int rows, int cols, time_t seed) {
    std::vector<double> mat(rows * cols);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dis(-10, 10);
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = dis(gen);
    }
    return mat;
}


std::vector<double> getSequentialMultiplication(std::vector<double> matrixA,
                    std::vector<double> matrixB, int aRows, int aCols, int bCols) {
    std::vector<double> matrixC(matrixA.size() / aCols * matrixB.size() / aCols);
    for (unsigned int i = 0; i < matrixA.size() / aCols; i++) {
        for (int j = 0; j < bCols; j++) {
            matrixC[i * bCols + j] = 0;
            for (int k = 0; k < aCols; k++) {
                matrixC[i * bCols + j] += matrixA[i * aCols + k] * matrixB[k * bCols + j];
            }
        }
    }
    return matrixC;
}

std::vector<double> getParallelMultiplication(std::vector<double> matrixA,
                    std::vector<double> matrixB, int aRows, int aCols, int bCols) {
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int stripeHeight = aRows / size;
    int stripeSize = stripeHeight * aCols;
    std::vector<double> matrixC(aRows * bCols);

    std::vector<double> matrixAStripe(stripeSize);
    std::vector<double> matrixBStripe(stripeSize);
    std::vector<double> matrixCStripe(stripeSize);

    MPI_Scatter(matrixA.data(), stripeSize, MPI_DOUBLE,
        matrixAStripe.data(), stripeSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(matrixB.data(), stripeSize, MPI_DOUBLE,
        matrixBStripe.data(), stripeSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double tmp = 0.0;
    int nextProc = rank + 1;
    if (rank == size - 1) nextProc = 0;
    int prevProc = rank - 1;
    if (rank == 0) prevProc = size - 1;
    MPI_Status Status;
    int prevData = rank;
    for (int p = 0; p < size; p++) {
        for (int i = 0; i < stripeHeight; i++) {
            for (int j = 0; j < bCols; j++) {
                tmp = 0;
                for (int k = 0; k < stripeHeight; k++)
                    tmp += matrixAStripe[prevData * stripeHeight + i * aCols + k] * matrixBStripe[k * bCols + j];
                matrixCStripe[i * bCols + j] += tmp;
            }
        }
        prevData -= 1;
        if (prevData < 0) prevData = size - 1;
        MPI_Sendrecv_replace(matrixBStripe.data(), stripeSize, MPI_DOUBLE,
            nextProc, 0, prevProc, 0, MPI_COMM_WORLD, &Status);
    }
    MPI_Gather(matrixCStripe.data(), stripeSize, MPI_DOUBLE, matrixC.data(), stripeSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return matrixC;
}
