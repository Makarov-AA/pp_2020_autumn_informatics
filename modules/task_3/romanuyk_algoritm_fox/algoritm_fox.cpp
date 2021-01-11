// Copyright 2020 Romanuyk Sergey
#include <mpi.h>
#include <vector>
#include <random>
#include <ctime>
#include <numeric>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include"../../../modules/task_3/romanuyk_algoritm_fox/algoritm_fox.h"

MPI_Comm GridComm;
MPI_Comm ColComm;
MPI_Comm RowComm;

bool assertMatrix(const std::vector<double>& A, const std::vector<double>& B) {
    for (size_t i = 0; i < A.size(); i++) {
        if ((std::fabs(A[i] - B[i]) >= std::numeric_limits<double>::epsilon() * 1000000000.0))
            return false;
    }
    return true;
}

std::vector<double> genMatrix(int n) {
    int SIZE = n * n;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> urd(-50, 50);
    std::vector<double> arr(SIZE);
    for (int i = 0; i < SIZE; i++) {
        arr[i] = urd(gen);
    }
    return arr;
}

std::vector<double> SequentinalMultiMatrix(const std::vector<double>& A, const std::vector<double>& B, int n) {
    std::vector<double> res(n * n, 0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                res[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
    return res;
}

void createGrid(int GridSize, int procrank, int* GridCoords) {
    std::vector<int> DimSize(2, GridSize);
    std::vector<int> Periodic(2, 0);
    std::vector<int> Subdims(2);

    MPI_Cart_create(MPI_COMM_WORLD, 2, DimSize.data(), Periodic.data(), 0, &GridComm);

    MPI_Cart_coords(GridComm, procrank, 2, GridCoords);

    Subdims[0] = 0;
    Subdims[1] = 1;
    MPI_Cart_sub(GridComm, Subdims.data(), &RowComm);

    Subdims[0] = 1;
    Subdims[1] = 0;
    MPI_Cart_sub(GridComm, Subdims.data(), &ColComm);
}

void MultiplyMatrixforParallel(double* A, double* B, double* C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            double temp = 0;
            for (int k = 0; k < size; k++)
                temp += A[i*size + k] * B[k*size + j];
            C[i*size + j] += temp;
        }
    }
}

std::vector<double> MultiplyMatrixParallel(const std::vector<double>& A, const std::vector<double>& B, int size) {
    int procnum, procrank;
    MPI_Comm_size(MPI_COMM_WORLD, &procnum);
    MPI_Comm_rank(MPI_COMM_WORLD, &procrank);
    MPI_Status status;

    int GridSize = sqrt(procnum);
    std::vector<int> GridCoords(2);
    createGrid(GridSize, procrank, GridCoords.data());

    int BlockSize;
    if (procrank == 0) {
        BlockSize = size / GridSize;
    }
    MPI_Bcast(&BlockSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Datatype matrixBlock;
    MPI_Type_vector(BlockSize, BlockSize, GridSize * BlockSize, MPI_DOUBLE, &matrixBlock);
    MPI_Type_commit(&matrixBlock);

    std::vector<double> Ablock(BlockSize * BlockSize, 0);
    std::vector<double> Bblock(BlockSize * BlockSize, 0);

    if (procrank == 0) {
        for (int i = 0; i < BlockSize; i++) {
            for (int j = 0; j < BlockSize; j++) {
                Ablock[i * BlockSize + j] = A[i * size + j];
                Bblock[i * BlockSize + j] = B[i * size + j];
            }
        }
        for (int i = 1; i < procnum; i++) {
            MPI_Send(A.data() + (i % GridSize) * BlockSize + (i / GridSize) * size * BlockSize,
                1, matrixBlock, i, 0, GridComm);
            MPI_Send(B.data() + (i % GridSize) * BlockSize + (i / GridSize) * size * BlockSize,
                1, matrixBlock, i, 1, GridComm);
        }
    } else {
        MPI_Recv(Ablock.data(), BlockSize * BlockSize, MPI_DOUBLE, 0, 0, GridComm, &status);
        MPI_Recv(Bblock.data(), BlockSize * BlockSize, MPI_DOUBLE, 0, 1, GridComm, &status);
    }

    std::vector<double> Cblock(BlockSize * BlockSize, 0);

    for (int i = 0; i < GridSize; i++) {
        std::vector<double> MatrixAblock(BlockSize * BlockSize, 0);
        int pivot = (GridCoords[0] + i) % GridSize;
        if (GridCoords[1] == pivot) {
            MatrixAblock = Ablock;
        }
        MPI_Bcast(MatrixAblock.data(), BlockSize*BlockSize, MPI_DOUBLE, pivot, RowComm);

        MultiplyMatrixforParallel(MatrixAblock.data(), Bblock.data(), Cblock.data(), BlockSize);

        int NextProc = GridCoords[0] + 1;
        if (GridCoords[0] == GridSize - 1) NextProc = 0;
        int PrevProc = GridCoords[0] - 1;
        if (GridCoords[0] == 0) PrevProc = GridSize - 1;
        MPI_Sendrecv_replace(Bblock.data(), BlockSize*BlockSize, MPI_DOUBLE, PrevProc, 0,
            NextProc, 0, ColComm, &status);
    }

    std::vector<double> result(size*size);
    if (procrank == 0) {
        for (int i = 0; i < BlockSize; i++) {
            for (int j = 0; j < BlockSize; j++) {
                result[i * size + j] = Cblock[i * BlockSize + j];
            }
        }
        for (int i = 1; i < procnum; i++) {
            MPI_Recv(result.data() + (i / GridSize) * size * BlockSize + BlockSize * (i % GridSize),
                BlockSize * BlockSize, matrixBlock, i, 3, MPI_COMM_WORLD, &status);
        }
    } else {
        MPI_Send(Cblock.data(), BlockSize * BlockSize, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
    }
    MPI_Type_free(&matrixBlock);
    return result;
}
