// Copyright 2020 Kirillov Konstantin
#include <mpi.h>
#include <vector>
#include <ctime>
#include <random>
#include <algorithm>
#include "../../../modules/task_2/kirillov_k_contrast_anhancement/contrast_enhancement.h"
Matrix getRandomMatrix(int rows, int cols) {
    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));
    Matrix def(rows, std::vector<int>(cols));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            def[i][j] = gen() % 255;
        }
    }
    return def;
}

int check(int tmp) {
    if (tmp > 255) return 255;
    if (tmp < 0) return 0;
    return tmp;
}
int getSequentialContrast(Matrix mat, int rows, int cols, double alpha, int beta) {
    return check(mat[rows][cols]*alpha+beta);
}
Matrix getParallelContrast(Matrix mat, int rows, int cols, double alpha, int beta) {
    int procNum, procRank;
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    int delta = rows / procNum;
    int tmp = rows % procNum;
    int* sendbuf = new int[rows];
    int* sendcounts = new int[procNum];
    int* displs = new int[procNum];
    for (int i = 0; i < rows; i++) {
        sendbuf[i] = i;
    }
    int k = 0;
    for (int i=0; i < procNum; i++) {
        if (tmp > 0) {
            sendcounts[i] = delta+1;
            tmp--;
        } else {
            sendcounts[i] = delta;
        }
    }
    for (int i=0; i < procNum; i++) {
        displs[i] = k;
        k+=sendcounts[i];
    }
    std::vector<int>recvbuf(sendcounts[procRank]);
    MPI_Scatterv(&sendbuf[0], sendcounts, displs, MPI_INT, &recvbuf[0], sendcounts[procRank],
    MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<int>local_result(cols*recvbuf.size());
    for (int i = 0 ; i < sendcounts[procRank] ; i++) {
        for (int j =0; j < cols; j++) {
            local_result[i*cols+j] = getSequentialContrast(mat, recvbuf[i], j, alpha, beta);
        }
    }
    int *global_result = new int[rows*cols];
    int *sendcountsG = new int[procNum];
    int *displsG = new int[procNum];
    for (int i = 0; i < procNum; i++) {
        sendcountsG[i] = cols*recvbuf.size();
    }
    int k1 = 0;
    for (int i = 0; i < procNum; i++) {
        displsG[i] = k1*cols;
        k1+=sendcounts[i];
    }
    MPI_Gatherv(&local_result[0], sendcountsG[procRank], MPI_INT, &global_result[0],
    sendcountsG, displsG, MPI_INT, 0, MPI_COMM_WORLD);
    Matrix result_Mat(rows, std::vector<int>(cols));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result_Mat[i][j] = global_result[i*cols+j];
        }
    }
    return result_Mat;
}
