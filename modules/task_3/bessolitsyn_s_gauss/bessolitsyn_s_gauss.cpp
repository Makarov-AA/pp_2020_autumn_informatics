// Copyright 2020 Bessolitsyn Sergey
#include <mpi.h>
#include <string>
#include <random>
#include <ctime>
#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>
#include <cstring>
#include "../../../modules/task_3/bessolitsyn_s_gauss/bessolitsyn_s_gauss.h"

const std::vector<std::vector<double>> gauss_matrix = { {1, 2, 1},
                                                        {2, 4, 2},
                                                        {1, 2, 1} };

std::vector<double> filter_seq(std::vector<double> input_image, int w, int h) {
    std::vector<double> bordered_image((w + 2) * (h + 2));
    double t = MPI_Wtime();
    for (int i = 0; i < h + 2; i++) {
        for (int j = 0; j < w + 2; j++) {
            if ((i == 0) || (j == 0) || (i == h + 1) || (j == w + 1)) {
                bordered_image[i * (w + 2) + j] = 0;
            } else {
                bordered_image[i * (w + 2) + j] = input_image[(i - 1) * w + j - 1];
            }
        }
    }

    int pos = 0;
    double sum = 0;
    for (int i = 1; i < h + 1; i++) {
        for (int j = 1; j < w + 1; j++) {
            sum = 0;
            for (int ii = -1; ii < 2; ++ii)
                for (int jj = -1; jj < 2; ++jj) {
                    sum += bordered_image[(i + ii) * (w + 2) + j + jj] * gauss_matrix[ii + 1][jj + 1];
                }
            input_image[pos] = sum / 16;
            ++pos;
        }
    }
    int p_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &p_rank);
    std::cout << "Procces " << p_rank << " filter time:" << -t + MPI_Wtime() << std::endl;
    return input_image;
}

std::vector<double> filter_par(std::vector<double> input_image, int w, int h) {
    int p_size, p_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &p_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &p_rank);

    int n = w * h;

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&w, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&h, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int* array_counts = new int[p_size];
    int* array_dist = new int[p_size];

    if (p_size != 1) {
        if (p_rank == 0) {
            int length = 0;
            int i;
            for (i = 0; i < ((h % p_size == 0) ? p_size : (p_size - 1)); i++) {
                array_counts[i] = h / ((h % p_size == 0) ? p_size : (p_size - 1)) * w;
                array_dist[i] = length;
                length += h / ((h % p_size == 0) ? p_size : (p_size - 1)) * w;
            }
            if (h % p_size != 0) {
                array_counts[i] = (h % (p_size - 1)) * w;
                array_dist[i] = length;
            }
        }
    } else {
        array_counts[0] = h * w;
        array_dist[0] = 0;
    }

    int local_image_size;

    if (p_size != 1) {
        if ((p_rank < p_size && h % p_size == 0) || (p_rank < (p_size - 1) && h % p_size != 0)) {
            local_image_size = h / ((h % p_size == 0) ? p_size : (p_size - 1)) * w;
        } else {
            local_image_size = h % ((h % p_size == 0) ? p_size : (p_size - 1)) * w;
        }
    } else {
        local_image_size = h * w;
    }


    std::vector<double> local_image(local_image_size);

    MPI_Scatterv(input_image.data(), array_counts, array_dist, MPI_DOUBLE,
        local_image.data(), local_image_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    local_image = filter_seq(local_image, w, local_image_size / w);

    std::vector<double> res;
    if (p_rank == 0) {
        res.resize(w * h);
    }
    MPI_Gatherv(local_image.data(), local_image_size, MPI_DOUBLE,
        res.data(), array_counts, array_dist, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (p_size != 1 && p_rank == 0) {
        for (int i = 0; i < p_size - 1; i++) {
            for (int j = 0; j < w; j++) {
                res[local_image_size + i * local_image_size + j] =
                input_image[local_image_size + i * local_image_size +j];
                if (local_image_size + i  * local_image_size + j + w < static_cast<int>(res.size()))
                    res[local_image_size + i  * local_image_size + j + w] =
                    input_image[local_image_size + i  * local_image_size + j + w];
                res[local_image_size + i  * local_image_size + j - w] =
                input_image[local_image_size + i  * local_image_size + j - w];
            }
        }
    }

    return res;
}
