// Copyright 2020 Rustamov Azer
#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <limits>
#include "../../../modules/task_3/rustamov_a_dijkstra_algorithm/dijkstra_algorithm.h"

Matrix RandomGraph(int verts, int edges, bool oriented) {
    int max_edges = verts * (verts - 1) / 2;
    if (oriented) {
        max_edges *= 2;
    }
    if ((verts == 0) || (edges > max_edges)) {
        throw "Incorrect graph";
    }
    if (edges > max_edges / 2) {
        return RandomGraphDence(verts, edges, oriented);
    } else {
        return RandomGraphSparce(verts, edges, oriented);
    }
}

Matrix RandomGraphSparce(int verts, int edges, bool oriented) {
    int max_edges = verts * (verts - 1) / 2;
    if (oriented) {
        max_edges *= 2;
    }
    std::random_device rd;
    std::mt19937 mersenne(rd());
    std::uniform_real_distribution<> weight(0, 30);
    std::uniform_int_distribution<> pos(0, verts * verts - 1);
    double inf = std::numeric_limits<double>::infinity();
    Matrix matrix(verts * verts);
    std::fill(matrix.begin(), matrix.end(), inf);
    double edge_weight;
    int vert1, vert2, position;
    for (int edge = 0; edge < edges; edge++) {
        position = static_cast<int>(pos(mersenne));
        vert1 = position / verts;
        vert2 = position % verts;
        if ((vert1 == vert2) || (matrix[vert1 * verts + vert2] != inf)) {
            edge--;
        } else {
            edge_weight = static_cast<double>(weight(mersenne));
            matrix[vert1 * verts + vert2] = edge_weight;
            if (!oriented) {
                matrix[vert2 * verts + vert1] = edge_weight;
            }
        }
    }
    return matrix;
}

Matrix RandomGraphDence(int verts, int edges, bool oriented) {
    int max_edges = verts * (verts - 1) / 2;
    if (oriented) {
        max_edges *= 2;
    }
    std::random_device rd;
    std::mt19937 mersenne(rd());
    std::uniform_real_distribution<> weight(0, 30);
    std::uniform_int_distribution<> pos(0, verts * verts - 1);
    double inf = std::numeric_limits<double>::infinity();
    Matrix matrix(verts * verts);
    double edge_weight;
    for (int i = 0; i < verts; i++) {
        for (int j = 0; j < verts; j++) {
            if (i == j) {
                matrix[i * verts + j] = inf;
            } else {
                edge_weight = static_cast<double>(weight(mersenne));
                matrix[i * verts + j] = edge_weight;
            }
        }
    }
    int vert1, vert2, position;
    for (int cut = 0; cut < max_edges - edges; cut++) {
        position = static_cast<int>(pos(mersenne));
        vert1 = position / verts;
        vert2 = position % verts;
        if (matrix[vert1 * verts + vert2] == inf) {
            cut--;
        } else {
            matrix[vert1 * verts + vert2] = inf;
            if (!oriented) {
                matrix[vert2 * verts + vert1] = inf;
            }
        }
    }
    return matrix;
}

Matrix SequentialDijkstraAlgorithm(Matrix graph, int verts, int source_vertex) {
    if (graph.size() != verts * verts) {
        throw "Incorrect graph";
    }
    if (source_vertex >= verts) {
        throw "Incorrecr source vertex";
    }
    Matrix shortest_path_tree(verts * verts);
    double inf = std::numeric_limits<double>::infinity();
    std::fill(shortest_path_tree.begin(), shortest_path_tree.end(), inf);
    Matrix distance_to_verex(verts);
    std::fill(distance_to_verex.begin(), distance_to_verex.end(), inf);
    distance_to_verex[source_vertex] = 0.0;
    int closest_vert_in, closest_vert_out;
    double shortest_path;
    while (true) {
        shortest_path = inf;
        for (int vert_in = 0; vert_in < verts; vert_in++) {
            if (distance_to_verex[vert_in] != inf) {
                for (int vert_out = 0; vert_out < verts; vert_out++) {
                    if ((distance_to_verex[vert_out] == inf) &&
                        (graph[vert_in * verts + vert_out] < shortest_path)) {
                        closest_vert_in = vert_in;
                        closest_vert_out = vert_out;
                        shortest_path = graph[vert_in * verts + vert_out];
                    }
                }
            }
        }
        if (shortest_path == inf) {
            break;
        }
        shortest_path_tree[closest_vert_in * verts + closest_vert_out] = shortest_path;
        distance_to_verex[closest_vert_out] = distance_to_verex[closest_vert_in] +
           shortest_path;
    }
    return shortest_path_tree;
}


Matrix ParallelDijkstraAlgorithm(Matrix graph, int verts, int source_vertex) {
    if (graph.size() != verts * verts)
        throw "Incorrect graph";
    if (source_vertex >= verts)
        throw "Incorrecr source vertex";
    double inf = std::numeric_limits<double>::infinity();
    int procNum, procRank;
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    if (procNum == 1)
        return SequentialDijkstraAlgorithm(graph, verts, source_vertex);
    Matrix shortest_path_tree(verts * verts);
    std::fill(shortest_path_tree.begin(), shortest_path_tree.end(), inf);
    Matrix distance_to_verex(verts);
    std::fill(distance_to_verex.begin(), distance_to_verex.end(), inf);
    distance_to_verex[source_vertex] = 0.0;
    MPI_Status status;
    const int delta = verts / procNum;
    const int remain = verts % procNum;
    const int remain_for_proc = procRank < remain ? 1 : 0;
    Matrix local_graph((delta + remain_for_proc) * verts);
    std::fill(local_graph.begin(), local_graph.end(), inf);
    // Передать граф
    if (procRank == 0) {
        for (int row = 0; row < verts; row++) {
            if (row % procNum != 0) {
                MPI_Send(graph.data() + row * verts, verts, MPI_DOUBLE, row % procNum, 0, MPI_COMM_WORLD);
            } else {
                for (int vert = 0; vert < verts; vert++) {
                    local_graph[(row / procNum) * verts + vert] = graph[row * verts + vert];
                }
            }
        }
    } else {
        for (int local_vert = 0; local_vert < delta + remain_for_proc; local_vert++)
            MPI_Recv(local_graph.data() + local_vert * verts, verts, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
    }
    double local_shortest_path;
    Matrix global_shortest_paths(procNum);
    int local_closest_vert_in, local_closest_vert_out;
    bool done = false;
    int result_in, result_out;
    double result_path;
    while (true) {
        result_in = result_out = -1;
        result_path = inf;
        // Найти ближайшую вершину и грань к ней
        local_closest_vert_in = local_closest_vert_out = -1;
        local_shortest_path = inf;
        if (procRank == 0) {
            std::fill(global_shortest_paths.begin(), global_shortest_paths.end(), inf);
        }
        for (int vert_in = 0; vert_in < delta + remain_for_proc; vert_in++) {
            if (distance_to_verex[vert_in * procNum + procRank] != inf) {
                for (int vert_out = 0; vert_out < verts; vert_out++) {
                    if ((distance_to_verex[vert_out] == inf) &&
                    (local_graph[vert_in * verts + vert_out] < local_shortest_path)) {
                        local_closest_vert_in = vert_in * procNum + procRank;
                        local_closest_vert_out = vert_out;
                        local_shortest_path = local_graph[vert_in * verts + vert_out];
                    }
                }
            }
        }
        int curr_proc;
        // Передать найденные грани
        if (procRank != 0) {
            MPI_Send(&local_shortest_path, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            MPI_Bcast(&done, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
        } else {
            global_shortest_paths[0] = local_shortest_path;
            for (int proc = 1; proc < procNum; proc++) {
                MPI_Recv(global_shortest_paths.data() + proc, 1, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD, &status);
            }
            for (int proc = 0; proc < procNum; proc++) {
                if (result_path > global_shortest_paths[proc]) {
                    result_path = global_shortest_paths[proc];
                    curr_proc = proc;
                }
            }
            if (result_path == inf) {
                done = true;
            }
            MPI_Bcast(&done, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
        }
        if (done) {
            break;
        }
        MPI_Bcast(&curr_proc, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (curr_proc != 0) {
            if (curr_proc == procRank) {
                MPI_Send(&local_closest_vert_in, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
                MPI_Send(&local_closest_vert_out, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            }
            if (procRank == 0) {
                MPI_Recv(&result_in, 1, MPI_INT, curr_proc, 0, MPI_COMM_WORLD, &status);
                MPI_Recv(&result_out, 1, MPI_INT, curr_proc, 0, MPI_COMM_WORLD, &status);
            }
        } else {
            result_in = local_closest_vert_in;
            result_out = local_closest_vert_out;
        }
        // Добавить минимальную грань к дереву (если она есть)
        if (procRank == 0) {
            distance_to_verex[result_out] = distance_to_verex[result_in] + result_path;
            shortest_path_tree[result_in * verts + result_out] = result_path;
        }
        // Если грань не добавленна, завершить работу
        MPI_Bcast(distance_to_verex.data(), verts, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    return shortest_path_tree;
}
