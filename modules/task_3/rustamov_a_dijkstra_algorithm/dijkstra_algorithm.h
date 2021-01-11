// Copyright 2020 Rustamov Azer
#ifndef MODULES_TASK_3_RUSTAMOV_A_DIJKSTRA_ALGORITHM_DIJKSTRA_ALGORITHM_H_
#define MODULES_TASK_3_RUSTAMOV_A_DIJKSTRA_ALGORITHM_DIJKSTRA_ALGORITHM_H_
#include <vector>

using Matrix = std::vector<double>;

Matrix RandomGraph(int verts, int edges, bool oriented = false);
Matrix RandomGraphSparce(int verts, int edges, bool oriented = false);
Matrix RandomGraphDence(int verts, int edges, bool oriented = false);

Matrix SequentialDijkstraAlgorithm(Matrix graph, int verts, int source_vertex);
Matrix ParallelDijkstraAlgorithm(Matrix graph, int verts, int source_vertex);

#endif  // MODULES_TASK_3_RUSTAMOV_A_DIJKSTRA_ALGORITHM_DIJKSTRA_ALGORITHM_H_
