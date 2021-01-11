// Copyright 2020 Rustamov Azer
#include <mpi.h>
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <vector>
#include <limits>
#include "./dijkstra_algorithm.h"

#define EPSILON 0.000001

TEST(Dijkstra_Algorithm, Incorrect_Graph) {
    int procNum, procRank;
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    int verts = 2, edges = 10;
    Matrix graph;
    if (procRank == 0) {
        ASSERT_ANY_THROW(RandomGraph(verts, edges));
    }
}

TEST(Dijkstra_Algorithm, Correct_Answer_Unoriented_5_Seq) {
    int procRank;
    double inf = std::numeric_limits<double>::infinity();
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    int verts = 5;
    Matrix graph = {inf, inf, 7,   3,   9,
                    inf, inf, 2,   15,  4,
                    7,   2,   inf, 10,  inf,
                    3,   15,  10,  inf, inf,
                    9,   4,   inf, inf, inf };
    if (procRank == 0) {
        Matrix spt = SequentialDijkstraAlgorithm(graph, verts, 0);
        Matrix exspt = {inf, inf, 7,   3,   inf,
                        inf, inf, inf, inf, 4,
                        inf, 2,   inf, inf, inf,
                        inf, inf, inf, inf, inf,
                        inf, inf, inf, inf, inf };
        for (int pos = 0; pos < verts * verts; pos++) {
            if ((spt[pos] != inf) || (exspt[pos] != inf)) {
                ASSERT_NEAR(spt[pos], exspt[pos], EPSILON);
            }
        }
    }
}

TEST(Dijkstra_Algorithm, Correct_Answer_Unoriented_5_Par) {
    int procRank;
    double inf = std::numeric_limits<double>::infinity();
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    int verts = 5;
    Matrix graph = {inf, inf, 7,   3,   9,
                    inf, inf, 2,   15,  4,
                    7,   2,   inf, 10,  inf,
                    3,   15,  10,  inf, inf,
                    9,   4,   inf, inf, inf };
    Matrix spt = ParallelDijkstraAlgorithm(graph, verts, 0);
    if (procRank == 0) {
        Matrix exspt = {inf, inf, 7,   3,   inf,
                        inf, inf, inf, inf, 4,
                        inf, 2,   inf, inf, inf,
                        inf, inf, inf, inf, inf,
                        inf, inf, inf, inf, inf };
        for (int pos = 0; pos < verts * verts; pos++) {
            if ((spt[pos] != inf) || (exspt[pos] != inf)) {
                ASSERT_NEAR(spt[pos], exspt[pos], EPSILON);
            }
        }
    }
}

TEST(Dijkstra_Algorithm, 5_9_Unoriented_Source_0) {
    int procRank;
    double inf = std::numeric_limits<double>::infinity();
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    int verts = 5;
    int edges = 9;
    bool is_oriented = false;
    Matrix graph = RandomGraph(verts, edges, is_oriented);
    Matrix spt = ParallelDijkstraAlgorithm(graph, verts, 0);
    if (procRank == 0) {
        Matrix seq_spt = SequentialDijkstraAlgorithm(graph, verts, 0);
        for (int pos = 0; pos < verts * verts; pos++) {
            if ((spt[pos] != inf) || (seq_spt[pos] != inf)) {
                ASSERT_NEAR(spt[pos], seq_spt[pos], EPSILON);
            }
        }
    }
}

TEST(Dijkstra_Algorithm, 5_9_Oriented_Source_0) {
    int procRank;
    double inf = std::numeric_limits<double>::infinity();
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    int verts = 5;
    int edges = 9;
    bool is_oriented = true;
    Matrix graph = RandomGraph(verts, edges, is_oriented);
    Matrix spt = ParallelDijkstraAlgorithm(graph, verts, 0);
    if (procRank == 0) {
        Matrix seq_spt = SequentialDijkstraAlgorithm(graph, verts, 0);
        for (int pos = 0; pos < verts * verts; pos++) {
            if ((spt[pos] != inf) || (seq_spt[pos] != inf)) {
                ASSERT_NEAR(spt[pos], seq_spt[pos], EPSILON);
            }
        }
    }
}


TEST(Dijkstra_Algorithm, 5_9_Unoriented_Source_1) {
    int procRank;
    double inf = std::numeric_limits<double>::infinity();
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    int verts = 5;
    int edges = 9;
    bool is_oriented = false;
    Matrix graph = RandomGraph(verts, edges, is_oriented);
    Matrix spt = ParallelDijkstraAlgorithm(graph, verts, 1);
    if (procRank == 0) {
        Matrix seq_spt = SequentialDijkstraAlgorithm(graph, verts, 1);
        for (int pos = 0; pos < verts * verts; pos++) {
            if ((spt[pos] != inf) || (seq_spt[pos] != inf)) {
                ASSERT_NEAR(spt[pos], seq_spt[pos], EPSILON);
            }
        }
    }
}

TEST(Dijkstra_Algorithm, 5_9_Oriented_Source_3) {
    int procRank;
    double inf = std::numeric_limits<double>::infinity();
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    int verts = 5;
    int edges = 9;
    bool is_oriented = true;
    Matrix graph = RandomGraph(verts, edges, is_oriented);
    Matrix spt = ParallelDijkstraAlgorithm(graph, verts, 3);
    if (procRank == 0) {
        Matrix seq_spt = SequentialDijkstraAlgorithm(graph, verts, 3);
        for (int pos = 0; pos < verts * verts; pos++) {
            if ((spt[pos] != inf) || (seq_spt[pos] != inf)) {
                ASSERT_NEAR(spt[pos], seq_spt[pos], EPSILON);
            }
        }
    }
}

TEST(Dijkstra_Algorithm, 10_30_Unoriented_Source_0) {
    int procRank;
    double inf = std::numeric_limits<double>::infinity();
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    int verts = 10;
    int edges = 30;
    bool is_oriented = false;
    Matrix graph = RandomGraph(verts, edges, is_oriented);
    double time_start_par = MPI_Wtime();
    Matrix spt = ParallelDijkstraAlgorithm(graph, verts, 0);
    if (procRank == 0) {
        double time_end_par = MPI_Wtime();
        std::cout << "PAR: " << time_end_par - time_start_par << std::endl;
    }
    if (procRank == 0) {
        double time_start_seq = MPI_Wtime();
        Matrix seq_spt = SequentialDijkstraAlgorithm(graph, verts, 0);
        double time_end_seq = MPI_Wtime();
        std::cout << "SEQ: " << time_end_seq - time_start_seq << std::endl;
        for (int pos = 0; pos < verts * verts; pos++) {
            if ((spt[pos] != inf) || (seq_spt[pos] != inf)) {
                ASSERT_NEAR(spt[pos], seq_spt[pos], EPSILON);
            }
        }
    }
}

TEST(Dijkstra_Algorithm, 100_400_Unoriented_Source_0) {
    int procRank;
    double inf = std::numeric_limits<double>::infinity();
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    int verts = 100;
    int edges = 400;
    bool is_oriented = false;
    Matrix graph = RandomGraph(verts, edges, is_oriented);
    double time_start_par = MPI_Wtime();
    Matrix spt = ParallelDijkstraAlgorithm(graph, verts, 0);
    if (procRank == 0) {
        double time_end_par = MPI_Wtime();
        std::cout << "PAR: " << time_end_par - time_start_par << std::endl;
    }
    if (procRank == 0) {
        double time_start_seq = MPI_Wtime();
        Matrix seq_spt = SequentialDijkstraAlgorithm(graph, verts, 0);
        double time_end_seq = MPI_Wtime();
        std::cout << "SEQ: " << time_end_seq - time_start_seq << std::endl;
        for (int pos = 0; pos < verts * verts; pos++) {
            if ((spt[pos] != inf) || (seq_spt[pos] != inf)) {
                ASSERT_NEAR(spt[pos], seq_spt[pos], EPSILON);
            }
        }
    }
}

TEST(Dijkstra_Algorithm, 200_400_Unoriented_Source_0) {
    int procRank;
    double inf = std::numeric_limits<double>::infinity();
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    int verts = 200;
    int edges = 400;
    bool is_oriented = false;
    Matrix graph = RandomGraph(verts, edges, is_oriented);
    double time_start_par = MPI_Wtime();
    Matrix spt = ParallelDijkstraAlgorithm(graph, verts, 0);
    if (procRank == 0) {
        double time_end_par = MPI_Wtime();
        std::cout << "PAR: " << time_end_par - time_start_par << std::endl;
    }
    if (procRank == 0) {
        double time_start_seq = MPI_Wtime();
        Matrix seq_spt = SequentialDijkstraAlgorithm(graph, verts, 0);
        double time_end_seq = MPI_Wtime();
        std::cout << "SEQ: " << time_end_seq - time_start_seq << std::endl;
        for (int pos = 0; pos < verts * verts; pos++) {
            if ((spt[pos] != inf) || (seq_spt[pos] != inf)) {
                ASSERT_NEAR(spt[pos], seq_spt[pos], EPSILON);
            }
        }
    }
}

TEST(Dijkstra_Algorithm, 500_500_Unoriented_Source_0) {
    int procRank;
    double inf = std::numeric_limits<double>::infinity();
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    int verts = 500;
    int edges = 500;
    bool is_oriented = false;
    Matrix graph = RandomGraph(verts, edges, is_oriented);
    double time_start_par = MPI_Wtime();
    Matrix spt = ParallelDijkstraAlgorithm(graph, verts, 0);
    if (procRank == 0) {
        double time_end_par = MPI_Wtime();
        std::cout << "PAR: " << time_end_par - time_start_par << std::endl;
    }
    if (procRank == 0) {
        double time_start_seq = MPI_Wtime();
        Matrix seq_spt = SequentialDijkstraAlgorithm(graph, verts, 0);
        double time_end_seq = MPI_Wtime();
        std::cout << "SEQ: " << time_end_seq - time_start_seq << std::endl;
        for (int pos = 0; pos < verts * verts; pos++) {
            if ((spt[pos] != inf) || (seq_spt[pos] != inf)) {
                ASSERT_NEAR(spt[pos], seq_spt[pos], EPSILON);
            }
        }
    }
}

TEST(Dijkstra_Algorithm, 500_70000_Unoriented_Source_0) {
    int procRank;
    double inf = std::numeric_limits<double>::infinity();
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    int verts = 500;
    int edges = 70000;
    bool is_oriented = false;
    Matrix graph = RandomGraph(verts, edges, is_oriented);
    double time_start_par = MPI_Wtime();
    Matrix spt = ParallelDijkstraAlgorithm(graph, verts, 0);
    if (procRank == 0) {
        double time_end_par = MPI_Wtime();
        std::cout << "PAR: " << time_end_par - time_start_par << std::endl;
    }
    if (procRank == 0) {
        double time_start_seq = MPI_Wtime();
        Matrix seq_spt = SequentialDijkstraAlgorithm(graph, verts, 0);
        double time_end_seq = MPI_Wtime();
        std::cout << "SEQ: " << time_end_seq - time_start_seq << std::endl;
        for (int pos = 0; pos < verts * verts; pos++) {
            if ((spt[pos] != inf) || (seq_spt[pos] != inf)) {
                ASSERT_NEAR(spt[pos], seq_spt[pos], EPSILON);
            }
        }
    }
}

TEST(Dijkstra_Algorithm, 500_220000_Oriented_Source_0) {
    int procRank;
    double inf = std::numeric_limits<double>::infinity();
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    int verts = 500;
    int edges = 220000;
    bool is_oriented = true;
    Matrix graph = RandomGraph(verts, edges, is_oriented);
    double time_start_par = MPI_Wtime();
    Matrix spt = ParallelDijkstraAlgorithm(graph, verts, 0);
    if (procRank == 0) {
        double time_end_par = MPI_Wtime();
        std::cout << "PAR: " << time_end_par - time_start_par << std::endl;
    }
    if (procRank == 0) {
        double time_start_seq = MPI_Wtime();
        Matrix seq_spt = SequentialDijkstraAlgorithm(graph, verts, 0);
        double time_end_seq = MPI_Wtime();
        std::cout << "SEQ: " << time_end_seq - time_start_seq << std::endl;
        for (int pos = 0; pos < verts * verts; pos++) {
            if ((spt[pos] != inf) || (seq_spt[pos] != inf)) {
                ASSERT_NEAR(spt[pos], seq_spt[pos], EPSILON);
            }
        }
    }
}

TEST(Dijkstra_Algorithm, 1000_998000_Oriented_Source_0) {
    int procRank;
    double inf = std::numeric_limits<double>::infinity();
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    int verts = 1000;
    int edges = 998000;
    bool is_oriented = true;
    Matrix graph = RandomGraph(verts, edges, is_oriented);
    double time_start_par = MPI_Wtime();
    Matrix spt = ParallelDijkstraAlgorithm(graph, verts, 0);
    if (procRank == 0) {
        double time_end_par = MPI_Wtime();
        std::cout << "PAR: " << time_end_par - time_start_par << std::endl;
    }
    if (procRank == 0) {
        double time_start_seq = MPI_Wtime();
        Matrix seq_spt = SequentialDijkstraAlgorithm(graph, verts, 0);
        double time_end_seq = MPI_Wtime();
        std::cout << "SEQ: " << time_end_seq - time_start_seq << std::endl;
        for (int pos = 0; pos < verts * verts; pos++) {
            if ((spt[pos] != inf) || (seq_spt[pos] != inf)) {
                ASSERT_NEAR(spt[pos], seq_spt[pos], EPSILON);
            }
        }
    }
}

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);

    ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();

    listeners.Release(listeners.default_result_printer());
    listeners.Release(listeners.default_xml_generator());

    listeners.Append(new GTestMPIListener::MPIMinimalistPrinter);
    return RUN_ALL_TESTS();
}
