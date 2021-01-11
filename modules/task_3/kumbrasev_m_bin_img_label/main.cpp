// Copyright 2020 Kumbrasev Mark
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <time.h>
#include <iostream>
#include "./bin_img_labeling.h"

TEST(label, test1) {
  int rank, size, exp_count = 2;
  double start_time, end_time;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  image img(4, 4);

  if (rank == 0) {
    img.data[0][0] = 1;
    img.data[1][0] = 1;
    img.data[0][3] = 1;
    img.data[0][2] = 1;
    img.data[1][3] = 1;
    img.data[3][1] = 1;
    img.data[3][2] = 1;
    img.data[3][3] = 1;
  }

  start_time = MPI_Wtime();
  labeling(&img);
  end_time = MPI_Wtime();

  printf("\tTime  = %f\n", end_time - start_time);

  if (rank == 0) {
    ASSERT_EQ(exp_count, 2);
  }
}

TEST(label, test2) {
  int rank, size, exp_count = 3;
  double start_time, end_time;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  image img(6, 6);

  if (rank == 0) {
    img.data[0][0] = 1;
    img.data[1][0] = 1;
    img.data[0][1] = 1;
    img.data[1][1] = 1;
    img.data[0][4] = 1;
    img.data[0][5] = 1;
    img.data[1][4] = 1;
    img.data[4][3] = 1;
    img.data[4][2] = 1;
    img.data[5][5] = 1;
  }
  start_time = MPI_Wtime();
  labeling(&img);
  end_time = MPI_Wtime();

  printf("\tTime  = %f\n", end_time - start_time);

  if (rank == 0) {
    ASSERT_EQ(exp_count, 3);
  }
}

TEST(label, test3) {
  int rank, size, exp_count = 1;
  double start_time, end_time;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  image img(4, 4);

  if (rank == 0) {
    img.data[0][0] = 1;
    img.data[1][0] = 1;
    img.data[2][0] = 1;
    img.data[3][0] = 1;
    img.data[3][2] = 1;
    img.data[3][3] = 1;
    img.data[1][1] = 1;
    img.data[1][2] = 1;
  }

  start_time = MPI_Wtime();
  labeling(&img);
  end_time = MPI_Wtime();

  printf("\tTime  = %f\n", end_time - start_time);

  if (rank == 0) {
    ASSERT_EQ(exp_count, 1);
  }
}

TEST(label, test4) {
  int rank, size, exp_count = 8;
  double start_time, end_time;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  image img(6, 6);

  if (rank == 0) {
    img.data[0][0] = 1;
    img.data[1][1] = 1;
    img.data[2][2] = 1;
    img.data[3][3] = 1;
    img.data[4][4] = 1;
    img.data[5][5] = 1;
    img.data[4][1] = 1;
    img.data[0][5] = 1;
    img.data[1][4] = 1;
    img.data[2][3] = 1;
    img.data[3][2] = 1;
    img.data[5][0] = 1;
  }
  start_time = MPI_Wtime();
  labeling(&img);
  end_time = MPI_Wtime();

  printf("\tTime  = %f\n", end_time - start_time);

  if (rank == 0) {
    ASSERT_EQ(exp_count, 8);
  }
}

TEST(label, test5) {
  int rank, size, exp_count = 2;
  double start_time, end_time;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  image img(6, 6);

  if (rank == 0) {
    img.data[0][0] = 1;
    img.data[1][0] = 1;
    img.data[2][0] = 1;
    img.data[3][0] = 1;
    img.data[4][0] = 1;
    img.data[5][0] = 1;
    img.data[0][1] = 1;
    img.data[0][2] = 1;
    img.data[0][3] = 1;
    img.data[0][4] = 1;
    img.data[0][5] = 1;
    img.data[4][2] = 1;
    img.data[4][3] = 1;
    img.data[4][4] = 1;
    img.data[3][4] = 1;
    img.data[2][4] = 1;
  }
  start_time = MPI_Wtime();
  labeling(&img);
  end_time = MPI_Wtime();

  printf("\tTime  = %f\n", end_time - start_time);

  if (rank == 0) {
    ASSERT_EQ(exp_count, 2);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);

  ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
  ::testing::TestEventListeners& listeners =
    ::testing::UnitTest::GetInstance()->listeners();

  listeners.Release(listeners.default_result_printer());
  listeners.Release(listeners.default_xml_generator());

  listeners.Append(new GTestMPIListener::MPIMinimalistPrinter);
  return RUN_ALL_TESTS();
}
