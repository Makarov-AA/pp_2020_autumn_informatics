// Copyright 2020 Kumbrasev Mark
#include <mpi.h>
#include <iostream>
#include "../../../modules/task_3/kumbrasev_m_bin_img_label/bin_img_labeling.h"

void labeling(image* img) {
  int rank, size;
  int c_str, new_c_str, label, c_label = 0, res_exp = 0, rest;
  bool flag = 0;
  int* img_arr, *loc_img_arr;
  image* loc_img;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Status status;
  c_str = img->height / size;
  rest = img->height % size;

  if (rank == 0) {
    new_c_str = c_str + rest;
  } else {
    new_c_str = c_str;
  }

  label = rank * 100 + 2;
  img_arr = new int[img->height * img->width];
  loc_img_arr = new int[new_c_str * img->width];
  loc_img = new image(new_c_str, img->width);

  if (rank == 0) {
    for (int i = 0; i < img->height; i++) {
      for (int j = 0; j < img->width; j++) {
        img_arr[i * img->width + j] = img->data[i][j];
      }
    }

    for (int i = 0; i < (new_c_str) * img->width; i++) {
      loc_img_arr[i] = img_arr[i];
    }

    for (int i = 1; i < size; i++) {
      MPI_Send(&img_arr[i * img->width * c_str + rest * img->width], c_str * img->width, MPI_INT, i, 0, MPI_COMM_WORLD);
    }
  } else {
    MPI_Recv(&loc_img_arr[0], c_str * img->width, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    }

  for (int i = 0; i < new_c_str; i++) {
    for (int j = 0; j < img->width; j++) {
      loc_img->data[i][j] = loc_img_arr[i * img->width + j];
    }
  }

  for (int i = 0; i < loc_img->height; i++) {
    for (int j = 0; j < img->width; j++) {
      if (loc_img->data[i][j] == 1) {
        flag = 1;
        loc_img->data[i][j] = label;
        if (i + 1 < new_c_str && loc_img->data[i + 1][j] == 1) {
          int k = i + 1;
          while (k < new_c_str && loc_img->data[k][j] == 1) {
            loc_img->data[k][j] = label;
            if (j - 1 >= 0 && loc_img->data[k][j - 1]) {
              int l = j - 1;
              while (l >= 0 && loc_img->data[k][l]) {
                loc_img->data[k][l] = label;
                l--;
              }
            }
            if (j + 1 < img->width && loc_img->data[k][j + 1] == 1) {
              int l = j + 1;
              while (l < img->width && loc_img->data[k][l]) {
                loc_img->data[k][l] = label;
                l++;
              }
            }
            k++;
          }
        }
      } else {
        if (flag) {
          flag = 0;
          label++;
          c_label++;
        }
      }
    }
  }

  for (int i = 0; i < new_c_str; i++) {
    for (int j = 0; j < img->width; j++) {
      loc_img_arr[i * img->width + j] = loc_img->data[i][j];
    }
  }

  if (rank == 0) {
    for (int i = 0; i < new_c_str * img->width; i++) {
      img_arr[i] = loc_img_arr[i];
    }
  }

  if (rank != 0) {
      MPI_Send(&loc_img_arr[0], new_c_str * img->width, MPI_INT, 0, 0, MPI_COMM_WORLD);
  } else {
    for (int i = 1; i < size; i++) {
      MPI_Recv(&img_arr[i * img->width * c_str + rest * img->width],
      new_c_str * img->width, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
    }
  }

  MPI_Reduce(&c_label, &res_exp, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    for (int i = 0; i < img->height; i++) {
      for (int j = 0; j < img->width; j++) {
        img->data[i][j] = img_arr[i * img->width + j];
      }
    }

    flag = 0;
    for (int i = new_c_str - 1; i < img->height - c_str; i += c_str) {
      for (int j = 0; j < img->width; j++) {
        if (img->data[i][j] != 0 && img->data[i + 1][j] != 0) {
          flag = 1;
          int k = i + 1;
          while (k < img->height && img->data[k][j] != 0) {
            img->data[k][j] = img->data[i][j];
            if (j > 0 && img->data[k][j - 1] != 0) {
              int l = j - 1;
              while (l >= 0 && img->data[k][l] != 0) {
                img->data[k][l] = img->data[i][j];
                l--;
              }
            }

            if (j < img->width - 1 && img->data[k][j + 1] != 0) {
              int l = j + 1;
              while (l < img->width && img->data[k][l] != 0) {
                img->data[k][l] = img->data[i][j];
                img->data[k][l] = img->data[i][j];
                l++;
              }
            }
            k++;
          }
        } else {
          if (flag) {
            flag = 0;
            res_exp--;
          }
        }
      }
    }
  }
  img->count = res_exp;
  delete loc_img;
}
