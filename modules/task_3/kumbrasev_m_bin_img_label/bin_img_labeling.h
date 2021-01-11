// Copyright 2020 Kumbrasev Mark
#ifndef MODULES_TASK_3_KUMBRASEV_M_BIN_IMG_LABEL_BIN_IMG_LABELING_H_
#define MODULES_TASK_3_KUMBRASEV_M_BIN_IMG_LABEL_BIN_IMG_LABELING_H_

#include <mpi.h>
#include <iostream>

struct image {
  int height, width, count;
  int** data;

  image(int _height, int _width) : height(_height), width(_width), count(1) {
    data = new int*[height];
    for (int i = 0; i < height; i++) {
      data[i] = new int[width];
      for (int j = 0; j < width; j++) {
        data[i][j] = 0;
      }
    }
  }

  friend std::ostream& operator<< (std::ostream& os, const image& img) {
    for (int i = 0; i < img.height; i++) {
      for (int j = 0; j < img.width; j++) {
        os << img.data[i][j] << "\t";
      }
      os << std::endl;
    }
    return os;
  }
};

void labeling(image* img);

#endif  // MODULES_TASK_3_KUMBRASEV_M_BIN_IMG_LABEL_BIN_IMG_LABELING_H_
