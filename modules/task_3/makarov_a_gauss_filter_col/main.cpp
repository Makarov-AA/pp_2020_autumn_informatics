// Copyright 2020 Makarov Alexander
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <vector>
#include <iostream>
#include "./gauss_filter_col.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

int clamp(int min, int value, int max) {
	if (value < min) return min;
	if (value > max) return max;
	return value;
}

TEST(GaussFilter, My_image_test) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double sigma = 1.;
    unsigned int radius = 1;
    double start_time, end_time;
	//reading image
	cv::Mat read_image;
	if (rank == 0){
		read_image = cv::imread("D:/Pictures/TestPicture.jpg",
		                                cv::IMREAD_GRAYSCALE );
		if (read_image.empty()) {
			std::cout << "Could not read the image" << std::endl;
		} else {
			cv::imshow("Original", read_image);
			int k = cv::waitKey(0);
		}
	}
	unsigned int w;
    unsigned int h;
	if (rank == 0) { 
		w = read_image.cols;
		h = read_image.rows;
	}
	MPI_Bcast(&w, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&h, 1, MPI_INT, 0, MPI_COMM_WORLD);
	std::vector<unsigned int> image;
    if (rank == 0) {
		cv::Mat flat = read_image.reshape(1, read_image.total() * read_image.channels());
		std::vector<int> vec = read_image.isContinuous()? flat : flat.clone();
		std::cout << "w : " << w << " h : " << h << " vec.size = " << vec.size() << std::endl;
		image.resize(vec.size());
		for (int i = 0; i < image.size(); i++) 
			image[i] = static_cast<unsigned int>(vec[i]);
    }
	/*if (rank == 0) {
        image = generate_image(w, h);
    }*/
    if (rank == 0) start_time = MPI_Wtime();
    std::vector<unsigned int> result_par = gaussFilterParallel(image, w, h,
                                                                sigma, radius);
    if (rank == 0) end_time = MPI_Wtime();
    if (rank == 0) {
        double parr_time = end_time - start_time;
        start_time = MPI_Wtime();
        std::vector<unsigned int> result_seq = gaussFilterSequential(image,
                                                        w, h, sigma, radius);
        end_time = MPI_Wtime();
        double seq_time = end_time - start_time;
		
		std::vector<unsigned char> result_seq_uchar(result_seq.size());
		for (int i = 0; i < result_seq.size(); i++)
			result_seq_uchar[i] = static_cast<unsigned char>(clamp(0, result_seq[i], 255));
		cv::Mat result_seq_img(h, w, CV_8U, result_seq_uchar.data());
		cv::imshow("Sequential result", result_seq_img);
		cv::waitKey(0);
		
		std::vector<unsigned char> result_par_uchar(result_par.size());
		for (int i = 0; i < result_par.size(); i++)
			result_par_uchar[i] = static_cast<unsigned char>(clamp(0, result_par[i], 255));
		cv::Mat result_par_img(h, w, CV_8U, result_par_uchar.data());
		cv::imshow("Parallel result", result_seq_img);
		cv::waitKey(0);
		
        std::cout << "Sequential time = " << seq_time << " s" << std::endl;
        std::cout << "Parallel time = " << parr_time << " s" << std::endl;
        ASSERT_EQ(result_par, result_seq);
    }
}

TEST(GaussFilter, 5x10_generation_test) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double sigma = 1.;
    unsigned int radius = 1;
    unsigned int w = 5;
    unsigned int h = 10;
    double start_time, end_time;
    std::vector<unsigned int> image;
    if (rank == 0) {
        image = generate_image(w, h);
    }
    if (rank == 0) start_time = MPI_Wtime();
    std::vector<unsigned int> result_par = gaussFilterParallel(image, w, h,
                                                                sigma, radius);
    if (rank == 0) end_time = MPI_Wtime();
    if (rank == 0) {
        double parr_time = end_time - start_time;
        start_time = MPI_Wtime();
        std::vector<unsigned int> result_seq = gaussFilterSequential(image,
                                                        w, h, sigma, radius);
        end_time = MPI_Wtime();
        double seq_time = end_time - start_time;
        std::cout << "Sequential time = " << seq_time << " s" << std::endl;
        std::cout << "Parallel time = " << parr_time << " s" << std::endl;
        ASSERT_EQ(result_par, result_seq);
    }
}

TEST(GaussFilter, 100x200_generation_test) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double sigma = 1.;
    unsigned int radius = 1;
    unsigned int w = 100;
    unsigned int h = 200;
    double start_time, end_time;
    std::vector<unsigned int> image;
    if (rank == 0) {
        image = generate_image(w, h);
    }
    if (rank == 0) start_time = MPI_Wtime();
    std::vector<unsigned int> result_par = gaussFilterParallel(image, w, h,
                                                                sigma, radius);
    if (rank == 0) end_time = MPI_Wtime();
    if (rank == 0) {
        double parr_time = end_time - start_time;
        start_time = MPI_Wtime();
        std::vector<unsigned int> result_seq = gaussFilterSequential(image,
                                                        w, h, sigma, radius);
        end_time = MPI_Wtime();
        double seq_time = end_time - start_time;
        std::cout << "Sequential time = " << seq_time << " s" << std::endl;
        std::cout << "Parallel time = " << parr_time << " s" << std::endl;
        ASSERT_EQ(result_par, result_seq);
    }
}

TEST(GaussFilter, 300x200_generation_test) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double sigma = 1.;
    unsigned int radius = 1;
    unsigned int w = 300;
    unsigned int h = 200;
    double start_time, end_time;
    std::vector<unsigned int> image;
    if (rank == 0) {
        image = generate_image(w, h);
    }
    if (rank == 0) start_time = MPI_Wtime();
    std::vector<unsigned int> result_par = gaussFilterParallel(image, w, h,
                                                                sigma, radius);
    if (rank == 0) end_time = MPI_Wtime();
    if (rank == 0) {
        double parr_time = end_time - start_time;
        start_time = MPI_Wtime();
        std::vector<unsigned int> result_seq = gaussFilterSequential(image,
                                                        w, h, sigma, radius);
        end_time = MPI_Wtime();
        double seq_time = end_time - start_time;
        std::cout << "Sequential time = " << seq_time << " s" << std::endl;
        std::cout << "Parallel time = " << parr_time << " s" << std::endl;
        ASSERT_EQ(result_par, result_seq);
    }
}

TEST(GaussFilter, 1000x2000_generation_test) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double sigma = 1.;
    unsigned int radius = 1;
    unsigned int w = 1000;
    unsigned int h = 2000;
    double start_time, end_time;
    std::vector<unsigned int> image;
    if (rank == 0) {
        image = generate_image(w, h);
    }
    if (rank == 0) start_time = MPI_Wtime();
    std::vector<unsigned int> result_par = gaussFilterParallel(image, w, h,
                                                                sigma, radius);
    if (rank == 0) end_time = MPI_Wtime();
    if (rank == 0) {
        double parr_time = end_time - start_time;
        start_time = MPI_Wtime();
        std::vector<unsigned int> result_seq = gaussFilterSequential(image,
                                                        w, h, sigma, radius);
        end_time = MPI_Wtime();
        double seq_time = end_time - start_time;
        std::cout << "Sequential time = " << seq_time << " s" << std::endl;
        std::cout << "Parallel time = " << parr_time << " s" << std::endl;
        ASSERT_EQ(result_par, result_seq);
    }
}

TEST(GaussFilter, 3000x2000_generation_test) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double sigma = 1.;
    unsigned int radius = 1;
    unsigned int w = 3000;
    unsigned int h = 2000;
    double start_time, end_time;
    std::vector<unsigned int> image;
    if (rank == 0) {
        image = generate_image(w, h);
    }
    if (rank == 0) start_time = MPI_Wtime();
    std::vector<unsigned int> result_par = gaussFilterParallel(image, w, h,
                                                                sigma, radius);
    if (rank == 0) end_time = MPI_Wtime();
    if (rank == 0) {
        double parr_time = end_time - start_time;
        start_time = MPI_Wtime();
        std::vector<unsigned int> result_seq = gaussFilterSequential(image,
                                                        w, h, sigma, radius);
        end_time = MPI_Wtime();
        double seq_time = end_time - start_time;
        std::cout << "Sequential time = " << seq_time << " s" << std::endl;
        std::cout << "Parallel time = " << parr_time << " s" << std::endl;
        ASSERT_EQ(result_par, result_seq);
    }
}

TEST(GaussFilter, 5000x5000_generation_test) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double sigma = 1.;
    unsigned int radius = 1;
    unsigned int w = 5000;
    unsigned int h = 5000;
    double start_time, end_time;
    std::vector<unsigned int> image;
    if (rank == 0) {
        image = generate_image(w, h);
    }
    if (rank == 0) start_time = MPI_Wtime();
    std::vector<unsigned int> result_par = gaussFilterParallel(image, w, h,
                                                                sigma, radius);
    if (rank == 0) end_time = MPI_Wtime();
    if (rank == 0) {
        double parr_time = end_time - start_time;
        start_time = MPI_Wtime();
        std::vector<unsigned int> result_seq = gaussFilterSequential(image,
                                                        w, h, sigma, radius);
        end_time = MPI_Wtime();
        double seq_time = end_time - start_time;
        std::cout << "Sequential time = " << seq_time << " s" << std::endl;
        std::cout << "Parallel time = " << parr_time << " s" << std::endl;
        ASSERT_EQ(result_par, result_seq);
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
