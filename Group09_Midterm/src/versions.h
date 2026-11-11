// =============================================================================
// versions.h - Interface shared by all versions of the stitching pipeline
// CSC14120 Parallel Programming - Final Project - Group 09
// Authors: Le Nhat Khoi (23127004), Nguyen Hoang Gia Bao (23127157),
//          Nguyen Hai Dang (23127165)
// Written by the team. No external code.
//
// Every version has the same signature: it receives the two RGB images and the
// parameters, runs the 6 stages, fills the result (panorama, homography,
// statistics) and the time of each stage. main.cu selects the version with
// --version N, so every version runs on exactly the same input.
// =============================================================================
#pragma once
#include "common.h"

typedef void (*StitchFunction)(const ImageRGB & imgA, const ImageRGB & imgB,
		const Params & params, StitchResult & result);

// V0: sequential, single-threaded C++ (v0_orb.cpp)
void stitchV0(const ImageRGB & imgA, const ImageRGB & imgB,
		const Params & params, StitchResult & result);

// V1, V2, ... will be added for the final submission (v1_*.cu, v2_*.cu, ...)

// Returns the function of version v, or nullptr if this version does not exist yet
inline StitchFunction getStitchFunction(int version)
{
	switch (version)
	{
	case 0: return stitchV0;
	default: return nullptr;
	}
}

// Helpers implemented in v0_orb.cpp and reused by every version (host side)
#include <vector>
void makeGaussianKernel(float sigma, int radius, std::vector<float> & kernel);
void makeBriefPattern(unsigned seed, int8_t pattern[256 * 4]);
void makeRansacSamples(unsigned seed, int iters, int n, std::vector<int> & samples);
void applyHomography(const double H[9], double x, double y, double & outX, double & outY);
bool invertHomography(const double H[9], double Hinv[9]);
