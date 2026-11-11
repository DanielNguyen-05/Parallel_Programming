// =============================================================================
// common.h - Shared data types and parameters for the panorama stitching project
// CSC14120 Parallel Programming - Final Project - Group 09
// Authors: Le Nhat Khoi (23127004), Nguyen Hoang Gia Bao (23127157),
//          Nguyen Hai Dang (23127165)
// Written by the team (with the help of an AI assistant, see the notebook's
// AI usage declaration). No external code.
//
// Every version (V0 sequential, V1 basic CUDA, V2+ optimized) uses the SAME
// types and the SAME parameters below, so that results and timings can be
// compared fairly.
// =============================================================================
#pragma once
#include <stdint.h>
#include <vector>
#include <string>

// ----------------------------------------------------------------------------- images
// RGB image, 3 bytes per pixel (r, g, b), row-major (interleaved channels)
struct ImageRGB {
	int width = 0, height = 0;
	std::vector<uint8_t> data; // size = width * height * 3
};

// Single-channel 8-bit image, row-major
struct ImageGray {
	int width = 0, height = 0;
	std::vector<uint8_t> data; // size = width * height
};

// ----------------------------------------------------------------------------- features
struct Keypoint {
	float x, y;     // position (pixels) in the image
	float score;    // FAST score (higher = stronger corner)
	float angle;    // orientation (radians), from the intensity centroid
};

// 256-bit binary descriptor (rotated BRIEF), stored as 8 x 32-bit words
struct Descriptor {
	uint32_t w[8];
};

struct Match {
	int idxA;       // index of the keypoint in image A
	int idxB;       // index of the keypoint in image B
	int dist;       // Hamming distance of the best match
};

// ----------------------------------------------------------------------------- parameters
// One set of parameters, used by every version.
struct Params {
	// Stage 1: preprocessing
	float gaussSigma = 1.5f;   // sigma of the Gaussian blur (separable)
	int   gaussRadius = 4;     // kernel size = 2 * radius + 1 = 9 taps

	// Stage 2: keypoint detection (FAST-9 + 3x3 non-maximum suppression + top-K)
	int fastThreshold = 20;    // intensity difference threshold of the segment test
	int maxKeypoints = 2000;   // K: keypoints kept per image (strongest scores)
	int border = 20;           // keypoints closer to the border are dropped (descriptor patch)

	// Stage 3: description (ORB-style: intensity-centroid orientation + rotated BRIEF)
	int orientRadius = 15;     // radius of the circular patch used for the orientation
	unsigned briefSeed = 2026; // seed of the (fixed) BRIEF sampling pattern

	// Stage 4: matching (brute force, Hamming distance)
	float ratio = 0.8f;        // Lowe's ratio test: best < ratio * second best
	int maxHamming = 80;       // reject matches whose best distance is larger

	// Stage 5: homography (RANSAC + 4-point DLT, then least-squares refinement on inliers)
	int ransacIters = 2000;    // number of hypotheses (fixed, so every version does the same work)
	float ransacThreshold = 3.0f; // inlier threshold in pixels at 1280 px width; scaled with width
	unsigned ransacSeed = 12345;  // seed used to pre-generate the 4-point samples
	int refineIters = 3;       // least-squares refinement iterations on the inliers
	int minInliers = 12;       // fewer inliers than this = stitching failure

	// Stage 6: warping and blending
	int maxCanvasFactor = 4;   // canvas larger than factor x (A's width / height) = failure
};

// ----------------------------------------------------------------------------- timing
enum Stage {
	STAGE_PREPROCESS = 0, // 1. grayscale + Gaussian blur (both images)
	STAGE_DETECT,         // 2. FAST + NMS + top-K (both images)
	STAGE_DESCRIBE,       // 3. orientation + rotated BRIEF (both images)
	STAGE_MATCH,          // 4. brute-force Hamming matching + ratio test
	STAGE_HOMOGRAPHY,     // 5. RANSAC + refinement
	STAGE_WARP,           // 6. inverse warping + feathering
	NUM_STAGES
};

static const char * const STAGE_NAMES[NUM_STAGES] = {
	"1_preprocess", "2_detect", "3_describe", "4_match", "5_homography", "6_warp_blend"
};

struct StageTimes {
	double ms[NUM_STAGES] = {0};  // time of each stage (milliseconds)
	double h2dMs = 0, d2hMs = 0;  // host <-> device transfers (0 for V0)
	double hostSetupMs = 0;       // part of the stage times that stays on the host in every version:
	                              // Gaussian kernel, BRIEF pattern and RANSAC samples generation
	double totalMs = 0;           // end-to-end time (image loading/saving excluded)
};

// ----------------------------------------------------------------------------- result
struct StitchResult {
	bool ok = false;
	std::string message;          // reason of a failure
	ImageRGB panorama;
	double H[9];                  // homography mapping B -> A (row-major, H[8] = 1)
	int numKeypointsA = 0, numKeypointsB = 0;
	int numMatches = 0, numInliers = 0;
	int offsetX = 0, offsetY = 0; // position of A's origin inside the panorama
	StageTimes times;
	// For figures in the report (optional dump)
	std::vector<Keypoint> kpA, kpB;
	std::vector<Match> matches;
	std::vector<uint8_t> inlierMask;
};
