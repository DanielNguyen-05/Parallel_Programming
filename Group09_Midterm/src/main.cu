// =============================================================================
// main.cu - Command-line driver: loads the images, runs one version of the
//           stitching pipeline several times, reports per-stage timings,
//           checks the homography against a ground truth and saves the panorama.
// CSC14120 Parallel Programming - Final Project - Group 09
// Authors: Le Nhat Khoi (23127004), Nguyen Hoang Gia Bao (23127157),
//          Nguyen Hai Dang (23127165)
// Written by the team. Image I/O uses stb (see stb_impl.cpp).
//
// Usage:
//   ./stitch <left_image> <right_image> <output_image> [--version N] [--runs N]
//            [--gt H_gt.txt] [--json result.json] [--dump matches.txt]
//   --version N : 0 = sequential (default), 1, 2, ... = CUDA versions (final submission)
//   --runs N    : number of timed runs (default 5); one extra warm-up run is done first
//   --gt FILE   : ground-truth homography B -> A (9 numbers) -> homography error
//   --json FILE : write all results (timings, statistics) as JSON for the notebook
//   --dump FILE : write keypoints and matches (for the figures of the report)
// Image B (right) is warped onto image A's (left) frame.
// =============================================================================
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <float.h>
#include <string>
#include <vector>
#include "common.h"
#include "versions.h"
#include "cuda_utils.cuh"
#include "stb_image.h"
#include "stb_image_write.h"

static bool loadImage(const char * fileName, ImageRGB & img)
{
	int w, h, c;
	uint8_t * pixels = stbi_load(fileName, &w, &h, &c, 3); // force RGB
	if (pixels == NULL)
	{
		fprintf(stderr, "Cannot read %s: %s\n", fileName, stbi_failure_reason());
		return false;
	}
	img.width = w;
	img.height = h;
	img.data.assign(pixels, pixels + (size_t)w * h * 3);
	stbi_image_free(pixels);
	return true;
}

static bool endsWith(const std::string & s, const char * suffix)
{
	size_t n = strlen(suffix);
	return s.size() >= n && strcasecmp(s.c_str() + s.size() - n, suffix) == 0;
}

static bool saveImage(const char * fileName, const ImageRGB & img)
{
	std::string name(fileName);
	int ok;
	if (endsWith(name, ".jpg") || endsWith(name, ".jpeg"))
		ok = stbi_write_jpg(fileName, img.width, img.height, 3, img.data.data(), 92);
	else
		ok = stbi_write_png(fileName, img.width, img.height, 3, img.data.data(), img.width * 3);
	if (!ok)
		fprintf(stderr, "Cannot write %s\n", fileName);
	return ok != 0;
}

static void getCpuName(char * name, size_t size)
{
	snprintf(name, size, "unknown CPU");
	FILE * f = fopen("/proc/cpuinfo", "r");
	if (f == NULL)
		return;
	char line[512];
	while (fgets(line, sizeof(line), f))
	{
		if (strncmp(line, "model name", 10) == 0)
		{
			char * p = strchr(line, ':');
			if (p != NULL)
			{
				p++;
				while (*p == ' ') p++;
				p[strcspn(p, "\n")] = 0;
				snprintf(name, size, "%s", p);
			}
			break;
		}
	}
	fclose(f);
}

static bool readHomography(const char * fileName, double H[9])
{
	FILE * f = fopen(fileName, "r");
	if (f == NULL)
		return false;
	for (int i = 0; i < 9; i++)
	{
		if (fscanf(f, "%lf", &H[i]) != 1)
		{
			fclose(f);
			return false;
		}
	}
	fclose(f);
	for (int i = 0; i < 9; i++)
		H[i] /= H[8];
	return true;
}

int main(int argc, char ** argv)
{
	if (argc < 4)
	{
		printf("Usage: %s <left_image> <right_image> <output_image> [--version N] [--runs N]"
				" [--gt H_gt.txt] [--json result.json] [--dump matches.txt]\n", argv[0]);
		return EXIT_FAILURE;
	}
	const char * fileA = argv[1], * fileB = argv[2], * fileOut = argv[3];
	int version = 0, runs = 5;
	const char * fileGt = NULL, * fileJson = NULL, * fileDump = NULL;
	for (int i = 4; i < argc; i++)
	{
		if (strcmp(argv[i], "--version") == 0 && i + 1 < argc) version = atoi(argv[++i]);
		else if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) runs = atoi(argv[++i]);
		else if (strcmp(argv[i], "--gt") == 0 && i + 1 < argc) fileGt = argv[++i];
		else if (strcmp(argv[i], "--json") == 0 && i + 1 < argc) fileJson = argv[++i];
		else if (strcmp(argv[i], "--dump") == 0 && i + 1 < argc) fileDump = argv[++i];
		else
		{
			printf("Unknown argument: %s\n", argv[i]);
			return EXIT_FAILURE;
		}
	}
	if (runs < 1)
		runs = 1;

	StitchFunction stitch = getStitchFunction(version);
	if (stitch == NULL)
	{
		printf("Version %d is not implemented yet\n", version);
		return EXIT_FAILURE;
	}

	// ---- Load the images (not timed)
	ImageRGB imgA, imgB;
	if (!loadImage(fileA, imgA) || !loadImage(fileB, imgB))
		return EXIT_FAILURE;

	char gpuName[256], cpuName[256];
	getGpuName(gpuName, sizeof(gpuName));
	getCpuName(cpuName, sizeof(cpuName));
	printf("Version: V%d\n", version);
	printf("GPU: %s\n", gpuName);
	printf("CPU: %s\n", cpuName);
	printf("Image A: %s (%d x %d)\n", fileA, imgA.width, imgA.height);
	printf("Image B: %s (%d x %d)\n", fileB, imgB.width, imgB.height);

	Params params; // the same parameters for every version

	// ---- Warm-up run (not timed): excludes one-time costs (CUDA context, first launches, caches)
	StitchResult result;
	stitch(imgA, imgB, params, result);

	// ---- Timed runs
	std::vector<StageTimes> times;
	for (int r = 0; r < runs; r++)
	{
		result = StitchResult();
		stitch(imgA, imgB, params, result);
		times.push_back(result.times);
	}

	// ---- Average and minimum of each stage
	double avg[NUM_STAGES], mn[NUM_STAGES];
	double avgTotal = 0, minTotal = DBL_MAX, avgH2D = 0, avgD2H = 0, avgSetup = 0;
	for (int s = 0; s < NUM_STAGES; s++)
	{
		avg[s] = 0;
		mn[s] = DBL_MAX;
		for (int r = 0; r < runs; r++)
		{
			avg[s] += times[r].ms[s] / runs;
			mn[s] = fmin(mn[s], times[r].ms[s]);
		}
	}
	for (int r = 0; r < runs; r++)
	{
		avgTotal += times[r].totalMs / runs;
		minTotal = fmin(minTotal, times[r].totalMs);
		avgH2D += times[r].h2dMs / runs;
		avgD2H += times[r].d2hMs / runs;
		avgSetup += times[r].hostSetupMs / runs;
	}

	printf("\n%-16s %12s %12s %8s\n", "Stage", "avg (ms)", "min (ms)", "% total");
	for (int s = 0; s < NUM_STAGES; s++)
		printf("%-16s %12.3f %12.3f %7.1f%%\n", STAGE_NAMES[s], avg[s], mn[s],
				100.0 * avg[s] / avgTotal);
	if (version > 0)
		printf("%-16s %12.3f\n%-16s %12.3f\n", "H2D transfer", avgH2D, "D2H transfer", avgD2H);
	printf("%-16s %12.3f %12.3f\n", "TOTAL", avgTotal, minTotal);
	printf("(host-side setup included in the stages: %.3f ms = %.3f%% of the total)\n", avgSetup, 100.0 * avgSetup / avgTotal);
	printf("(%d timed runs after 1 warm-up run; image loading/saving excluded)\n\n", runs);

	// ---- Statistics of the last run
	printf("Keypoints: A = %d, B = %d\n", result.numKeypointsA, result.numKeypointsB);
	printf("Matches after ratio test: %d, RANSAC inliers: %d (%.1f%%)\n", result.numMatches,
			result.numInliers, result.numMatches > 0 ? 100.0 * result.numInliers / result.numMatches : 0.0);
	printf("Homography B -> A:\n");
	for (int i = 0; i < 3; i++)
		printf("  [%12.6f %12.6f %12.6f]\n", result.H[3 * i], result.H[3 * i + 1], result.H[3 * i + 2]);
	if (!result.ok)
		printf("STITCHING FAILED: %s\n", result.message.c_str());
	else
		printf("Panorama: %d x %d (A placed at offset %d, %d)\n", result.panorama.width,
				result.panorama.height, result.offsetX, result.offsetY);

	// ---- Homography error against the ground truth (synthetic pair)
	double gtMean = -1, gtMax = -1;
	if (fileGt != NULL)
	{
		double Hgt[9];
		if (!readHomography(fileGt, Hgt))
			printf("Cannot read the ground-truth homography %s\n", fileGt);
		else
		{
			// Corner error: distance between the corners of B mapped by H and by H_gt
			double cx[4] = {0, (double)imgB.width, (double)imgB.width, 0};
			double cy[4] = {0, 0, (double)imgB.height, (double)imgB.height};
			gtMean = 0;
			gtMax = 0;
			for (int k = 0; k < 4; k++)
			{
				double x1, y1, x2, y2;
				applyHomography(result.H, cx[k], cy[k], x1, y1);
				applyHomography(Hgt, cx[k], cy[k], x2, y2);
				double d = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
				gtMean += d / 4;
				gtMax = fmax(gtMax, d);
			}
			printf("Homography error vs ground truth (corners of B): mean = %.3f px, max = %.3f px\n",
					gtMean, gtMax);
		}
	}

	// ---- Save the panorama (not timed)
	if (result.ok)
		saveImage(fileOut, result.panorama);

	// ---- JSON for the notebook
	if (fileJson != NULL)
	{
		FILE * f = fopen(fileJson, "w");
		if (f != NULL)
		{
			fprintf(f, "{\n  \"version\": %d,\n  \"gpu\": \"%s\",\n  \"cpu\": \"%s\",\n", version, gpuName, cpuName);
			fprintf(f, "  \"imageA\": \"%s\",\n  \"imageB\": \"%s\",\n", fileA, fileB);
			fprintf(f, "  \"width\": %d,\n  \"height\": %d,\n  \"runs\": %d,\n", imgA.width, imgA.height, runs);
			fprintf(f, "  \"ok\": %s,\n  \"message\": \"%s\",\n", result.ok ? "true" : "false", result.message.c_str());
			fprintf(f, "  \"stages_avg_ms\": {");
			for (int s = 0; s < NUM_STAGES; s++)
				fprintf(f, "%s\"%s\": %.4f", s ? ", " : "", STAGE_NAMES[s], avg[s]);
			fprintf(f, "},\n  \"stages_min_ms\": {");
			for (int s = 0; s < NUM_STAGES; s++)
				fprintf(f, "%s\"%s\": %.4f", s ? ", " : "", STAGE_NAMES[s], mn[s]);
			fprintf(f, "},\n  \"h2d_avg_ms\": %.4f,\n  \"d2h_avg_ms\": %.4f,\n  \"host_setup_avg_ms\": %.4f,\n", avgH2D, avgD2H, avgSetup);
			fprintf(f, "  \"total_avg_ms\": %.4f,\n  \"total_min_ms\": %.4f,\n", avgTotal, minTotal);
			fprintf(f, "  \"total_runs_ms\": [");
			for (int r = 0; r < runs; r++)
				fprintf(f, "%s%.4f", r ? ", " : "", times[r].totalMs);
			fprintf(f, "],\n  \"keypointsA\": %d,\n  \"keypointsB\": %d,\n", result.numKeypointsA, result.numKeypointsB);
			fprintf(f, "  \"matches\": %d,\n  \"inliers\": %d,\n", result.numMatches, result.numInliers);
			fprintf(f, "  \"H\": [");
			for (int i = 0; i < 9; i++)
				fprintf(f, "%s%.10g", i ? ", " : "", result.H[i]);
			fprintf(f, "],\n  \"panorama_width\": %d,\n  \"panorama_height\": %d,\n",
					result.panorama.width, result.panorama.height);
			fprintf(f, "  \"gt_corner_error_mean_px\": %.4f,\n  \"gt_corner_error_max_px\": %.4f\n}\n", gtMean, gtMax);
			fclose(f);
		}
	}

	// ---- Keypoints and matches for the figures
	if (fileDump != NULL)
	{
		FILE * f = fopen(fileDump, "w");
		if (f != NULL)
		{
			fprintf(f, "# keypointsA %zu\n", result.kpA.size());
			for (const Keypoint & k : result.kpA)
				fprintf(f, "%.2f %.2f %.3f\n", k.x, k.y, k.angle);
			fprintf(f, "# keypointsB %zu\n", result.kpB.size());
			for (const Keypoint & k : result.kpB)
				fprintf(f, "%.2f %.2f %.3f\n", k.x, k.y, k.angle);
			fprintf(f, "# matches %zu (idxA idxB dist inlier)\n", result.matches.size());
			for (size_t i = 0; i < result.matches.size(); i++)
				fprintf(f, "%d %d %d %d\n", result.matches[i].idxA, result.matches[i].idxB,
						result.matches[i].dist, i < result.inlierMask.size() ? (int)result.inlierMask[i] : 0);
			fclose(f);
		}
	}
	return result.ok ? EXIT_SUCCESS : 2;
}
