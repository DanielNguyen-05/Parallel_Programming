// =============================================================================
// v0_orb.cpp - V0: sequential (single-threaded) panorama stitching pipeline
// CSC14120 Parallel Programming - Final Project - Group 09
// Authors: Le Nhat Khoi (23127004), Nguyen Hoang Gia Bao (23127157),
//          Nguyen Hai Dang (23127165)
// Written by the team (with the help of an AI assistant, see the notebook's AI
// usage declaration). No code copied from external libraries; the algorithms
// follow these references (cited in the notebook):
//   - FAST: E. Rosten, T. Drummond, "Machine learning for high-speed corner
//     detection", ECCV 2006.
//   - BRIEF / ORB: M. Calonder et al., "BRIEF: Binary Robust Independent
//     Elementary Features", ECCV 2010; E. Rublee et al., "ORB: an efficient
//     alternative to SIFT or SURF", ICCV 2011 (intensity-centroid orientation,
//     steered BRIEF). Our sampling pattern is random (Gaussian), not ORB's
//     learned pattern.
//   - Ratio test: D. Lowe, "Distinctive image features from scale-invariant
//     keypoints", IJCV 2004.
//   - RANSAC + DLT with normalization: M. Fischler, R. Bolles (1981);
//     R. Hartley, A. Zisserman, "Multiple View Geometry", 2nd ed., ch. 4.
//
// Rules for V0 (fair baseline): one thread, no OpenMP / std::thread, no
// hand-written SIMD. Each stage is timed with std::chrono::high_resolution_clock.
// =============================================================================
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include "common.h"
#include "versions.h"

typedef std::chrono::high_resolution_clock Clock;

static double elapsedMs(Clock::time_point t0, Clock::time_point t1)
{
	return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static inline int clampInt(int v, int lo, int hi)
{
	return v < lo ? lo : (v > hi ? hi : v);
}

// =============================================================================
// Shared helpers (host side, reused by the CUDA versions)
// =============================================================================

// Fixed random sampling pattern of rotated BRIEF: 256 pairs (x1, y1, x2, y2),
// coordinates drawn from an isotropic Gaussian (sigma = 31 / 5) and clipped to
// [-13, 13], so that a rotated point stays within radius 13 * sqrt(2) < 19 pixels.
static uint32_t lcgNext(uint32_t & state)
{
	state = state * 1664525u + 1013904223u; // Numerical Recipes LCG
	return state;
}

static double lcgUniform(uint32_t & state) // in (0, 1)
{
	return ((lcgNext(state) >> 8) + 0.5) / 16777216.0;
}

void makeBriefPattern(unsigned seed, int8_t pattern[256 * 4])
{
	uint32_t state = seed;
	const double sigma = 31.0 / 5.0;
	for (int i = 0; i < 256 * 4; i++)
	{
		// Box-Muller transform
		double u1 = lcgUniform(state), u2 = lcgUniform(state);
		double g = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
		int v = (int)lround(g * sigma);
		pattern[i] = (int8_t)clampInt(v, -13, 13);
	}
	// Avoid degenerate pairs (two identical points always give the same bit)
	for (int i = 0; i < 256; i++)
		if (pattern[4 * i] == pattern[4 * i + 2] && pattern[4 * i + 1] == pattern[4 * i + 3])
			pattern[4 * i + 2] = (int8_t)(pattern[4 * i + 2] < 13 ? pattern[4 * i + 2] + 1 : 12);
}

// Pre-generated RANSAC samples: iters x 4 distinct indices in [0, n).
// Generating them once on the host lets every version test exactly the same hypotheses.
void makeRansacSamples(unsigned seed, int iters, int n, std::vector<int> & samples)
{
	samples.resize((size_t)iters * 4);
	uint32_t state = seed;
	for (int it = 0; it < iters; it++)
	{
		int * s = &samples[(size_t)it * 4];
		for (int k = 0; k < 4; k++)
		{
			int v;
			bool dup;
			do
			{
				v = (int)((lcgNext(state) >> 8) % (uint32_t)n);
				dup = false;
				for (int j = 0; j < k; j++)
					if (s[j] == v) dup = true;
			} while (dup);
			s[k] = v;
		}
	}
}

void applyHomography(const double H[9], double x, double y, double & outX, double & outY)
{
	double w = H[6] * x + H[7] * y + H[8];
	outX = (H[0] * x + H[1] * y + H[2]) / w;
	outY = (H[3] * x + H[4] * y + H[5]) / w;
}

bool invertHomography(const double H[9], double Hi[9])
{
	double det = H[0] * (H[4] * H[8] - H[5] * H[7])
			- H[1] * (H[3] * H[8] - H[5] * H[6])
			+ H[2] * (H[3] * H[7] - H[4] * H[6]);
	if (fabs(det) < 1e-12)
		return false;
	Hi[0] =  (H[4] * H[8] - H[5] * H[7]) / det;
	Hi[1] = -(H[1] * H[8] - H[2] * H[7]) / det;
	Hi[2] =  (H[1] * H[5] - H[2] * H[4]) / det;
	Hi[3] = -(H[3] * H[8] - H[5] * H[6]) / det;
	Hi[4] =  (H[0] * H[8] - H[2] * H[6]) / det;
	Hi[5] = -(H[0] * H[5] - H[2] * H[3]) / det;
	Hi[6] =  (H[3] * H[7] - H[4] * H[6]) / det;
	Hi[7] = -(H[0] * H[7] - H[1] * H[6]) / det;
	Hi[8] =  (H[0] * H[4] - H[1] * H[3]) / det;
	for (int i = 0; i < 9; i++)
		Hi[i] /= Hi[8];
	return true;
}

// Solve the n x n system A x = b (A row-major, modified in place) with Gaussian
// elimination and partial pivoting. Returns false if the system is (nearly) singular.
static bool solveLinear(double * A, double * b, double * x, int n)
{
	for (int col = 0; col < n; col++)
	{
		int piv = col;
		for (int r = col + 1; r < n; r++)
			if (fabs(A[r * n + col]) > fabs(A[piv * n + col]))
				piv = r;
		if (fabs(A[piv * n + col]) < 1e-12)
			return false;
		if (piv != col)
		{
			for (int c = 0; c < n; c++)
				std::swap(A[col * n + c], A[piv * n + c]);
			std::swap(b[col], b[piv]);
		}
		for (int r = col + 1; r < n; r++)
		{
			double f = A[r * n + col] / A[col * n + col];
			for (int c = col; c < n; c++)
				A[r * n + c] -= f * A[col * n + c];
			b[r] -= f * b[col];
		}
	}
	for (int r = n - 1; r >= 0; r--)
	{
		double s = b[r];
		for (int c = r + 1; c < n; c++)
			s -= A[r * n + c] * x[c];
		x[r] = s / A[r * n + r];
	}
	return true;
}

// Homography (h33 = 1) from exactly 4 correspondences src -> dst (DLT, 8 x 8 system)
static bool homographyFrom4(const float * sx, const float * sy, const float * dx, const float * dy, double H[9])
{
	double A[64], b[8], h[8];
	for (int k = 0; k < 4; k++)
	{
		double x = sx[k], y = sy[k], u = dx[k], v = dy[k];
		double * r1 = &A[(2 * k) * 8], * r2 = &A[(2 * k + 1) * 8];
		r1[0] = x; r1[1] = y; r1[2] = 1; r1[3] = 0; r1[4] = 0; r1[5] = 0; r1[6] = -u * x; r1[7] = -u * y;
		r2[0] = 0; r2[1] = 0; r2[2] = 0; r2[3] = x; r2[4] = y; r2[5] = 1; r2[6] = -v * x; r2[7] = -v * y;
		b[2 * k] = u;
		b[2 * k + 1] = v;
	}
	if (!solveLinear(A, b, h, 8))
		return false;
	for (int i = 0; i < 8; i++)
		H[i] = h[i];
	H[8] = 1;
	return true;
}

// Least-squares homography from n >= 4 correspondences, with Hartley normalization
// (centroid at the origin, mean distance sqrt(2)), solved with the normal equations.
static bool homographyLeastSquares(const std::vector<float> & sx, const std::vector<float> & sy,
		const std::vector<float> & dx, const std::vector<float> & dy, double H[9])
{
	int n = (int)sx.size();
	if (n < 4)
		return false;
	// Normalization transforms
	double msx = 0, msy = 0, mdx = 0, mdy = 0;
	for (int i = 0; i < n; i++) { msx += sx[i]; msy += sy[i]; mdx += dx[i]; mdy += dy[i]; }
	msx /= n; msy /= n; mdx /= n; mdy /= n;
	double ds = 0, dd = 0;
	for (int i = 0; i < n; i++)
	{
		ds += sqrt((sx[i] - msx) * (sx[i] - msx) + (sy[i] - msy) * (sy[i] - msy));
		dd += sqrt((dx[i] - mdx) * (dx[i] - mdx) + (dy[i] - mdy) * (dy[i] - mdy));
	}
	double ss = sqrt(2.0) * n / fmax(ds, 1e-9), sd = sqrt(2.0) * n / fmax(dd, 1e-9);

	// Normal equations (A^T A) h = A^T b for the 2n x 8 system
	double ATA[64] = {0}, ATb[8] = {0}, h[8];
	for (int i = 0; i < n; i++)
	{
		double x = (sx[i] - msx) * ss, y = (sy[i] - msy) * ss;
		double u = (dx[i] - mdx) * sd, v = (dy[i] - mdy) * sd;
		double r1[8] = {x, y, 1, 0, 0, 0, -u * x, -u * y};
		double r2[8] = {0, 0, 0, x, y, 1, -v * x, -v * y};
		for (int a = 0; a < 8; a++)
		{
			for (int c = 0; c < 8; c++)
				ATA[a * 8 + c] += r1[a] * r1[c] + r2[a] * r2[c];
			ATb[a] += r1[a] * u + r2[a] * v;
		}
	}
	if (!solveLinear(ATA, ATb, h, 8))
		return false;
	double Hn[9] = {h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1};
	// Denormalize: H = Td^-1 * Hn * Ts
	double Ts[9] = {ss, 0, -ss * msx, 0, ss, -ss * msy, 0, 0, 1};
	double TdInv[9] = {1 / sd, 0, mdx, 0, 1 / sd, mdy, 0, 0, 1};
	double tmp[9];
	for (int r = 0; r < 3; r++)
		for (int c = 0; c < 3; c++)
			tmp[r * 3 + c] = Hn[r * 3 + 0] * Ts[0 * 3 + c] + Hn[r * 3 + 1] * Ts[1 * 3 + c] + Hn[r * 3 + 2] * Ts[2 * 3 + c];
	for (int r = 0; r < 3; r++)
		for (int c = 0; c < 3; c++)
			H[r * 3 + c] = TdInv[r * 3 + 0] * tmp[0 * 3 + c] + TdInv[r * 3 + 1] * tmp[1 * 3 + c] + TdInv[r * 3 + 2] * tmp[2 * 3 + c];
	if (fabs(H[8]) < 1e-12)
		return false;
	for (int i = 0; i < 9; i++)
		H[i] /= H[8];
	return true;
}

// =============================================================================
// Stage 1: preprocessing (grayscale + separable Gaussian blur)
// =============================================================================
static void rgbToGray(const ImageRGB & in, ImageGray & out)
{
	out.width = in.width;
	out.height = in.height;
	out.data.resize((size_t)in.width * in.height);
	for (int r = 0; r < in.height; r++)
		for (int c = 0; c < in.width; c++)
		{
			size_t i = (size_t)r * in.width + c;
			const uint8_t * p = &in.data[3 * i];
			out.data[i] = (uint8_t)(0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2]);
		}
}

void makeGaussianKernel(float sigma, int radius, std::vector<float> & k)
{
	k.resize(2 * radius + 1);
	float sum = 0;
	for (int i = -radius; i <= radius; i++)
	{
		k[i + radius] = expf(-(float)(i * i) / (2 * sigma * sigma));
		sum += k[i + radius];
	}
	for (float & v : k)
		v /= sum;
}

// Separable filter: horizontal pass (into a float buffer), then vertical pass.
// Border: nearest real pixel (clamped index). Work per pixel: 2(2r+1) instead of (2r+1)^2.
static void gaussianBlur(const ImageGray & in, ImageGray & out, const std::vector<float> & k)
{
	int w = in.width, h = in.height, radius = (int)k.size() / 2;
	std::vector<float> tmp((size_t)w * h);
	for (int r = 0; r < h; r++)
		for (int c = 0; c < w; c++)
		{
			float s = 0;
			for (int i = -radius; i <= radius; i++)
				s += k[i + radius] * in.data[(size_t)r * w + clampInt(c + i, 0, w - 1)];
			tmp[(size_t)r * w + c] = s;
		}
	out.width = w;
	out.height = h;
	out.data.resize((size_t)w * h);
	for (int r = 0; r < h; r++)
		for (int c = 0; c < w; c++)
		{
			float s = 0;
			for (int i = -radius; i <= radius; i++)
				s += k[i + radius] * tmp[(size_t)clampInt(r + i, 0, h - 1) * w + c];
			out.data[(size_t)r * w + c] = (uint8_t)fminf(s + 0.5f, 255.0f);
		}
}

// =============================================================================
// Stage 2: keypoint detection (FAST-9 + 3x3 non-maximum suppression + top-K)
// =============================================================================
// The 16 pixels of the Bresenham circle of radius 3 around the candidate
static const int FAST_DX[16] = {0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1};
static const int FAST_DY[16] = {-3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3};

// True if the 16-bit circular mask contains at least 9 consecutive ones
static inline bool hasArc9(uint32_t mask16)
{
	uint32_t m = mask16 | (mask16 << 16); // unroll the circle
	uint32_t run = m;
	for (int i = 1; i < 9; i++)
		run &= m >> i;
	return run != 0;
}

// FAST score of pixel (x, y) (0 if it is not a corner): the larger of the sums
// of (|I_k - I_p| - t) over the brighter pixels and over the darker pixels.
static float fastScore(const uint8_t * img, int w, int x, int y, int t)
{
	int p = img[y * w + x];
	// Quick rejection: an arc of 9 contains at least 2 of the 4 compass pixels
	int nb = 0, nd = 0;
	for (int k = 0; k < 16; k += 4)
	{
		int v = img[(y + FAST_DY[k]) * w + x + FAST_DX[k]];
		nb += v > p + t;
		nd += v < p - t;
	}
	if (nb < 2 && nd < 2)
		return 0;
	uint32_t brighter = 0, darker = 0;
	int sumB = 0, sumD = 0;
	for (int k = 0; k < 16; k++)
	{
		int v = img[(y + FAST_DY[k]) * w + x + FAST_DX[k]];
		if (v > p + t) { brighter |= 1u << k; sumB += v - p - t; }
		else if (v < p - t) { darker |= 1u << k; sumD += p - v - t; }
	}
	bool corner = hasArc9(brighter) || hasArc9(darker);
	return corner ? (float)std::max(sumB, sumD) : 0.0f;
}

static void detectKeypoints(const ImageGray & gray, const Params & prm, std::vector<Keypoint> & kps)
{
	int w = gray.width, h = gray.height, b = prm.border;
	kps.clear();
	if (w <= 2 * b + 2 || h <= 2 * b + 2)
		return;
	// 1) FAST score of every pixel far enough from the border
	std::vector<float> score((size_t)w * h, 0.0f);
	for (int y = b; y < h - b; y++)
		for (int x = b; x < w - b; x++)
			score[(size_t)y * w + x] = fastScore(gray.data.data(), w, x, y, prm.fastThreshold);

	// 2) 3x3 non-maximum suppression (ties: the first pixel in raster order wins)
	std::vector<Keypoint> cand;
	for (int y = b; y < h - b; y++)
		for (int x = b; x < w - b; x++)
		{
			float s = score[(size_t)y * w + x];
			if (s <= 0)
				continue;
			bool isMax = true;
			for (int dy = -1; dy <= 1 && isMax; dy++)
				for (int dx = -1; dx <= 1; dx++)
				{
					if (dx == 0 && dy == 0)
						continue;
					float n = score[(size_t)(y + dy) * w + x + dx];
					bool before = dy < 0 || (dy == 0 && dx < 0);
					if (before ? n >= s : n > s) { isMax = false; break; }
				}
			if (isMax)
				cand.push_back({(float)x, (float)y, s, 0.0f});
		}

	// 3) Top-K by score (deterministic order: score desc, then y, then x)
	auto stronger = [](const Keypoint & a, const Keypoint & c) {
		if (a.score != c.score) return a.score > c.score;
		if (a.y != c.y) return a.y < c.y;
		return a.x < c.x;
	};
	int K = std::min((int)cand.size(), prm.maxKeypoints);
	std::partial_sort(cand.begin(), cand.begin() + K, cand.end(), stronger);
	kps.assign(cand.begin(), cand.begin() + K);
}

// =============================================================================
// Stage 3: description (intensity-centroid orientation + rotated BRIEF, 256 bits)
// =============================================================================
static void describeKeypoints(const ImageGray & blur, const Params & prm, const int8_t * pattern,
		std::vector<Keypoint> & kps, std::vector<Descriptor> & desc)
{
	int w = blur.width, R = prm.orientRadius;
	const uint8_t * img = blur.data.data();
	// Half-width of the circular patch for each row offset dy
	std::vector<int> umax(R + 1);
	for (int dy = 0; dy <= R; dy++)
		umax[dy] = (int)floor(sqrt((double)R * R - dy * dy));

	desc.resize(kps.size());
	for (size_t i = 0; i < kps.size(); i++)
	{
		int x = (int)kps[i].x, y = (int)kps[i].y;
		// Orientation: angle of the intensity centroid (m10, m01) of the circular patch
		long m10 = 0, m01 = 0;
		for (int dy = -R; dy <= R; dy++)
		{
			int u = umax[abs(dy)];
			const uint8_t * row = img + (size_t)(y + dy) * w + x;
			for (int dx = -u; dx <= u; dx++)
			{
				int v = row[dx];
				m10 += dx * v;
				m01 += dy * v;
			}
		}
		float angle = atan2f((float)m01, (float)m10);
		kps[i].angle = angle;
		float ca = cosf(angle), sa = sinf(angle);

		// Steered BRIEF: rotate each sampling pair by the orientation, compare intensities
		Descriptor d;
		memset(&d, 0, sizeof(d));
		for (int bit = 0; bit < 256; bit++)
		{
			const int8_t * p = &pattern[4 * bit];
			int x1 = x + (int)lroundf(ca * p[0] - sa * p[1]);
			int y1 = y + (int)lroundf(sa * p[0] + ca * p[1]);
			int x2 = x + (int)lroundf(ca * p[2] - sa * p[3]);
			int y2 = y + (int)lroundf(sa * p[2] + ca * p[3]);
			if (img[(size_t)y1 * w + x1] < img[(size_t)y2 * w + x2])
				d.w[bit >> 5] |= 1u << (bit & 31);
		}
		desc[i] = d;
	}
}

// =============================================================================
// Stage 4: matching (brute-force Hamming distance + Lowe's ratio test)
// =============================================================================
static inline int hamming(const Descriptor & a, const Descriptor & b)
{
	int d = 0;
	for (int k = 0; k < 8; k++)
		d += __builtin_popcount(a.w[k] ^ b.w[k]);
	return d;
}

static void matchDescriptors(const std::vector<Descriptor> & dA, const std::vector<Descriptor> & dB,
		const Params & prm, std::vector<Match> & matches)
{
	matches.clear();
	if (dB.size() < 2)
		return;
	for (size_t i = 0; i < dA.size(); i++)
	{
		int best = 1 << 30, second = 1 << 30, bestIdx = -1;
		for (size_t j = 0; j < dB.size(); j++)
		{
			int d = hamming(dA[i], dB[j]);
			if (d < best) { second = best; best = d; bestIdx = (int)j; }
			else if (d < second) second = d;
		}
		if (best <= prm.maxHamming && (float)best < prm.ratio * (float)second)
			matches.push_back({(int)i, bestIdx, best});
	}
}

// =============================================================================
// Stage 5: homography B -> A (RANSAC with 4-point DLT, then least squares on inliers)
// =============================================================================
static int countInliers(const double H[9], const std::vector<float> & sx, const std::vector<float> & sy,
		const std::vector<float> & dx, const std::vector<float> & dy, float thr, uint8_t * mask)
{
	double thr2 = (double)thr * thr;
	int count = 0;
	for (size_t i = 0; i < sx.size(); i++)
	{
		double w = H[6] * sx[i] + H[7] * sy[i] + H[8];
		bool in = false;
		if (fabs(w) > 1e-12)
		{
			double u = (H[0] * sx[i] + H[1] * sy[i] + H[2]) / w - dx[i];
			double v = (H[3] * sx[i] + H[4] * sy[i] + H[5]) / w - dy[i];
			in = u * u + v * v < thr2;
		}
		if (mask) mask[i] = in;
		count += in;
	}
	return count;
}

static bool estimateHomography(const std::vector<Keypoint> & kA, const std::vector<Keypoint> & kB,
		const std::vector<Match> & matches, const Params & prm, float thr,
		double H[9], std::vector<uint8_t> & inlierMask, int & numInliers, double & setupMs)
{
	int n = (int)matches.size();
	numInliers = 0;
	inlierMask.assign(n, 0);
	for (int i = 0; i < 9; i++) H[i] = (i % 4 == 0); // identity
	if (n < 4)
		return false;
	// Correspondences: source = keypoint in B, destination = keypoint in A
	std::vector<float> sx(n), sy(n), dx(n), dy(n);
	for (int i = 0; i < n; i++)
	{
		sx[i] = kB[matches[i].idxB].x; sy[i] = kB[matches[i].idxB].y;
		dx[i] = kA[matches[i].idxA].x; dy[i] = kA[matches[i].idxA].y;
	}
	std::vector<int> samples;
	Clock::time_point s0 = Clock::now();
	makeRansacSamples(prm.ransacSeed, prm.ransacIters, n, samples);
	setupMs += elapsedMs(s0, Clock::now());

	// 1) RANSAC: every hypothesis comes from 4 pre-generated random matches
	int bestCount = -1;
	double bestH[9];
	for (int it = 0; it < prm.ransacIters; it++)
	{
		const int * s = &samples[(size_t)it * 4];
		float qsx[4], qsy[4], qdx[4], qdy[4];
		for (int k = 0; k < 4; k++)
		{
			qsx[k] = sx[s[k]]; qsy[k] = sy[s[k]];
			qdx[k] = dx[s[k]]; qdy[k] = dy[s[k]];
		}
		double Hh[9];
		if (!homographyFrom4(qsx, qsy, qdx, qdy, Hh))
			continue; // degenerate sample (e.g. collinear points)
		int c = countInliers(Hh, sx, sy, dx, dy, thr, NULL);
		if (c > bestCount) // ties: the first hypothesis wins
		{
			bestCount = c;
			memcpy(bestH, Hh, sizeof(bestH));
		}
	}
	if (bestCount < 4)
		return false;

	// 2) Refinement: a homography from only 4 points is noisy, so it is re-estimated by
	//    least squares on all its inliers; the inliers are then recomputed with the new
	//    homography and the fit is repeated (a few iterations are enough to converge).
	memcpy(H, bestH, sizeof(bestH));
	numInliers = countInliers(H, sx, sy, dx, dy, thr, inlierMask.data());
	for (int iter = 0; iter < prm.refineIters; iter++)
	{
		std::vector<float> isx, isy, idx, idy;
		for (int i = 0; i < n; i++)
			if (inlierMask[i])
			{
				isx.push_back(sx[i]); isy.push_back(sy[i]);
				idx.push_back(dx[i]); idy.push_back(dy[i]);
			}
		double refined[9];
		if (!homographyLeastSquares(isx, isy, idx, idy, refined))
			break;
		std::vector<uint8_t> mask2(n);
		int c2 = countInliers(refined, sx, sy, dx, dy, thr, mask2.data());
		if (c2 < numInliers / 2) // safety: never accept a fit that loses most inliers
			break;
		memcpy(H, refined, sizeof(refined));
		inlierMask = mask2;
		numInliers = c2;
	}
	return numInliers >= prm.minInliers;
}

// =============================================================================
// Stage 6: warping and blending (inverse warping, bilinear sampling, feathering)
// =============================================================================
static inline float sampleBilinear(const ImageRGB & img, float x, float y, int ch)
{
	int x0 = (int)floorf(x), y0 = (int)floorf(y);
	int x1 = std::min(x0 + 1, img.width - 1), y1 = std::min(y0 + 1, img.height - 1);
	float fx = x - x0, fy = y - y0;
	const uint8_t * d = img.data.data();
	int w = img.width;
	float p00 = d[((size_t)y0 * w + x0) * 3 + ch], p01 = d[((size_t)y0 * w + x1) * 3 + ch];
	float p10 = d[((size_t)y1 * w + x0) * 3 + ch], p11 = d[((size_t)y1 * w + x1) * 3 + ch];
	return (1 - fy) * ((1 - fx) * p00 + fx * p01) + fy * ((1 - fx) * p10 + fx * p11);
}

static bool warpAndBlend(const ImageRGB & A, const ImageRGB & B, const double H[9], const Params & prm,
		ImageRGB & pano, int & offX, int & offY, std::string & msg)
{
	// Canvas = bounding box of A and of the 4 corners of B mapped into A's frame
	double minX = 0, minY = 0, maxX = A.width, maxY = A.height;
	double cx[4] = {0, (double)B.width, (double)B.width, 0}, cy[4] = {0, 0, (double)B.height, (double)B.height};
	for (int k = 0; k < 4; k++)
	{
		double w = H[6] * cx[k] + H[7] * cy[k] + H[8];
		if (w <= 1e-6)
		{
			msg = "invalid homography (a corner of B maps behind the camera)";
			return false;
		}
		double x, y;
		applyHomography(H, cx[k], cy[k], x, y);
		minX = fmin(minX, x); minY = fmin(minY, y);
		maxX = fmax(maxX, x); maxY = fmax(maxY, y);
	}
	int W = (int)ceil(maxX - floor(minX)), Hc = (int)ceil(maxY - floor(minY));
	if (W > prm.maxCanvasFactor * A.width || Hc > prm.maxCanvasFactor * A.height)
	{
		msg = "invalid homography (panorama would be too large)";
		return false;
	}
	offX = (int)-floor(minX);
	offY = (int)-floor(minY);
	double Hinv[9];
	if (!invertHomography(H, Hinv))
	{
		msg = "homography is not invertible";
		return false;
	}
	float hi[9];
	for (int i = 0; i < 9; i++) hi[i] = (float)Hinv[i];

	pano.width = W;
	pano.height = Hc;
	pano.data.assign((size_t)W * Hc * 3, 0);
	for (int v = 0; v < Hc; v++)
		for (int u = 0; u < W; u++)
		{
			float xa = (float)(u - offX), ya = (float)(v - offY); // position in A's frame
			// Weight of A: distance to A's border (feathering), 0 outside A
			float wa = 0;
			int ia = (int)xa, ja = (int)ya;
			if (xa >= 0 && ya >= 0 && ia < A.width && ja < A.height)
				wa = fminf(fminf(xa + 1, A.width - xa), fminf(ya + 1, A.height - ya));
			// Position in B (inverse mapping) and weight of B
			float wz = hi[6] * xa + hi[7] * ya + hi[8];
			float xb = (hi[0] * xa + hi[1] * ya + hi[2]) / wz;
			float yb = (hi[3] * xa + hi[4] * ya + hi[5]) / wz;
			float wb = 0;
			if (wz > 0 && xb >= 0 && yb >= 0 && xb <= B.width - 1 && yb <= B.height - 1)
				wb = fminf(fminf(xb + 1, B.width - xb), fminf(yb + 1, B.height - yb));
			float sum = wa + wb;
			if (sum <= 0)
				continue; // outside both images: black
			uint8_t * out = &pano.data[((size_t)v * W + u) * 3];
			for (int ch = 0; ch < 3; ch++)
			{
				float ca = wa > 0 ? A.data[((size_t)ja * A.width + ia) * 3 + ch] : 0.0f;
				float cb = wb > 0 ? sampleBilinear(B, xb, yb, ch) : 0.0f;
				out[ch] = (uint8_t)fminf((wa * ca + wb * cb) / sum + 0.5f, 255.0f);
			}
		}
	return true;
}

// =============================================================================
// V0 pipeline
// =============================================================================
void stitchV0(const ImageRGB & imgA, const ImageRGB & imgB, const Params & prm, StitchResult & res)
{
	StageTimes & t = res.times;
	Clock::time_point start = Clock::now(), t0, t1;

	// Stage 1: preprocessing (both images)
	t0 = Clock::now();
	std::vector<float> gk;
	makeGaussianKernel(prm.gaussSigma, prm.gaussRadius, gk);
	t.hostSetupMs += elapsedMs(t0, Clock::now());
	ImageGray grayA, grayB, blurA, blurB;
	rgbToGray(imgA, grayA);
	rgbToGray(imgB, grayB);
	gaussianBlur(grayA, blurA, gk);
	gaussianBlur(grayB, blurB, gk);
	t1 = Clock::now();
	t.ms[STAGE_PREPROCESS] = elapsedMs(t0, t1);

	// Stage 2: keypoint detection on the (unblurred) gray images
	t0 = Clock::now();
	std::vector<Keypoint> kA, kB;
	detectKeypoints(grayA, prm, kA);
	detectKeypoints(grayB, prm, kB);
	t1 = Clock::now();
	t.ms[STAGE_DETECT] = elapsedMs(t0, t1);

	// Stage 3: description on the blurred images (BRIEF needs smoothing against noise)
	t0 = Clock::now();
	int8_t pattern[256 * 4];
	makeBriefPattern(prm.briefSeed, pattern);
	t.hostSetupMs += elapsedMs(t0, Clock::now());
	std::vector<Descriptor> dA, dB;
	describeKeypoints(blurA, prm, pattern, kA, dA);
	describeKeypoints(blurB, prm, pattern, kB, dB);
	t1 = Clock::now();
	t.ms[STAGE_DESCRIBE] = elapsedMs(t0, t1);

	// Stage 4: matching A -> B
	t0 = Clock::now();
	std::vector<Match> matches;
	matchDescriptors(dA, dB, prm, matches);
	t1 = Clock::now();
	t.ms[STAGE_MATCH] = elapsedMs(t0, t1);

	// Stage 5: homography B -> A
	t0 = Clock::now();
	float thr = prm.ransacThreshold * (float)imgA.width / 1280.0f; // threshold scales with the resolution
	std::vector<uint8_t> inlierMask;
	int numInliers = 0;
	bool okH = estimateHomography(kA, kB, matches, prm, thr, res.H, inlierMask, numInliers, t.hostSetupMs);
	t1 = Clock::now();
	t.ms[STAGE_HOMOGRAPHY] = elapsedMs(t0, t1);

	// Stage 6: warping + blending
	t0 = Clock::now();
	bool okW = false;
	if (okH)
		okW = warpAndBlend(imgA, imgB, res.H, prm, res.panorama, res.offsetX, res.offsetY, res.message);
	else
		res.message = "not enough RANSAC inliers";
	t1 = Clock::now();
	t.ms[STAGE_WARP] = elapsedMs(t0, t1);
	t.totalMs = elapsedMs(start, t1);

	// Statistics and data for the report (outside the timed region)
	res.ok = okH && okW;
	res.numKeypointsA = (int)kA.size();
	res.numKeypointsB = (int)kB.size();
	res.numMatches = (int)matches.size();
	res.numInliers = numInliers;
	res.kpA = std::move(kA);
	res.kpB = std::move(kB);
	res.matches = std::move(matches);
	res.inlierMask = std::move(inlierMask);
}
