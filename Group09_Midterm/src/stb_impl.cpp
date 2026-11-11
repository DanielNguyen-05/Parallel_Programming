// =============================================================================
// stb_impl.cpp - Compiles the implementation of the stb image libraries once
// CSC14120 Parallel Programming - Final Project - Group 09
// Authors: Le Nhat Khoi (23127004), Nguyen Hoang Gia Bao (23127157),
//          Nguyen Hai Dang (23127165)
// External code: stb_image.h (v2.30) and stb_image_write.h by Sean Barrett,
// https://github.com/nothings/stb (public domain / MIT). Used only to load and
// save JPG/PNG files; image I/O is excluded from all timings.
// =============================================================================
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
