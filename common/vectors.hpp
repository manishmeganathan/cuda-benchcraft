// common/vectors.hpp
// Vector Utilities

#pragma once
#include <random>
#include <vector>

// Fills a vector with deterministic Uniform(-1.0, 1.0) floats (mt19937 + seed).
// Note: same seed ⇒ same values; use different seeds for A and B to avoid structure.
void fill_vector_uniform(std::vector<float>& v, unsigned seed=1234) {
    // Create random distribution between -1.0 and 1.0 with given seed
    std::mt19937 gen(seed); 
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Create random value for the vector from the distribution
    for (float& x : v) x = dist(gen); 
}

// Compares two vectors and computes the maximum absolute error and the relative error.
// Error tolerance can be adjusted with the eps value.
void cmp_vectors(const std::vector<float>& a, const std::vector<float>& b, double& max_abs_err, double& rel_err, double eps=1e-7) {
    double max_ref_magnitude = 0.0;
    max_abs_err = 0.0; rel_err = 0.0;

    // Determine max absolute error and max reference magnitude
    // by comparing each element in both vectors
    for (size_t i = 0; i < a.size(); i++) {
        double diff = std::abs((double)a[i] - (double)b[i]);
        max_abs_err = std::max(max_abs_err, diff);

        double elem = std::abs((double)b[i]);    
        max_ref_magnitude = std::max(max_ref_magnitude, elem);
    }

    // Compute relative error from max absolute error and max reference magnitude
    // The 1e-7 correction is to avoid the numerical instability of divide-by-zero
    rel_err = max_abs_err / (max_ref_magnitude + eps);  
}