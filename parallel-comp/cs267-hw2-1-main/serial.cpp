#include "common.h"

#include <cmath>
#include <iostream>
#include <string.h>
#include <vector>

#include <immintrin.h>



// Apply the force from neighbor to particle
__attribute__((always_inline))
static inline void apply_force(particle_t& particle, particle_t& 
neighbor) {
	// Calculate Distance
	double dx = neighbor.x - particle.x;
	double dy = neighbor.y - particle.y;
	double r2 = (dx * dx) + (dy * dy);

	// Check if the two particles should interact
	if (r2 > cutoff * cutoff)
		return;

	r2 = fmax(r2, min_r * min_r);
	double r = sqrt(r2);

	// Very simple short-range repulsive force
	double coef = (1 - cutoff / r) / r2 / mass;
	particle.ax += coef * dx;
	particle.ay += coef * dy;

	// This should be the symmetric, opposite force
	neighbor.ax -= coef * dx;
	neighbor.ay -= coef * dy;
}

// Integrate the ODE
__attribute__((always_inline))
static inline void move(particle_t& p, double size) {

	// Slightly simplified Velocity Verlet integration
	// Conserves energy better than explicit Euler method
	p.vx += p.ax * dt;
	p.vy += p.ay * dt;
	p.x += p.vx * dt;
	p.y += p.vy * dt;

	// Bounce from walls
	while (p.x < 0 || p.x > size) {
		p.x = (p.x < 0) ? (-p.x) : ((2 * size) - p.x);
		p.vx = -p.vx;
	}
	while (p.y < 0 || p.y > size) {
		p.y = (p.y < 0) ? (-p.y) : ((2 * size) - p.y);
		p.vy = -p.vy;
	}

	p.ax = 0;
	p.ay = 0;

}

int num_blocks;
int row_length;
std::vector<std::vector<int>> blocks;
std::vector<std::vector<int>> block_neighbors;


static void inline assign_neighbor_blocks(int block_index) {
	int row_index = (block_index / row_length);
	int column_index = (block_index % row_length);
	if (row_index >= 1) {
		if (column_index <= (row_length - 2)) {
			int top_right_index = block_index - row_length + 1;
			block_neighbors[block_index].emplace_back(top_right_index);
		}
	}
	if (column_index <= (row_length - 2)) {
		int mid_right_index = block_index + 1;
		block_neighbors[block_index].emplace_back(mid_right_index);
	}
	if (row_index <= (row_length - 2)) {
		int bottom_center_index = block_index + row_length;
		block_neighbors[block_index].emplace_back(bottom_center_index);
		if (column_index <= (row_length - 2)) {
			int bottom_right_index = block_index + row_length + 1;
			block_neighbors[block_index].emplace_back(bottom_right_index);
		}
	}
}

__attribute__((optimize("unroll-loops")))
void init_simulation(particle_t* parts, int num_parts, double size) {
	row_length = std::ceil(size / cutoff);
	num_blocks = pow(row_length, 2);
	blocks.resize(num_blocks);
	block_neighbors.resize(num_blocks);
	for (int block_index = 0; block_index < num_blocks; block_index += 1) {
		assign_neighbor_blocks(block_index);
	}
	for (int i = 0; i < num_parts; i += 1) {
		int block_index = (floor(parts[i].x / cutoff) * row_length) + floor(parts[i].y / cutoff);
		blocks[block_index].push_back(i);
	}
	for (int i = 0; i < num_parts; i++) {
	    parts[i].ax = 0;     
        parts[i].ay = 0;  
	}
}



__attribute__((optimize("unroll-loops")))
void simulate_one_step(particle_t* parts, int num_parts, double size) {

	for (int block_index = 0; block_index < num_blocks; block_index += 1) {
		for (int i = 0; i < blocks[block_index].size(); i += 1) {
			for (int j = 0; j < i; j += 1) {
				apply_force(parts[blocks[block_index][i]], parts[blocks[block_index][j]]);
			}
			for (int neighbor_index : block_neighbors[block_index]) {
				for (int j = 0; j < blocks[neighbor_index].size(); j += 1) {
					apply_force(parts[blocks[block_index][i]], parts[blocks[neighbor_index][j]]);
				}
			}
		}
	}

	for(int block_index = 0; block_index < num_blocks; block_index += 1) {
		blocks[block_index].clear();
	}

	for (int i = 0; i < num_parts; i += 1) {
		move(parts[i], size);
		int block_index = (floor(parts[i].x / cutoff) * row_length) + floor(parts[i].y / cutoff);
		blocks[block_index].push_back(i);
	}

}
