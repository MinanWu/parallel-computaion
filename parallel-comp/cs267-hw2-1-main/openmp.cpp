#include "common.h"
#include "omp.h"

#include <cmath>
#include <iostream>
#include <string.h>
#include <vector>
#include <map>

#include <immintrin.h>



double block_width;
int num_blocks;
int row_length;

std::vector<std::vector<int>> blocks;
std::vector<std::vector<int>> block_neighbors;
std::vector<std::vector<int>> moved_particles;

std::vector<omp_lock_t> lock_table_parts;
std::vector<omp_lock_t> lock_table_moved_particles;



// Apply the force from neighbor to particle
__attribute__((always_inline))
static inline void apply_force(particle_t& particle, particle_t& 
neighbor, int particle_index, int neighbor_index) {

	double dx = neighbor.x - particle.x;
	double dy = neighbor.y - particle.y;
	double r2 = (dx * dx) + (dy * dy);

	if (r2 > cutoff * cutoff)
		return;

	r2 = fmax(r2, min_r * min_r);
	double r = sqrt(r2);
	double coef = (1 - cutoff / r) / r2 / mass;

	omp_set_lock(&lock_table_parts[particle_index]);
	particle.ax += coef * dx;
	particle.ay += coef * dy;
	omp_unset_lock(&lock_table_parts[particle_index]);

	// This should be the symmetric, opposite force
	omp_set_lock(&lock_table_parts[neighbor_index]);	
	neighbor.ax -= coef * dx;
	neighbor.ay -= coef * dy;
	omp_unset_lock(&lock_table_parts[neighbor_index]);	

}

__attribute__((always_inline))
static inline void move(particle_t& particle, double size) {

	particle.vx += particle.ax * dt;
	particle.vy += particle.ay * dt;
	particle.x += particle.vx * dt;
	particle.y += particle.vy * dt;

	// Bounce from walls
    while (particle.x < 0 || particle.x > size) {
		particle.x = (particle.x < 0) ? (-particle.x) : ((2 * size) - particle.x);
		particle.vx = -particle.vx;
	}
	while (particle.y < 0 || particle.y > size) {
		particle.y = (particle.y < 0) ? (-particle.y) : ((2 * size) - particle.y);
		particle.vy = -particle.vy;
	}

	particle.ax = 0;
	particle.ay = 0;
	
}

__attribute__((optimize("unroll-loops")))
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
	block_width = (4 * cutoff);
	row_length = std::ceil(size / block_width);
	num_blocks = pow(row_length, 2);
	blocks.resize(num_blocks);
	moved_particles.resize(num_blocks);
	block_neighbors.resize(num_blocks);
	for (int block_index = 0; block_index < num_blocks; block_index += 1) {
		assign_neighbor_blocks(block_index);
		omp_lock_t lock;
        omp_init_lock(&lock);
		lock_table_moved_particles.push_back(lock);
	}
	for (int i = 0; i < num_parts; i += 1) {
	    parts[i].ax = 0;     
        parts[i].ay = 0;  
		omp_lock_t lock;
		omp_init_lock(&lock);
		lock_table_parts.push_back(lock);
	}
	for (int i = 0; i < num_parts; i += 1) {
		int block_index = (floor(parts[i].x / block_width) * row_length) + floor(parts[i].y / block_width);
		blocks[block_index].push_back(i);
	}
}

__attribute__((optimize("unroll-loops")))
void simulate_one_step(particle_t* parts, int num_parts, double size) {

	#pragma omp for
	for (int block_index = 0; block_index < num_blocks; block_index += 1) {
		for (int i = 0; i < blocks[block_index].size(); i += 1) {
			for (int j = 0; j < i; j += 1) {
                apply_force(
                    parts[blocks[block_index][i]], parts[blocks[block_index][j]],
                    blocks[block_index][i], blocks[block_index][j]
                );
            }
            for (int neighbor_index : block_neighbors[block_index]) {
                for (int j = 0; j < blocks[neighbor_index].size(); j += 1) {
                    apply_force(
                        parts[blocks[block_index][i]], parts[blocks[neighbor_index][j]],
                        blocks[block_index][i], blocks[neighbor_index][j]
                    );
                }
            }
		}
	}

	# pragma omp for
	for (int block_index = 0; block_index < num_blocks; block_index += 1) {
		for (int i = blocks[block_index].size() - 1; i >= 0; i -= 1) {
			move(parts[blocks[block_index][i]], size);
			int new_block_index = 
            (floor(parts[blocks[block_index][i]].x / block_width) * row_length) + floor(parts[blocks[block_index][i]].y / block_width);
			if (new_block_index != block_index) {
				omp_set_lock(&lock_table_moved_particles[new_block_index]);
				moved_particles[new_block_index].push_back(blocks[block_index][i]);
				omp_unset_lock(&lock_table_moved_particles[new_block_index]);
				blocks[block_index].erase(blocks[block_index].begin() + i);
			}
		}
	}

	# pragma omp for
	for (int block_index = 0; block_index < num_blocks; block_index += 1) {
		while (moved_particles[block_index].size()) {
			int particle_index = moved_particles[block_index].back();
			moved_particles[block_index].pop_back();
			blocks[block_index].push_back(particle_index);	
		}
	}

}
