#include "common.h"
#include <cuda.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>



int num_blocks;
#define threads_per_block 256

double bin_width;
int row_length;
int num_bins;

int* part_ids;
int* part_bin_indices;
int* bin_counts;
int* bin_start_indices;



__global__ void set_ax_ay_gpu(particle_t* parts, int num_parts, int* bin_start_indices, int num_bins) {
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	if (tid >= num_parts) {
		return;
	}
	bin_start_indices[num_bins] = num_parts;
	parts[tid].ax = 0;
	parts[tid].ay = 0;
}

void init_simulation(particle_t* parts, int num_parts, double size) {
	num_blocks = ((num_parts + (threads_per_block - 1)) / threads_per_block);
	bin_width = (2 * cutoff);
	row_length = std::ceil(size / bin_width);
	num_bins = pow(row_length, 2);
	cudaMalloc(((void**) &part_ids), (num_parts * sizeof(int)));
	cudaMalloc(((void**) &part_bin_indices), (num_parts * sizeof(int)));
	cudaMalloc(((void**) &bin_counts), (num_bins * sizeof(int)));
	cudaMalloc(((void**) &bin_start_indices), ((num_bins + 1) * sizeof(int)));
	set_ax_ay_gpu<<<num_blocks, threads_per_block>>>(parts, num_parts, bin_start_indices, num_bins);
}



__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
	double dx = neighbor.x - particle.x;
	double dy = neighbor.y - particle.y;
	double r2 = (dx * dx) + (dy * dy);
	if (r2 > cutoff * cutoff) {
		return;
	}
	// r2 = fmax( r2, min_r*min_r );
	r2 = ((r2 > min_r * min_r) ? (r2) : (min_r * min_r));
	double r = sqrt(r2);
	double coef = (1 - cutoff / r) / r2 / mass;
	// To consider: How does cutting down computations compare to adding these barriers?
	atomicAdd(&(particle.ax), (coef * dx));
	atomicAdd(&(particle.ay), (coef * dy));
}

__device__ void apply_force_symmetric_gpu(particle_t& particle, particle_t& neighbor) {
	double dx = neighbor.x - particle.x;
	double dy = neighbor.y - particle.y;
	double r2 = (dx * dx) + (dy * dy);
	if (r2 > cutoff * cutoff) {
		return;
	}
	// r2 = fmax( r2, min_r*min_r );
	r2 = ((r2 > min_r * min_r) ? (r2) : (min_r * min_r));
	double r = sqrt(r2);
	double coef = (1 - cutoff / r) / r2 / mass;
	atomicAdd(&(particle.ax), (coef * dx));
	atomicAdd(&(particle.ay), (coef * dy));
	atomicAdd(&(neighbor.ax), -(coef * dx));
	atomicAdd(&(neighbor.ay), -(coef * dy));
}


__global__ void compute_forces_gpu(particle_t* parts, int num_parts, double bin_width, int row_length, int* bin_counts, int* bin_start_indices, int* part_ids, int* part_bin_indices) {

	int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	if (tid >= num_parts) {
		return;
	}

	int bin_index = part_bin_indices[tid];
	int row_index = bin_index / row_length;
	int column_index = bin_index % row_length;

	int bin_start_index = bin_start_indices[bin_index];
	int next_bin_start_index = bin_start_indices[bin_index + 1];
	for (int j = bin_start_index; ((j < next_bin_start_index) && (part_ids[j] != tid)); j += 1) {
		apply_force_symmetric_gpu(
			parts[tid],
			parts[part_ids[j]]
		);
	}

	if (column_index <= (row_length - 2)) {
		int mid_right_index = bin_index + 1;
		int bin_start_index = bin_start_indices[mid_right_index];
		int next_bin_start_index = bin_start_indices[mid_right_index + 1];
		for (int j = bin_start_index; j < next_bin_start_index; j += 1) {
			apply_force_symmetric_gpu(
				parts[tid],
				parts[part_ids[j]]
			);
		}
		if (row_index >= 1) {
			int top_right_index = bin_index - row_length + 1;
			int bin_start_index = bin_start_indices[top_right_index];
			int next_bin_start_index = bin_start_indices[top_right_index + 1];
			for (int j = bin_start_index; j < next_bin_start_index; j += 1) {
				apply_force_symmetric_gpu(
					parts[tid],
					parts[part_ids[j]]
				);
			}
		}
	}

	if (row_index <= (row_length - 2)) {
		int bottom_center_index = bin_index + row_length;
		int bin_start_index = bin_start_indices[bottom_center_index];
		int next_bin_start_index = bin_start_indices[bottom_center_index + 1];
		for (int j = bin_start_index; j < next_bin_start_index; j += 1) {
			apply_force_symmetric_gpu(
				parts[tid],
				parts[part_ids[j]]
			);
		}
		if (column_index <= (row_length - 2)) {
			int bottom_right_index = bin_index + row_length + 1;
			int bin_start_index = bin_start_indices[bottom_right_index];
			int next_bin_start_index = bin_start_indices[bottom_right_index + 1];
			for (int j = bin_start_index; j < next_bin_start_index; j += 1) {
				apply_force_symmetric_gpu(
					parts[tid],
					parts[part_ids[j]]
				);
			}
		}
	}

}


__global__ void move_gpu(particle_t* parts, int num_parts, double size) {
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	if (tid >= num_parts) {
		return;
	}
	particle_t* p = &(parts[tid]);
	p->vx += (p->ax * dt);
	p->vy += (p->ay * dt);
	p->x += (p->vx * dt);
	p->y += (p->vy * dt);
	// Bounce from walls
	while (p->x < 0 || p->x > size) {
		p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
		p->vx = -(p->vx);
	}
	while (p->y < 0 || p->y > size) {
		p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
		p->vy = -(p->vy);
	}
	parts[tid].ax = 0;
	parts[tid].ay = 0;
}


__global__ void count_particles_gpu(particle_t* parts, int num_parts, double bin_width, int row_length, int* bin_counts, int* part_bin_indices) {
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	if (tid >= num_parts) {
		return;
	}
	int bin_index = (
		 (floor(parts[tid].x / bin_width) * row_length)
		+ floor(parts[tid].y / bin_width)
	);
	atomicAdd(&(bin_counts[bin_index]), 1);
	part_bin_indices[tid] = bin_index;
}

__global__ void arrange_part_ids_gpu(particle_t* parts, int num_parts, double bin_width, int row_length, int* bin_counts, int* bin_start_indices, int* part_ids, int* part_bin_indices) {
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	if (tid >= num_parts) {
		return;
	}
	int bin_index = part_bin_indices[tid];
	int next_subindex = atomicAdd(&(bin_counts[bin_index]), -1);
	part_ids[(bin_start_indices[bin_index] + next_subindex - 1)] = tid;
}



void simulate_one_step(particle_t* parts, int num_parts, double size) {

	// Compute forces
	compute_forces_gpu<<<num_blocks, threads_per_block>>>(
		parts,
		num_parts,
		bin_width,
		row_length,
		bin_counts,
		bin_start_indices,
		part_ids,
		part_bin_indices
	);

	// Move particles
	move_gpu<<<num_blocks, threads_per_block>>>(parts, num_parts, size);

	// Re-bin particles
	cudaMemset(bin_counts, 0, (num_bins * sizeof(int)));
	count_particles_gpu<<<num_blocks, threads_per_block>>>(
		parts,
		num_parts,
		bin_width,
		row_length,
		bin_counts,
		part_bin_indices
	);
	thrust::exclusive_scan(
		thrust::device,
		bin_counts,
		(bin_counts + num_bins),
		bin_start_indices
	);
	arrange_part_ids_gpu<<<num_blocks, threads_per_block>>>(
		parts,
		num_parts,
		bin_width,
		row_length,
		bin_counts,
		bin_start_indices,
		part_ids,
		part_bin_indices
	);

}
