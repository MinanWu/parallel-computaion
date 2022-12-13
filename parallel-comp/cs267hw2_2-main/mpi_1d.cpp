#include "common.h"

#include <cmath>
#include <iostream>
#include <string.h>
#include <vector>

#include <mpi.h>




double block_width;
int num_blocks;
int row_length;

int panel_height_in_blocks;

std::vector<std::vector<particle_t>> blocks;
std::vector<std::vector<int>> block_neighbors;
std::vector<std::vector<particle_t>> moved_particles;

std::vector<int> my_block_indices;
std::vector<int> my_top_ghost_block_indices;
std::vector<int> my_bottom_ghost_block_indices;

std::vector<int> not_my_block_indices;
std::vector<int> top_neighbor_ghost_block_indices;
std::vector<int> bottom_neighbor_ghost_block_indices;

MPI_Request send_recv_requests[4];

int send_recv_buffer_length;
particle_t* top_send_buffer;
particle_t* bottom_send_buffer;
particle_t* top_recv_buffer;
particle_t* bottom_recv_buffer;



// Apply the force from neighbor to particle
__attribute__((always_inline))
static inline void apply_force(particle_t& particle, particle_t& neighbor, 
bool symmetric) {

	double dx = neighbor.x - particle.x;
	double dy = neighbor.y - particle.y;
	double r2 = (dx * dx) + (dy * dy);

	if (r2 > cutoff * cutoff)
		return;

	r2 = fmax(r2, min_r * min_r);
	double r = sqrt(r2);
	double coef = (1 - cutoff / r) / r2 / mass;

	particle.ax += coef * dx;
	particle.ay += coef * dy;

	if (symmetric) {
		neighbor.ax -= coef * dx;
		neighbor.ay -= coef * dy;
	}

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
		int top_center_index = block_index - row_length;
		block_neighbors[block_index].emplace_back(top_center_index);
		if (column_index >= 1) {
			int top_left_index = block_index - row_length - 1;
			block_neighbors[block_index].emplace_back(top_left_index);
		}
		if (column_index <= (row_length - 2)) {
			int top_right_index = block_index - row_length + 1;
			block_neighbors[block_index].emplace_back(top_right_index);
		}
	}
	if (column_index >= 1) {
		int mid_left_index = block_index - 1;
		block_neighbors[block_index].emplace_back(mid_left_index);
	}
	if (column_index <= (row_length - 2)) {
		int mid_right_index = block_index + 1;
		block_neighbors[block_index].emplace_back(mid_right_index);
	}
	if (row_index <= (row_length - 2)) {
		int bottom_center_index = block_index + row_length;
		block_neighbors[block_index].emplace_back(bottom_center_index);
		if (column_index >= 1) {
			int bottom_left_index = block_index + row_length - 1;
			block_neighbors[block_index].emplace_back(bottom_left_index);
		}
		if (column_index <= (row_length - 2)) {
			int bottom_right_index = block_index + row_length + 1;
			block_neighbors[block_index].emplace_back(bottom_right_index);
		}
	}
}

static void inline assign_my_blocks(int rank) {
	for (int block_index = 0; block_index < num_blocks; block_index += 1) {
		int row_index = (block_index / row_length);
		if ((row_index >= (rank * panel_height_in_blocks)) && (row_index < ((rank + 1) * panel_height_in_blocks))) {
			my_block_indices.push_back(block_index);
			if (row_index == (rank * panel_height_in_blocks)) {
				my_top_ghost_block_indices.push_back(block_index);
			}
			if (row_index == (((rank + 1) * panel_height_in_blocks) - 1)) {
				my_bottom_ghost_block_indices.push_back(block_index);
			}
		}
		else {
			not_my_block_indices.push_back(block_index);
			if (row_index == ((rank * panel_height_in_blocks) - 1)) {
				top_neighbor_ghost_block_indices.push_back(block_index);
			}
			else if (row_index == ((rank + 1) * panel_height_in_blocks)) {
				bottom_neighbor_ghost_block_indices.push_back(block_index);
			}
		}
	}
}

__attribute__((optimize("unroll-loops")))
void init_simulation(particle_t* parts, int num_parts, double size, int rank, 
int num_procs) {

	block_width = (4 * cutoff);
	row_length = std::ceil(size / block_width);
	num_blocks = pow(row_length, 2);

	panel_height_in_blocks = std::ceil(((double) row_length) / num_procs);

	blocks.resize(num_blocks);
	moved_particles.resize(num_blocks);
	block_neighbors.resize(num_blocks);

	for (int block_index = 0; block_index < num_blocks; block_index += 1) {
		assign_neighbor_blocks(block_index);
	}
	
	assign_my_blocks(rank);

	send_recv_buffer_length = num_parts;
	top_send_buffer = ((particle_t*) malloc(send_recv_buffer_length * sizeof(particle_t)));
	bottom_send_buffer = ((particle_t*) malloc(send_recv_buffer_length * sizeof(particle_t)));
	top_recv_buffer = ((particle_t*) malloc(send_recv_buffer_length * sizeof(particle_t)));
	bottom_recv_buffer = ((particle_t*) malloc(send_recv_buffer_length * sizeof(particle_t)));

	for (int i = 0; i < num_parts; i += 1) {
		parts[i].ax = 0;     
		parts[i].ay = 0;
	}

	for (int i = 0; i < num_parts; i += 1) {
		int block_index = (
			 (floor(parts[i].x / block_width) * row_length)
			+ floor(parts[i].y / block_width)
		);
		blocks[block_index].push_back(parts[i]);
	}

}



static inline bool is_valid_rank(int rank, int num_procs) {
	return ((rank >= 0) && (rank < num_procs));
}

__attribute__((optimize("unroll-loops")))
void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {

	for (int my_block_index : my_block_indices) {
		for (int i = 0; i < blocks[my_block_index].size(); i += 1) {
			for (int j = 0; j < i; j += 1) {
				apply_force(blocks[my_block_index][i], blocks[my_block_index][j], true);
			}
			for (int neighbor_index : block_neighbors[my_block_index]) {
				for (int j = 0; j < blocks[neighbor_index].size(); j += 1) {
					apply_force(blocks[my_block_index][i], blocks[neighbor_index][j], false);
				}
			}
		}
	}

	for (int not_my_block_index : not_my_block_indices) {
		blocks[not_my_block_index].clear();
	}
	for (int block_index : my_block_indices) {
		for (int i = blocks[block_index].size() - 1; i >= 0; i -= 1) {
			move(blocks[block_index][i], size);
			int new_block_index = (
				 (floor(blocks[block_index][i].x / block_width) * row_length)
				+ floor(blocks[block_index][i].y / block_width)
			);
			if (new_block_index != block_index) {
				moved_particles[new_block_index].push_back(blocks[block_index][i]);
				blocks[block_index].erase(blocks[block_index].begin() + i);
			}
		}
	}
	for (int block_index = 0; block_index < num_blocks; block_index += 1) {
		while (moved_particles[block_index].size()) {
			particle_t particle = moved_particles[block_index].back();
			moved_particles[block_index].pop_back();
			blocks[block_index].push_back(particle);
		}
	}

	int top_send_buffer_index = 1;
	int bottom_send_buffer_index = 1;
	for (int top_ghost_block_index : my_top_ghost_block_indices) {
		for (int i = 0; i < blocks[top_ghost_block_index].size(); i += 1) {
			top_send_buffer[top_send_buffer_index] = blocks[top_ghost_block_index][i];
			top_send_buffer[top_send_buffer_index].ax = top_ghost_block_index;
			top_send_buffer_index += 1;
		}
	}
	for (int bottom_ghost_block_index : my_bottom_ghost_block_indices) {
		for (int i = 0; i < blocks[bottom_ghost_block_index].size(); i += 1) {
			bottom_send_buffer[bottom_send_buffer_index] = blocks[bottom_ghost_block_index][i];
			bottom_send_buffer[bottom_send_buffer_index].ax = bottom_ghost_block_index;
			bottom_send_buffer_index += 1;
		}
	}
	for (int top_neighbor_ghost_block_index : top_neighbor_ghost_block_indices) {
		for (int i = 0; i < blocks[top_neighbor_ghost_block_index].size(); i += 1) {
			top_send_buffer[top_send_buffer_index] = blocks[top_neighbor_ghost_block_index][i];
			top_send_buffer[top_send_buffer_index].ax = top_neighbor_ghost_block_index;
			top_send_buffer_index += 1;
		}
	}
	for (int bottom_neighbor_ghost_block_index : bottom_neighbor_ghost_block_indices) {
		for (int i = 0; i < blocks[bottom_neighbor_ghost_block_index].size(); i += 1) {
			bottom_send_buffer[bottom_send_buffer_index] = blocks[bottom_neighbor_ghost_block_index][i];
			bottom_send_buffer[bottom_send_buffer_index].ax = bottom_neighbor_ghost_block_index;
			bottom_send_buffer_index += 1;
		}
	}
	// Use a first "dummy" particle to send the processor
	//	rank and total number of particles in the message
	top_send_buffer[0].ax = top_send_buffer_index;
	bottom_send_buffer[0].ax = bottom_send_buffer_index;

	for (int i = 0; i < 4; i += 1) {
		send_recv_requests[i] = MPI_REQUEST_NULL;
	}

	int top_proc_rank = rank - 1;
	int bottom_proc_rank = rank + 1;

	if (is_valid_rank(top_proc_rank, num_procs)) {
		MPI_Isend(
			top_send_buffer,
			top_send_buffer_index,
			PARTICLE,
			top_proc_rank,
			0,
			MPI_COMM_WORLD,
			&(send_recv_requests[0])
		);
		MPI_Irecv(
			top_recv_buffer,
			send_recv_buffer_length,
			PARTICLE,
			top_proc_rank,
			0,
			MPI_COMM_WORLD,
			&(send_recv_requests[2])
		);
	}
	if (is_valid_rank(bottom_proc_rank, num_procs)) {
		MPI_Isend(
			bottom_send_buffer,
			bottom_send_buffer_index,
			PARTICLE,
			bottom_proc_rank,
			0,
			MPI_COMM_WORLD,
			&(send_recv_requests[1])
		);
		MPI_Irecv(
			bottom_recv_buffer,
			send_recv_buffer_length,
			PARTICLE,
			bottom_proc_rank,
			0,
			MPI_COMM_WORLD,
			&(send_recv_requests[3])
		);
	}
	MPI_Waitall(4, send_recv_requests, MPI_STATUSES_IGNORE);

	if (is_valid_rank(top_proc_rank, num_procs)) {
		for (int i = 1; i < top_recv_buffer[0].ax; i += 1) {
			int block_index = ((int) top_recv_buffer[i].ax);
			top_recv_buffer[i].ax = 0;
			blocks[block_index].push_back(top_recv_buffer[i]);
		}
	}
	if (is_valid_rank(bottom_proc_rank, num_procs)) {
		for (int i = 1; i < bottom_recv_buffer[0].ax; i += 1) {
			int block_index = ((int) bottom_recv_buffer[i].ax);
			bottom_recv_buffer[i].ax = 0;
			blocks[block_index].push_back(bottom_recv_buffer[i]);
		}
	}

}



void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {

	int* particle_counts = new int[num_procs];
	int* displacements = new int[num_procs];
	particle_t* sent_particles = new particle_t[num_parts];
	particle_t* received_particles = new particle_t[num_parts];
	
	int my_particle_count = 0;
	for (int my_block_index : my_block_indices) {
		my_particle_count += blocks[my_block_index].size();
	}
	
	MPI_Gather(
		&my_particle_count,
		1,
		MPI_INT,
		particle_counts,
		1,
		MPI_INT,
		0,
		MPI_COMM_WORLD
	);

	int sent_particles_index = 0;
	for (int my_block_index : my_block_indices) {
		for (int i = 0; i < blocks[my_block_index].size(); i += 1) {
			sent_particles[sent_particles_index] = blocks[my_block_index][i];
			sent_particles_index += 1;
		}
	}

	if (rank == 0) {
		displacements[0] = 0;
		for (int i = 1; i < num_procs; i += 1) {
			displacements[i] = displacements[i - 1] + particle_counts[i - 1];
		}
	}

	MPI_Gatherv(
		sent_particles,
		my_particle_count,
		PARTICLE,
		received_particles,
		particle_counts,
		displacements,
		PARTICLE,
		0,
		MPI_COMM_WORLD
	);

	if (rank == 0) {
		for (int i = 0; i < num_parts; i += 1) {
			int particle_id = received_particles[i].id;
			parts[particle_id - 1] = received_particles[i];
		}
	}

}
