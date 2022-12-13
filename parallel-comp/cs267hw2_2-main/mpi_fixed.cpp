#include "common.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string.h>
#include <vector>

#include <mpi.h>



double block_width;
int blocks_per_row;
int panels_per_row;
int panels_per_col;
int blocks_per_panel_row;
int blocks_per_panel_col;
int num_blocks;

std::vector<std::vector<particle_t>> blocks;
std::vector<std::vector<int>> block_neighbors;
std::vector<std::vector<particle_t>> moved_particles;

std::vector<int> my_block_indices;
std::vector<int> not_my_block_indices;
std::vector<int> my_ghost_block_indices;

std::vector<int> my_neighbor_procs;

int buffer_length;
particle_t* send_buffer;
particle_t* receive_buffers[8];

int gather_buffer_length;
particle_t* gather_buffer;



static inline void assign_block_neighbors(int block_index) {
	int row_index = (block_index / blocks_per_row);
	int column_index = (block_index % blocks_per_row);
	if (row_index > 0) {
		int top_center_index = block_index - blocks_per_row;
		block_neighbors[block_index].emplace_back(top_center_index);
		if (column_index > 0) {
			int top_left_index = block_index - blocks_per_row - 1;
			block_neighbors[block_index].emplace_back(top_left_index);
		}
		if (column_index < (blocks_per_row - 1)) {
			int top_right_index = block_index - blocks_per_row + 1;
			block_neighbors[block_index].emplace_back(top_right_index);
		}
	}
	if (column_index > 0) {
		int mid_left_index = block_index - 1;
		block_neighbors[block_index].emplace_back(mid_left_index);
	}
	if (column_index < (blocks_per_row - 1)) {
		int mid_right_index = block_index + 1;
		block_neighbors[block_index].emplace_back(mid_right_index);
	}
	if (row_index < (blocks_per_row - 1)) {
		int bottom_center_index = block_index + blocks_per_row;
		block_neighbors[block_index].emplace_back(bottom_center_index);
		if (column_index > 0) {
			int bottom_left_index = block_index + blocks_per_row - 1;
			block_neighbors[block_index].emplace_back(bottom_left_index);
		}
		if (column_index < (blocks_per_row - 1)) {
			int bottom_right_index = block_index + blocks_per_row + 1;
			block_neighbors[block_index].emplace_back(bottom_right_index);
		}
	}
}

static inline void assign_my_blocks(int rank) {
	int first_block_row_in_my_panel = std::floor(rank / (panels_per_row + 1)) * blocks_per_panel_row;
	int last_block_row_in_my_panel = first_block_row_in_my_panel + (blocks_per_panel_row - 1);
	int first_block_column_in_my_panel = (rank % panels_per_col) * blocks_per_panel_col;
	int last_block_column_in_my_panel =  first_block_column_in_my_panel + (blocks_per_panel_col - 1);
	for (int block_index = 0; block_index < num_blocks; block_index += 1) {
		int block_row_index = block_index / blocks_per_row;
		int block_column_index = block_index % blocks_per_row;
		if (
			   block_row_index >= first_block_row_in_my_panel
			&& block_row_index <= last_block_row_in_my_panel
			&& block_column_index >= first_block_column_in_my_panel
			&& block_column_index <= last_block_column_in_my_panel
		) {
			my_block_indices.push_back(block_index);
			if (
				   block_row_index == first_block_column_in_my_panel
				|| block_row_index == last_block_row_in_my_panel
				|| block_column_index == first_block_column_in_my_panel
				|| block_column_index == last_block_column_in_my_panel
			) {
				my_ghost_block_indices.push_back(block_index);
			}
		}
	}
}

static inline void assign_my_neighbor_procs(int rank, int num_procs) {
	int first_block_row_in_panel = std::floor(rank / (panels_per_row+1)) * blocks_per_panel_row;
	int last_block_row_in_panel = first_block_row_in_panel + (blocks_per_panel_row - 1);
	int first_block_column_in_panel = (rank % panels_per_col) * blocks_per_panel_col;
	int last_block_column_in_panel =  first_block_column_in_panel + (blocks_per_panel_col - 1);
	if(first_block_row_in_panel != 0){
		//Neighbor top left
		if(first_block_column_in_panel != 0){
			my_neighbor_procs.push_back(rank - panels_per_col - 1);
		}

		//Neighbor top middle
		my_neighbor_procs.push_back(rank - panels_per_col);

		//Neighbor top right
		if(last_block_column_in_panel != blocks_per_row - 1)
		{
			my_neighbor_procs.push_back(rank - panels_per_col + 1);
		}		
	}

	// For neighbor left
	if(first_block_column_in_panel != 0){
		//Neighbor left
		my_neighbor_procs.push_back(rank - 1);
	}

	// For neighbor right
	if(last_block_column_in_panel != blocks_per_row - 1){
		my_neighbor_procs.push_back(rank + 1);
	}

	// For neighbors bottom left, bottom middle, bottom right
	if(last_block_row_in_panel != blocks_per_row - 1){
		//Neighbor bottom left
		if(first_block_column_in_panel != 0)
		{
			my_neighbor_procs.push_back(rank + panels_per_col - 1);
		}

		//Neighbor bottom middle
		my_neighbor_procs.push_back(rank + panels_per_col);

		//Neighbor bottom right
		if(last_block_column_in_panel != blocks_per_row - 1)
		{
			my_neighbor_procs.push_back(rank + panels_per_col + 1);
		}
	}
	for (int i = my_neighbor_procs.size() - 1; i >= 0; i += 1) {
		if (my_neighbor_procs[i] < 0 || my_neighbor_procs[i] >= num_procs) {
			my_neighbor_procs.erase(my_neighbor_procs.begin() + i);
		}
	}
}



void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {

	panels_per_row = std::floor(sqrt(num_procs));
	panels_per_col = num_procs / panels_per_row;

	while (num_procs % panels_per_row != 0) {
		panels_per_row -= 1;
	}

	block_width = cutoff;
	blocks_per_row = std::ceil(size / block_width);
	// panels_per_row = std::ceil(sqrt(num_procs));
	blocks_per_panel_row = std::ceil(((double) blocks_per_row) / panels_per_row);
	blocks_per_panel_col = std::ceil(((double) blocks_per_row) / panels_per_col);
	num_blocks = pow(blocks_per_row, 2);

	blocks.resize(num_blocks);
	block_neighbors.resize(num_blocks);
	moved_particles.resize(num_blocks);

	for (int block_index = 0; block_index < num_blocks; block_index += 1) {
		assign_block_neighbors(block_index);
	}

	// TO-DO: integrate this with the `assign_block_neighbors` step
	assign_my_blocks(rank);
	assign_my_neighbor_procs(rank, num_procs);

	// TO-DO: Cache-align these buffers and convert them to malloc?
	buffer_length = num_parts;
	send_buffer = ((particle_t*) calloc(buffer_length, sizeof(particle_t)));
	for (int i = 0; i < 8; i += 1) {
		receive_buffers[i] = ((particle_t*) calloc(buffer_length, sizeof(particle_t)));
	}

	gather_buffer_length = (3 * num_parts);
	gather_buffer = ((particle_t*) calloc(gather_buffer_length, sizeof(particle_t)));

	for (int i = 0; i < num_parts; i += 1) {
		parts[i].ax = 0;     
		parts[i].ay = 0;
		int block_index = (
			 (floor(parts[i].x / block_width) * blocks_per_row)
			+ floor(parts[i].y / block_width)
		);
		blocks[block_index].push_back(parts[i]);
	}

}



__attribute__((always_inline))
static inline void apply_force(particle_t& particle, particle_t& neighbor, bool symmetric) {

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

	if (symmetric) {
		// This should be the symmetric, opposite force
		neighbor.ax -= coef * dx;
		neighbor.ay -= coef * dy;
	}

}

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



__attribute__((optimize("unroll-loops")))
void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {

	for (int block_index : my_block_indices) {
		for (int i = 0; i < blocks[block_index].size(); i += 1) {
			for (int j = 0; j < i; j += 1) {
				apply_force(blocks[block_index][i], blocks[block_index][j], true);
			}
			for (int neighbor_index : block_neighbors[block_index]) {
				for (int j = 0; j < blocks[neighbor_index].size(); j += 1) {
					apply_force(blocks[block_index][i], blocks[neighbor_index][j], false);
				}
			}
		}
	}

	// Make a list of all the particles that are switching blocks
	for (int block_index : my_block_indices) {
		for (int i = blocks[block_index].size() - 1; i >= 0; i -= 1) {
			move(blocks[block_index][i], size);
			int new_block_index = (
				 (floor(blocks[block_index][i].x / block_width) * blocks_per_row)
				+ floor(blocks[block_index][i].y / block_width)
			);
			if (new_block_index != block_index) {
				moved_particles[new_block_index].push_back(blocks[block_index][i]);
				blocks[block_index].erase(blocks[block_index].begin() + i);
			}
		}
	}

	// Clear out the blocks that I don't own
	for (int not_my_block_index : not_my_block_indices) {
		blocks[not_my_block_index].clear();
	}
	// Move particles to new blocks
	for (int block_index = 0; block_index < num_blocks; block_index += 1) {
		while (moved_particles[block_index].size() > 0) {
			particle_t particle = moved_particles[block_index].back();
			moved_particles[block_index].pop_back();
			blocks[block_index].push_back(particle);	
		}
	}

	int send_buffer_index = 1;
	for (int ghost_block_index : my_ghost_block_indices) {
		for (int i = 0; i < blocks[ghost_block_index].size(); i += 1) {
			send_buffer[send_buffer_index] = blocks[ghost_block_index][i];
			send_buffer[send_buffer_index].ax = ghost_block_index;
			send_buffer_index += 1;
		}
	}
	for (int not_my_block_index : not_my_block_indices) {
		if (blocks[not_my_block_index].size() > 0) {
			for (int i = 0; i < blocks[not_my_block_index].size(); i += 1) {
				send_buffer[send_buffer_index] = blocks[not_my_block_index][i];
				send_buffer[send_buffer_index].ax = not_my_block_index;
				send_buffer_index += 1;
			}
		}
	}
	// Use a first "dummy" particle to send the processor
	//	rank and total number of particles in the message
	send_buffer[0].id = rank;
	send_buffer[0].ax = send_buffer_index;

	// TO-DO: Move this out of this function
	MPI_Request send_receive_requests[16];
	for (int i = 0; i < 16; i += 1) {
		send_receive_requests[i] = MPI_REQUEST_NULL;
	}

	int request_index = 0;
	for (int neighbor_index = 0; neighbor_index < my_neighbor_procs.size(); neighbor_index += 1) {
		int error = MPI_Isend(
			send_buffer,
			send_buffer_index,
			PARTICLE,
			my_neighbor_procs[neighbor_index],
			0,
			MPI_COMM_WORLD,
			&(send_receive_requests[request_index])
		);
		request_index += 1;
	}
	for (int neighbor_index = 0; neighbor_index < my_neighbor_procs.size(); neighbor_index += 1) {
		int error = MPI_Irecv(
			&(receive_buffers[neighbor_index][0]),
			buffer_length,
			PARTICLE,
			my_neighbor_procs[neighbor_index],
			MPI_ANY_TAG,
			MPI_COMM_WORLD,
			&(send_receive_requests[request_index])
		);
		request_index += 1;
	}
	MPI_Waitall(16, send_receive_requests, MPI_STATUSES_IGNORE);

	for (int neighbor_index = 0; neighbor_index < my_neighbor_procs.size(); neighbor_index += 1) {
		int num_parts_received = ((int) receive_buffers[neighbor_index][0].ax);
		for (int i = 1; i < num_parts_received; i += 1) {
			int block_index = ((int) receive_buffers[neighbor_index][i].ax);
			receive_buffers[neighbor_index][i].ax = 0;
			blocks[block_index].push_back(receive_buffers[neighbor_index][i]);
		}
	}

	// TO-DO: Do we need this?
	MPI_Barrier(MPI_COMM_WORLD);

}



// Maybe try to make these parameters pass-by-reference instead?
bool has_lower_ID(particle_t particle_a, particle_t particle_b) {
	return (particle_a.id < particle_b.id);
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {

	int send_buffer_index = 1;
	for (int my_block_index : my_block_indices) {
		for (int i = 0; i < blocks[my_block_index].size(); i += 1) {
			send_buffer[send_buffer_index] = blocks[my_block_index][i];
			send_buffer_index += 1;
		}
	}
	// TO-DO: Do we need this?
	send_buffer[0].id = rank;
	send_buffer[0].ax = send_buffer_index;

	int send_size = std::floor(((double) gather_buffer_length) / num_procs);

	int mpi_result = MPI_Gather(
		send_buffer,
		send_size,
		PARTICLE,
		gather_buffer,
		send_size,	// Maybe try expanding this?
		PARTICLE,
		0,
		MPI_COMM_WORLD
	);
	if (mpi_result != MPI_SUCCESS) {
		std::cout << "MPI_Gather returned error code: " << mpi_result << std::endl;
	}

	if (rank == 0) {
		int parts_index = 0;
		for (int proc_rank = 0; proc_rank < num_procs; proc_rank += 1) {
			particle_t* proc_section = &(gather_buffer[send_size * proc_rank]);
			for (int i = 1; i < ((int) proc_section[0].ax); i += 1) {
				parts[parts_index] = proc_section[i];
				parts_index += 1;
			}
		}
		std::sort(parts, (parts + num_parts), has_lower_ID);
	}

}
