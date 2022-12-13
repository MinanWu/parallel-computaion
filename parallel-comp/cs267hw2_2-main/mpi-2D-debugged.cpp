#include "common.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string.h>
#include <vector>
#include <numeric>

#include <mpi.h>

int panel_buffer_size;
int block_buffer_size;
int rank_size;
int possible_procs;


double block_width;
int blocks_per_row;
int blocks_per_col;
int ghost_size;

int panel_x;
int panel_y;

int panels_per_row;
int panels_per_col;

int blocks_per_panel_row;
int blocks_per_panel_col;

int num_blocks;
int num_panels;
int num_blocks_per_panel;

typedef struct ghost_particle {
	uint64_t id;
	int block_index;
	double x;
	double y;
} ghost_particle_t;

std::vector<std::vector<particle_t>> blocks;
std::vector<std::vector<particle_t>> ghost_blocks;

std::vector<std::vector<int>> block_neighbors;
std::vector<std::vector<int>> block_ghost_neighbors;

std::vector<std::vector<particle_t>> send_parts_buffer;
std::vector<std::vector<particle_t>> recv_parts_buffer;
std::vector<std::vector<particle_t>> blocks_moved_parts;

std::vector<MPI_Request> recv_ghost_requests;
std::vector<MPI_Request> recv_requests;

std::vector<MPI_Status> recv_ghost_status;
std::vector<MPI_Status> recv_status;

std::vector<int> my_block_indices;
std::vector<int> not_my_block_indices;
std::vector<int> my_ghost_block_indices;
std::vector<int> my_proc_neighbors;

std::vector<std::vector<ghost_particle_t>> proc_ghost_particles;

MPI_Request request_send_parts, request_recv_parts, request_send_ghost, request_recv_ghost;
MPI_Status status_ghost, status_parts;

MPI_Group world_group;
std::vector<MPI_Group> proc_neighbor_groups;
std::vector<MPI_Comm> proc_neighbor_comms;
//MPI_Comm WORK_PROC;

MPI_Datatype GHOST_PARTICLE_DATATYPE;

void reset_ghost_blocks() {
	for (int i = 0; i < ghost_size; i++) {
		ghost_blocks[i].clear();
		ghost_blocks[i].resize(block_buffer_size);
	}
}

void reset_send_buffer() {
	for (int i = 0; i < send_parts_buffer.size(); i++) {
		send_parts_buffer[i].clear();
	}	
}

void reset_recv_buffer() {
	for (int i = 0; i < recv_parts_buffer.size(); i++) {
		recv_parts_buffer[i].clear();
		recv_parts_buffer[i].resize(panel_buffer_size);
	}
}

static inline void assign_ghost_block_neighbors(int block_index) {
	int row_index = std::floor(block_index / blocks_per_panel_row);
	int column_index = block_index % blocks_per_panel_row;

	// if (rank == 135 && block_index == 17) {
	// 	std::cout << "row_index : " << row_index << "  column_index :  " << column_index << "\n";
	// }

	// Top row
	if (panel_y > 0 && row_index == 0 && column_index > 0 && column_index < (blocks_per_panel_row - 1)) {
		int top_left_index = column_index + 3;
		int top_center_index = column_index + 4;
		int top_right_index = column_index + 5;
		block_ghost_neighbors[block_index].emplace_back(top_left_index);
		block_ghost_neighbors[block_index].emplace_back(top_center_index);
		block_ghost_neighbors[block_index].emplace_back(top_right_index);
	}
	// Left most column
	if (panel_x > 0 && column_index == 0 && row_index > 0 && row_index < (blocks_per_panel_col - 1)) {
		int top_left_index = 2 * blocks_per_panel_row + row_index + 3;
		int mid_left_index = 2 * blocks_per_panel_row + row_index + 4;
		int bottom_left_index = 2 * blocks_per_panel_row + row_index + 5;
		block_ghost_neighbors[block_index].emplace_back(top_left_index);
		block_ghost_neighbors[block_index].emplace_back(mid_left_index);
		block_ghost_neighbors[block_index].emplace_back(bottom_left_index);
	}
	// Right most column
	if (panel_x < (panels_per_row - 1) && column_index == (blocks_per_panel_row - 1) && row_index > 0 && row_index < (blocks_per_panel_col - 1)) {
		int top_right_index = 2 * blocks_per_panel_row + blocks_per_panel_col + row_index + 3;
		int mid_right_index = 2 * blocks_per_panel_row + blocks_per_panel_col + row_index + 4;
		int bottom_right_index = 2 * blocks_per_panel_row + blocks_per_panel_col + row_index + 5;
		block_ghost_neighbors[block_index].emplace_back(top_right_index);
		block_ghost_neighbors[block_index].emplace_back(mid_right_index);
		block_ghost_neighbors[block_index].emplace_back(bottom_right_index);
	}
	// Bottom row
	if (panel_y < (panels_per_col - 1) && row_index == (blocks_per_panel_col - 1) && column_index > 0 && column_index < (blocks_per_panel_row - 1)) {
		int bottom_left_index = blocks_per_panel_row + column_index + 3;
		int bottom_center_index = blocks_per_panel_row + column_index + 4;
		int bottom_right_index = blocks_per_panel_row + column_index + 5;
		block_ghost_neighbors[block_index].emplace_back(bottom_left_index);
		block_ghost_neighbors[block_index].emplace_back(bottom_center_index);
		block_ghost_neighbors[block_index].emplace_back(bottom_right_index);
	}
	// Top left corner
	if (row_index == 0 && column_index == 0) {
		if (panel_y > 0) {
			int top_center_index = 4 + column_index;
			int top_right_index = 4 + column_index + 1;
			block_ghost_neighbors[block_index].emplace_back(top_center_index);
			block_ghost_neighbors[block_index].emplace_back(top_right_index);
		}
		if (panel_x > 0) {
			int mid_left_index = 4 + 2 * blocks_per_panel_row + row_index;
			int bottom_left_index = 4 + 2 * blocks_per_panel_row + row_index + 1;
			block_ghost_neighbors[block_index].emplace_back(mid_left_index);
			block_ghost_neighbors[block_index].emplace_back(bottom_left_index);
		}
		if (panel_x > 0 && panel_y > 0) {
			int top_left_index = 0;
			block_ghost_neighbors[block_index].emplace_back(top_left_index);			
		}
	}	
	// Top right corner
	if (row_index == 0 && column_index == (blocks_per_panel_row - 1)) {
		if (panel_y > 0) {
			int top_left_index = 4 + column_index - 1;
			int top_center_index = 4 + column_index;
			block_ghost_neighbors[block_index].emplace_back(top_center_index);
			block_ghost_neighbors[block_index].emplace_back(top_left_index);
		}
		if (panel_x < (panels_per_row - 1)) {
			int mid_right_index = 4 + blocks_per_panel_col + 2 * blocks_per_panel_row + row_index;
			int bottom_right_index = 4 + blocks_per_panel_col + 2 * blocks_per_panel_row + row_index + 1;
			block_ghost_neighbors[block_index].emplace_back(mid_right_index);
			block_ghost_neighbors[block_index].emplace_back(bottom_right_index);
		}
		if (panel_x < (panels_per_row - 1) && panel_y > 0) {
			int top_right_index = 1;
			block_ghost_neighbors[block_index].emplace_back(top_right_index);			
		}
	}	
	// Bottom left corner
	if (row_index == (blocks_per_panel_col - 1) && column_index == 0) {
		if (panel_y < (panels_per_col - 1)) {
			int bottom_center_index = 4 + blocks_per_panel_row + column_index;
			int bottom_right_index = 4 + blocks_per_panel_row + column_index + 1;
			block_ghost_neighbors[block_index].emplace_back(bottom_center_index);
			block_ghost_neighbors[block_index].emplace_back(bottom_right_index);
		}
		if (panel_x > 0) {
			int top_left_index = 4 + 2 * blocks_per_panel_row + row_index - 1;
			int mid_left_index = 4 + 2 * blocks_per_panel_row + row_index;
			block_ghost_neighbors[block_index].emplace_back(mid_left_index);
			block_ghost_neighbors[block_index].emplace_back(top_left_index);
		}
		if (panel_x > 0 && panel_y < (panels_per_col - 1)) {
			int bottom_left_index = 2;
			block_ghost_neighbors[block_index].emplace_back(bottom_left_index);			
		}
	}	
	// Bottom right corner
	if (row_index == (blocks_per_panel_col - 1) && column_index == (blocks_per_panel_row - 1)) {
		if (panel_y < (panels_per_col - 1)) {
			int bottom_center_index = 4 + blocks_per_panel_row + column_index;
			int bottom_left_index = 4 + blocks_per_panel_row + column_index - 1;
			block_ghost_neighbors[block_index].emplace_back(bottom_center_index);
			block_ghost_neighbors[block_index].emplace_back(bottom_left_index);
		}
		if (panel_x < (panels_per_row - 1)) {
			int mid_right_index = 4 + blocks_per_panel_col + 2 * blocks_per_panel_row + row_index;
			int top_right_index = 4 + blocks_per_panel_col + 2 * blocks_per_panel_row + row_index - 1;
			block_ghost_neighbors[block_index].emplace_back(mid_right_index);
			block_ghost_neighbors[block_index].emplace_back(top_right_index);
		}
		if (panel_y < (panels_per_col - 1) && panel_x < (panels_per_row - 1)) {
			int bottom_right_index = 3;
			block_ghost_neighbors[block_index].emplace_back(bottom_right_index);			
		}
	}		
}

static inline void assign_block_neighbors(int block_index) {
	int row_index = std::floor(block_index / blocks_per_panel_row);
	int column_index = block_index % blocks_per_panel_row;
	if (row_index > 0) {
		if (column_index < (blocks_per_panel_row - 1)) {
			int top_right_index = block_index - blocks_per_panel_row + 1;
			block_neighbors[block_index].emplace_back(top_right_index);
		}
	}
	if (column_index < (blocks_per_panel_row - 1)) {
		int mid_right_index = block_index + 1;
		block_neighbors[block_index].emplace_back(mid_right_index);
	}
	if (row_index < (blocks_per_panel_col - 1)) {
		int bottom_center_index = block_index + blocks_per_panel_row;
		block_neighbors[block_index].emplace_back(bottom_center_index);
		if (column_index < (blocks_per_panel_row - 1)) {
			int bottom_right_index = block_index + blocks_per_panel_row + 1;
			block_neighbors[block_index].emplace_back(bottom_right_index);
		}
	}
}

void send_ghost_parts(int rank) {
	if (panel_x > 0 && panel_y > 0) {
		int block_id = 0;
		int size = blocks[block_id].size();
		//MPI_Isend(&blocks[block_id][0], size, PARTICLE, rank - panels_per_row - 1, 0, WORK_PROC, &request_send_ghost);
		MPI_Isend(&blocks[block_id][0], size, PARTICLE, rank - panels_per_row - 1, 0, MPI_COMM_WORLD, &request_send_ghost);
	}
	if (panel_x < (panels_per_row - 1) && panel_y > 0) {
		int block_id = blocks_per_panel_row - 1;
		int size = blocks[block_id].size();
		//MPI_Isend(&blocks[block_id][0], size, PARTICLE, rank - panels_per_row + 1, 1, WORK_PROC, &request_send_ghost);
		MPI_Isend(&blocks[block_id][0], size, PARTICLE, rank - panels_per_row + 1, 1, MPI_COMM_WORLD, &request_send_ghost);
	}
	if (panel_x > 0 && panel_y < (panels_per_col - 1)) {
		int block_id = (blocks_per_panel_col - 1) * blocks_per_panel_row;
		int size = blocks[block_id].size();
		//MPI_Isend(&blocks[block_id][0], size, PARTICLE, rank + panels_per_row - 1, 2, WORK_PROC, &request_send_ghost);
		MPI_Isend(&blocks[block_id][0], size, PARTICLE, rank + panels_per_row - 1, 2, MPI_COMM_WORLD, &request_send_ghost);
	} 
	if (panel_x < (panels_per_row - 1) && panel_y < (panels_per_col - 1)) {
		int block_id = blocks_per_panel_col * blocks_per_panel_row - 1;
		int size = blocks[block_id].size();
		//MPI_Isend(&blocks[block_id][0], size, PARTICLE, rank + panels_per_row + 1, 3, WORK_PROC, &request_send_ghost);
		MPI_Isend(&blocks[block_id][0], size, PARTICLE, rank + panels_per_row + 1, 3, MPI_COMM_WORLD, &request_send_ghost);
	}
	if (panel_y > 0) {
		for (int i = 0; i < blocks_per_panel_row; i++) {
			int block_id = i;
			int size = blocks[block_id].size();
			//MPI_Isend(&blocks[block_id][0], size, PARTICLE, rank - panels_per_row, 4 + i, WORK_PROC, &request_send_ghost);
			MPI_Isend(&blocks[block_id][0], size, PARTICLE, rank - panels_per_row, 4 + i, MPI_COMM_WORLD, &request_send_ghost);
		}
	}
	if (panel_y < (panels_per_col - 1)) {
		for (int i = 0; i < blocks_per_panel_row; i++) {
			int block_id = i + (blocks_per_panel_col - 1) * blocks_per_panel_row;
			int size = blocks[block_id].size();
			//MPI_Isend(&blocks[block_id][0], size, PARTICLE, rank + panels_per_row, 4 + blocks_per_panel_row + i, WORK_PROC, &request_send_ghost);
			MPI_Isend(&blocks[block_id][0], size, PARTICLE, rank + panels_per_row, 4 + blocks_per_panel_row + i, MPI_COMM_WORLD, &request_send_ghost);
		}
	}
	if (panel_x > 0) {
		for (int i = 0; i < blocks_per_panel_col; i++) {
			int block_id = i * blocks_per_panel_row;
			int size = blocks[block_id].size();
			//MPI_Isend(&blocks[block_id][0], size, PARTICLE, rank - 1, 4 + 2 * blocks_per_panel_row + i, WORK_PROC, &request_send_ghost);
			MPI_Isend(&blocks[block_id][0], size, PARTICLE, rank - 1, 4 + 2 * blocks_per_panel_row + i, MPI_COMM_WORLD, &request_send_ghost);
		}
	}
	if (panel_x < (panels_per_row - 1)) {
		for (int i = 0; i < blocks_per_panel_col; i++) {
			int block_id = (i + 1) * blocks_per_panel_row - 1;
			int size = blocks[block_id].size();
			//MPI_Isend(&blocks[block_id][0], size, PARTICLE, rank + 1, 4 + 2 * blocks_per_panel_row + blocks_per_panel_col + i, WORK_PROC, &request_send_ghost);
			MPI_Isend(&blocks[block_id][0], size, PARTICLE, rank + 1, 4 + 2 * blocks_per_panel_row + blocks_per_panel_col + i, MPI_COMM_WORLD, &request_send_ghost);
		}
	}
}


void receive_ghost_parts(int rank) {
	if (panel_x > 0 && panel_y > 0) {
		//MPI_Irecv(&ghost_blocks[0][0], block_buffer_size, PARTICLE, rank-panels_per_row-1, 3, WORK_PROC, &recv_ghost_requests[0]);
		MPI_Irecv(&ghost_blocks[0][0], block_buffer_size, PARTICLE, rank-panels_per_row-1, 3, MPI_COMM_WORLD, &recv_ghost_requests[0]);
	}

	if (panel_x < (panels_per_row-1) && panel_y > 0) {
		//MPI_Irecv(&ghost_blocks[1][0], block_buffer_size, PARTICLE, rank-panels_per_row+1, 2, WORK_PROC, &recv_ghost_requests[1]);
		MPI_Irecv(&ghost_blocks[1][0], block_buffer_size, PARTICLE, rank-panels_per_row+1, 2, MPI_COMM_WORLD, &recv_ghost_requests[1]);
	}

	if (panel_x > 0 && panel_y < (panels_per_col-1)) {
		//MPI_Irecv(&ghost_blocks[2][0], block_buffer_size, PARTICLE, rank+panels_per_row-1, 1, WORK_PROC, &recv_ghost_requests[2]);
		MPI_Irecv(&ghost_blocks[2][0], block_buffer_size, PARTICLE, rank+panels_per_row-1, 1, MPI_COMM_WORLD, &recv_ghost_requests[2]);
	} 

	if (panel_x < (panels_per_row-1) && panel_y < (panels_per_col-1)) {
		//MPI_Irecv(&ghost_blocks[3][0], block_buffer_size, PARTICLE, rank+panels_per_row+1, 0, WORK_PROC, &recv_ghost_requests[3]);
		MPI_Irecv(&ghost_blocks[3][0], block_buffer_size, PARTICLE, rank+panels_per_row+1, 0, MPI_COMM_WORLD, &recv_ghost_requests[3]);
	}

	if (panel_y > 0) {
		for (int i = 0; i < blocks_per_panel_row; i++) {
			//MPI_Irecv(&ghost_blocks[4+i][0], block_buffer_size, PARTICLE, rank - panels_per_row, 4 + blocks_per_panel_row + i, WORK_PROC, &recv_ghost_requests[4+i]);
			MPI_Irecv(&ghost_blocks[4+i][0], block_buffer_size, PARTICLE, rank - panels_per_row, 4 + blocks_per_panel_row + i, MPI_COMM_WORLD, &recv_ghost_requests[4+i]);
		}
	}
	if (panel_y < (panels_per_col-1)) {
		for (int i = 0; i < blocks_per_panel_row; i++) {
			//MPI_Irecv(&ghost_blocks[4+blocks_per_panel_row+i][0], block_buffer_size, PARTICLE, rank + panels_per_row, 4 + i, WORK_PROC, &recv_ghost_requests[4+blocks_per_panel_row+i]);
			MPI_Irecv(&ghost_blocks[4+blocks_per_panel_row+i][0], block_buffer_size, PARTICLE, rank + panels_per_row, 4 + i, MPI_COMM_WORLD, &recv_ghost_requests[4+blocks_per_panel_row+i]);
		}
	}
	if (panel_x > 0) {
		for (int i = 0; i < blocks_per_panel_col; i++) {
			//MPI_Irecv(&ghost_blocks[4+2*blocks_per_panel_row+i][0], block_buffer_size, PARTICLE, rank - 1, 4 + 2 * blocks_per_panel_row + blocks_per_panel_col + i, WORK_PROC, &recv_ghost_requests[4+2*blocks_per_panel_row+i]);
			MPI_Irecv(&ghost_blocks[4+2*blocks_per_panel_row+i][0], block_buffer_size, PARTICLE, rank - 1, 4 + 2 * blocks_per_panel_row + blocks_per_panel_col + i, MPI_COMM_WORLD, &recv_ghost_requests[4+2*blocks_per_panel_row+i]);
		}	
	}
	if (panel_x < (panels_per_row - 1)) {
		for (int i = 0; i < blocks_per_panel_col; i++) {
			//MPI_Irecv(&ghost_blocks[4+2*blocks_per_panel_row+blocks_per_panel_col+i][0], block_buffer_size, PARTICLE, rank + 1, 4 + 2 * blocks_per_panel_row + i, WORK_PROC, &recv_ghost_requests[4+2*blocks_per_panel_row+blocks_per_panel_col+i]);
			MPI_Irecv(&ghost_blocks[4+2*blocks_per_panel_row+blocks_per_panel_col+i][0], block_buffer_size, PARTICLE, rank + 1, 4 + 2 * blocks_per_panel_row + i, MPI_COMM_WORLD, &recv_ghost_requests[4+2*blocks_per_panel_row+blocks_per_panel_col+i]);
		}
	}
}

void wait_ghost_parts(int rank) {
	if (panel_x > 0 && panel_y > 0) {
		MPI_Wait(&recv_ghost_requests[0], &recv_ghost_status[0]);
	}
	if (panel_x < (panels_per_row-1) && panel_y > 0) {
		MPI_Wait(&recv_ghost_requests[1], &recv_ghost_status[1]);
	}
	if (panel_x > 0 && panel_y < (panels_per_col-1)) {
		MPI_Wait(&recv_ghost_requests[2], &recv_ghost_status[2]);
	} 
	if (panel_x < (panels_per_row-1) && panel_y < (panels_per_col-1)) {
		MPI_Wait(&recv_ghost_requests[3], &recv_ghost_status[3]);
	}
	if (panel_y > 0) {
		for (int i = 0; i < blocks_per_panel_row; i++) {
			MPI_Wait(&recv_ghost_requests[4+i], &recv_ghost_status[4+i]);
		}
	}
	if (panel_y < (panels_per_col-1)) {
		for (int i = 0; i < blocks_per_panel_row; i++) {
			MPI_Wait(&recv_ghost_requests[4+blocks_per_panel_row+i], &recv_ghost_status[4+blocks_per_panel_row+i]);
		}
	}
	if (panel_x > 0) {
		for (int i = 0; i < blocks_per_panel_col; i++) {
			MPI_Wait(&recv_ghost_requests[4+2*blocks_per_panel_row+i], &recv_ghost_status[4+2*blocks_per_panel_row+i]);
		}	
	}
	if (panel_x < (panels_per_row-1)) {
		for (int i = 0; i < blocks_per_panel_col; i++) {
			MPI_Wait(&recv_ghost_requests[4+2*blocks_per_panel_row+blocks_per_panel_col+i], &recv_ghost_status[4+2*blocks_per_panel_row+blocks_per_panel_col+i]);
		}
	}
}


void send_parts(int rank) {
	if (panel_x > 0 && panel_y > 0) {
		int size = send_parts_buffer[0].size();
		//MPI_Isend(&send_parts_buffer[0][0], size, PARTICLE, rank - panels_per_row - 1, 0, WORK_PROC, &request_send_parts);
		MPI_Isend(&send_parts_buffer[0][0], size, PARTICLE, rank - panels_per_row - 1, 0, MPI_COMM_WORLD, &request_send_parts);
	}
	if (panel_x < (panels_per_row - 1) && panel_y > 0) {
		int size = send_parts_buffer[1].size();
		//MPI_Isend(&send_parts_buffer[1][0], size, PARTICLE, rank - panels_per_row + 1, 1, WORK_PROC, &request_send_parts);
		MPI_Isend(&send_parts_buffer[1][0], size, PARTICLE, rank - panels_per_row + 1, 1, MPI_COMM_WORLD, &request_send_parts);
	}
	if (panel_x > 0 && panel_y < (panels_per_col - 1)) {
		int size = send_parts_buffer[2].size();
		//MPI_Isend(&send_parts_buffer[2][0], size, PARTICLE, rank + panels_per_row - 1, 2, WORK_PROC, &request_send_parts);
		MPI_Isend(&send_parts_buffer[2][0], size, PARTICLE, rank + panels_per_row - 1, 2, MPI_COMM_WORLD, &request_send_parts);
	} 
	if (panel_x < (panels_per_row - 1) && panel_y < (panels_per_col - 1)) {
		int size = send_parts_buffer[3].size();
		//MPI_Isend(&send_parts_buffer[3][0], size, PARTICLE, rank + panels_per_row + 1, 3, WORK_PROC, &request_send_parts);
		MPI_Isend(&send_parts_buffer[3][0], size, PARTICLE, rank + panels_per_row + 1, 3, MPI_COMM_WORLD, &request_send_parts);
	}
	if (panel_y > 0) {
		int size = send_parts_buffer[4].size();
		//MPI_Isend(&send_parts_buffer[4][0], size, PARTICLE, rank - panels_per_row, 4, WORK_PROC, &request_send_parts);
		MPI_Isend(&send_parts_buffer[4][0], size, PARTICLE, rank - panels_per_row, 4, MPI_COMM_WORLD, &request_send_parts);
	}
	if (panel_y < (panels_per_col - 1)) {
		int size = send_parts_buffer[5].size();
		//MPI_Isend(&send_parts_buffer[5][0], size, PARTICLE, rank + panels_per_row, 5, WORK_PROC, &request_send_parts);
		MPI_Isend(&send_parts_buffer[5][0], size, PARTICLE, rank + panels_per_row, 5, MPI_COMM_WORLD, &request_send_parts);
	}
	if (panel_x > 0) {
		int size = send_parts_buffer[6].size();
		//MPI_Isend(&send_parts_buffer[6][0], size, PARTICLE, rank - 1, 6, WORK_PROC, &request_send_parts);
		MPI_Isend(&send_parts_buffer[6][0], size, PARTICLE, rank - 1, 6, MPI_COMM_WORLD, &request_send_parts);
	}
	if (panel_x < (panels_per_row - 1)) {
		int size = send_parts_buffer[7].size();
		//MPI_Isend(&send_parts_buffer[7][0], size, PARTICLE, rank + 1, 7, WORK_PROC, &request_send_parts);
		MPI_Isend(&send_parts_buffer[7][0], size, PARTICLE, rank + 1, 7, MPI_COMM_WORLD, &request_send_parts);
	}
}


void recv_parts(int rank) {
	if (panel_x > 0 && panel_y > 0) {
		//MPI_Irecv(&recv_parts_buffer[0][0], panel_buffer_size, PARTICLE, rank - panels_per_row - 1, 3, WORK_PROC, &recv_requests[0]);
		MPI_Irecv(&recv_parts_buffer[0][0], panel_buffer_size, PARTICLE, rank - panels_per_row - 1, 3, MPI_COMM_WORLD, &recv_requests[0]);
	}
	if (panel_x < (panels_per_row - 1) && panel_y > 0) {
		//MPI_Irecv(&recv_parts_buffer[1][0], panel_buffer_size, PARTICLE, rank - panels_per_row + 1, 2, WORK_PROC, &recv_requests[1]);
		MPI_Irecv(&recv_parts_buffer[1][0], panel_buffer_size, PARTICLE, rank - panels_per_row + 1, 2, MPI_COMM_WORLD, &recv_requests[1]);
	}
	if (panel_x > 0 && panel_y < (panels_per_col - 1)) {
		//MPI_Irecv(&recv_parts_buffer[2][0], panel_buffer_size, PARTICLE, rank + panels_per_row - 1, 1, WORK_PROC, &recv_requests[2]);
		MPI_Irecv(&recv_parts_buffer[2][0], panel_buffer_size, PARTICLE, rank + panels_per_row - 1, 1, MPI_COMM_WORLD, &recv_requests[2]);
	} 
	if (panel_x < (panels_per_row - 1) && panel_y < (panels_per_col - 1)) {
		//MPI_Irecv(&recv_parts_buffer[3][0], panel_buffer_size, PARTICLE, rank + panels_per_row + 1, 0, WORK_PROC, &recv_requests[3]);
		MPI_Irecv(&recv_parts_buffer[3][0], panel_buffer_size, PARTICLE, rank + panels_per_row + 1, 0, MPI_COMM_WORLD, &recv_requests[3]);
	}
	if (panel_y > 0) {
		//MPI_Irecv(&recv_parts_buffer[4][0], panel_buffer_size, PARTICLE, rank - panels_per_row, 5, WORK_PROC, &recv_requests[4]);
		MPI_Irecv(&recv_parts_buffer[4][0], panel_buffer_size, PARTICLE, rank - panels_per_row, 5, MPI_COMM_WORLD, &recv_requests[4]);
	}
	if (panel_y < (panels_per_col - 1)) {
		//MPI_Irecv(&recv_parts_buffer[5][0], panel_buffer_size, PARTICLE, rank + panels_per_row, 4, WORK_PROC, &recv_requests[5]);
		MPI_Irecv(&recv_parts_buffer[5][0], panel_buffer_size, PARTICLE, rank + panels_per_row, 4, MPI_COMM_WORLD, &recv_requests[5]);
	}
	if (panel_x > 0) {
		//MPI_Irecv(&recv_parts_buffer[6][0], panel_buffer_size, PARTICLE, rank - 1, 7, WORK_PROC, &recv_requests[6]);
		MPI_Irecv(&recv_parts_buffer[6][0], panel_buffer_size, PARTICLE, rank - 1, 7, MPI_COMM_WORLD, &recv_requests[6]);
	}
	if (panel_x < (panels_per_row - 1)) {
		//MPI_Irecv(&recv_parts_buffer[7][0], panel_buffer_size, PARTICLE, rank + 1, 6, WORK_PROC, &recv_requests[7]);
		MPI_Irecv(&recv_parts_buffer[7][0], panel_buffer_size, PARTICLE, rank + 1, 6, MPI_COMM_WORLD, &recv_requests[7]);
	}
}


void wait_parts(int rank) {
	if (panel_x > 0 && panel_y > 0) {
		MPI_Wait(&recv_requests[0], &recv_status[0]);
	}
	if (panel_x < (panels_per_row-1) && panel_y > 0) {
		MPI_Wait(&recv_requests[1], &recv_status[1]);
	}
	if (panel_x > 0 && panel_y < (panels_per_col-1)) {
		MPI_Wait(&recv_requests[2], &recv_status[2]);
	}
	if (panel_x < (panels_per_row-1) && panel_y < (panels_per_col-1)) {
		MPI_Wait(&recv_requests[3], &recv_status[3]);
	}
	if (panel_y > 0) {
		MPI_Wait(&recv_requests[4], &recv_status[4]);
	}
	if (panel_y < (panels_per_col-1)) {
		MPI_Wait(&recv_requests[5], &recv_status[5]);
	}
	if (panel_x > 0) {
		MPI_Wait(&recv_requests[6], &recv_status[6]);
	}
	if (panel_x < (panels_per_row-1)) {
		MPI_Wait(&recv_requests[7], &recv_status[7]);	
	}
}


void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
	possible_procs = num_procs;

	do{	
    		panels_per_col = std::ceil(sqrt(possible_procs));
    		while(possible_procs % panels_per_col != 0) {
        		panels_per_col += 1;
    		}
    		panels_per_row = possible_procs / panels_per_col;

		rank_size = panels_per_col * panels_per_row;
		block_width = cutoff*4;

        blocks_per_panel_row = std::floor(size / panels_per_row / block_width);
        blocks_per_panel_col = std::floor(size / panels_per_col / block_width);
		
		// num_blocks_per_panel = blocks_per_panel_row * blocks_per_panel_col;

		possible_procs -= 1;

	} while(panels_per_row < 2 || panels_per_col < 2 || blocks_per_panel_row < 3 || blocks_per_panel_col < 3); 
	// decrease number of processors used in the cases the blocks_per_panel_* become 0 to avoid issues downstream



	if (rank < rank_size) {
		//MPI_Comm_split(MPI_COMM_WORLD, 0, rank, &WORK_PROC); //0 is color of work

        blocks_per_row = blocks_per_panel_row * panels_per_row;
        blocks_per_col = blocks_per_panel_col * panels_per_col;

        num_blocks = blocks_per_row * blocks_per_col;
        num_panels = panels_per_row * panels_per_col;
        num_blocks_per_panel = blocks_per_panel_row * blocks_per_panel_col;

        panel_buffer_size = std::ceil(num_parts / num_panels) * 8;
        block_buffer_size = std::ceil(num_parts / num_blocks) * 8;

        panel_x = rank % panels_per_row;
        panel_y = std::floor(rank / panels_per_row);
        ghost_size = 4 + 2 * blocks_per_panel_row + 2 * blocks_per_panel_col;

		// if (rank == 0) {
		// 	std::cout << "panels_per_row : " << panels_per_row << "   panels_per_col : " << panels_per_col << "   blocks_per_panel_row : " << blocks_per_panel_row << "   blocks_per_panel_col : " << blocks_per_panel_col << "   panel_x : " << panel_x << "   panel_y : " << panel_y << "\n";
		// }

        for (int i = 0; i < ghost_size; i++) {
            MPI_Request r;
            MPI_Status s;
            recv_ghost_requests.push_back(r);
            recv_ghost_status.push_back(s);
        }
        
        for (int i = 0; i < 8; i++) {
            MPI_Request r;
            MPI_Status s;		
            recv_requests.push_back(r);
            recv_status.push_back(s);
        }
        
        blocks.resize(num_blocks_per_panel);
        blocks_moved_parts.resize(num_blocks_per_panel);
        ghost_blocks.resize(ghost_size);

        send_parts_buffer.resize(8);
        recv_parts_buffer.resize(8);
        reset_send_buffer();
        reset_recv_buffer();
        reset_ghost_blocks();

        block_neighbors.resize(num_blocks_per_panel);
        block_ghost_neighbors.resize(num_blocks_per_panel);

        for (int block_index = 0; block_index < num_blocks_per_panel; block_index += 1) {
            assign_block_neighbors(block_index);
            assign_ghost_block_neighbors(block_index);
        }

		// debug code
		// if (rank == 135) {

		// 	for (int i = 0; i < blocks_per_panel_col; i += 1) { 
		// 		for (int j = 0; j < blocks_per_panel_row; j+=1) {
		// 			int idx = i * blocks_per_panel_row + j;
		// 			std::cout << "block idx : " << idx << "  neighbors :  \n";
		// 			for (int k = 0; k < block_neighbors[idx].size(); k++) {
		// 				std::cout << block_neighbors[idx][k] << "  ";
		// 			}
		// 			std::cout << " \n ";
		// 			std::cout << "  ghost neighbors :  \n";
		// 			for (int k = 0; k < block_ghost_neighbors[idx].size(); k++) {
		// 				std::cout << block_ghost_neighbors[idx][k] << "  ";
		// 			}
		// 			std::cout << " \n ";
		// 			std::cout << " \n ";
		// 		}
		// 	}

		// }
		// debug code

        for (int i = 0; i < num_parts; i += 1) {
            int block_index_x = floor(parts[i].x / size * blocks_per_row);
            int block_index_y = floor(parts[i].y / size * blocks_per_col);
            int panel_index_x = floor(block_index_x / blocks_per_panel_row);
            int panel_index_y = floor(block_index_y / blocks_per_panel_col);
            if (panel_x == panel_index_x && panel_y == panel_index_y) {
                int block_panel_index = block_index_x % blocks_per_panel_row + block_index_y % blocks_per_panel_col * blocks_per_panel_row;
                parts[i].ax = 0;     
                parts[i].ay = 0;
                blocks[block_panel_index].push_back(parts[i]);
            }
        }
	}//else{
		//MPI_Comm_split(MPI_COMM_WORLD, 1, rank, &WORK_PROC); //1 is color of not work?
	//}

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

	// This should be the symmetric, opposite force
	if (symmetric) {
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

	if (rank < rank_size) {
        send_ghost_parts(rank);
        //MPI_Barrier(WORK_PROC);
	// MPI_Barrier(MPI_COMM_WORLD);
        receive_ghost_parts(rank);
        //MPI_Barrier(WORK_PROC);
	// MPI_Barrier(MPI_COMM_WORLD);
        MPI_Wait(&request_send_ghost, &status_ghost);
        //MPI_Barrier(WORK_PROC);
	// MPI_Barrier(MPI_COMM_WORLD);
        wait_ghost_parts(rank);
        //MPI_Barrier(WORK_PROC);
	// MPI_Barrier(MPI_COMM_WORLD);

        for (int block_index = 0; block_index < num_blocks_per_panel; block_index++) {
            for (int i = 0; i < blocks[block_index].size(); i += 1) {
                for (int j = 0; j < i; j += 1) {
                    apply_force(blocks[block_index][i], blocks[block_index][j], true);
                }
                for (int neighbor_index : block_neighbors[block_index]) {
                    for (int j = 0; j < blocks[neighbor_index].size(); j += 1) {
                        apply_force(blocks[block_index][i], blocks[neighbor_index][j], true);
                    }
                }
                for (int ghost_block_index : block_ghost_neighbors[block_index]) {
                    int cnt;
                    MPI_Get_count(&recv_ghost_status[ghost_block_index], PARTICLE, &cnt);
                    for (int j = 0; j < cnt; j += 1) {
                        apply_force(blocks[block_index][i], ghost_blocks[ghost_block_index][j], false);
                    }
                }
            }
        }
        //MPI_Barrier(WORK_PROC);
	// MPI_Barrier(MPI_COMM_WORLD);
        
        for (int block_index = 0; block_index < num_blocks_per_panel; block_index++) {
            for (int i = blocks[block_index].size() - 1; i >= 0; i--) {
                move(blocks[block_index][i], size);
                int block_index_x = floor(blocks[block_index][i].x / size * blocks_per_row);
                int block_index_y = floor(blocks[block_index][i].y / size * blocks_per_col);
                int panel_index_x = floor(block_index_x / blocks_per_panel_row);
                int panel_index_y = floor(block_index_y / blocks_per_panel_col);
                if (panel_index_x == panel_x && panel_index_y == panel_y) {
                    int block_panel_index = block_index_x % blocks_per_panel_row + block_index_y % blocks_per_panel_col * blocks_per_panel_row;
                    if (block_index != block_panel_index) {
                        particle_t particle = blocks[block_index][i];					
                        blocks_moved_parts[block_panel_index].push_back(particle);
                        blocks[block_index].erase(blocks[block_index].begin() + i);
                    }
                } else {
                    int new_rank = panel_index_y * panels_per_row + panel_index_x;
                    if (new_rank == rank - panels_per_row - 1) {
                        particle_t particle = blocks[block_index][i];
                        send_parts_buffer[0].push_back(particle);
                        blocks[block_index].erase(blocks[block_index].begin() + i);
                    } 
                    else if (new_rank == rank - panels_per_row + 1) {			
                        particle_t particle = blocks[block_index][i];
                        send_parts_buffer[1].push_back(particle);
                        blocks[block_index].erase(blocks[block_index].begin() + i);
                    } 
                    else if (new_rank == rank + panels_per_row - 1) {	
                        particle_t particle = blocks[block_index][i];
                        send_parts_buffer[2].push_back(particle);
                        blocks[block_index].erase(blocks[block_index].begin() + i);
                    } 
                    else if (new_rank == rank + panels_per_row + 1) {	
                        particle_t particle = blocks[block_index][i];
                        send_parts_buffer[3].push_back(particle);
                        blocks[block_index].erase(blocks[block_index].begin() + i);
                    } 
                    else if (new_rank == rank - panels_per_row) {
                        particle_t particle = blocks[block_index][i];
                        send_parts_buffer[4].push_back(particle);
                        blocks[block_index].erase(blocks[block_index].begin() + i);
                    } 
                    else if (new_rank == rank + panels_per_row) {
                        particle_t particle = blocks[block_index][i];
                        send_parts_buffer[5].push_back(particle);
                        blocks[block_index].erase(blocks[block_index].begin() + i);
                    } 
                    else if (new_rank == rank - 1) {
                        particle_t particle = blocks[block_index][i];
                        send_parts_buffer[6].push_back(particle);
                        blocks[block_index].erase(blocks[block_index].begin() + i);
                    } 
                    else if (new_rank == rank + 1) {			
                        particle_t particle = blocks[block_index][i];
                        send_parts_buffer[7].push_back(particle);
                        blocks[block_index].erase(blocks[block_index].begin() + i);					
                    } 
					// debug code
					else {
						// if (rank == 0) {
					std::cout << "bug regular target. " << new_rank << "  " << rank <<  "  \n";
					std::cout << blocks[block_index][i].x << "  " << blocks[block_index][i].y << "  " << block_index_x << "  " <<  block_index_y << "  " << panel_index_x << "  " << panel_index_y << "  " <<  panels_per_row << "  " << panels_per_col << " " << blocks_per_row << " " << blocks_per_col << "  \n";
						// }
					}
					// debug code
                }
            }
        }

        //MPI_Barrier(WORK_PROC);
	// MPI_Barrier(MPI_COMM_WORLD);
        send_parts(rank);
		recv_parts(rank);
        //MPI_Barrier(WORK_PROC);
	// MPI_Barrier(MPI_COMM_WORLD);	
        for (int block_index = 0; block_index < num_blocks_per_panel; block_index += 1) {
            while (blocks_moved_parts[block_index].size() > 0) {
                particle_t particle = blocks_moved_parts[block_index].back();
                blocks_moved_parts[block_index].pop_back();
                blocks[block_index].push_back(particle);	
            }
        }

        MPI_Wait(&request_send_parts, &status_parts);
        //MPI_Barrier(WORK_PROC);
	// MPI_Barrier(MPI_COMM_WORLD);
        reset_send_buffer();
        //MPI_Barrier(WORK_PROC);
	// MPI_Barrier(MPI_COMM_WORLD);
        wait_parts(rank);
        //MPI_Barrier(WORK_PROC);
	// MPI_Barrier(MPI_COMM_WORLD);

        for (int i = 0; i < 8; i ++) {
			if (panel_x == 0 && panel_y == 0 && i == 0) {
				continue;
			} else if (panel_x == (panels_per_row-1) && panel_y == 0 && i == 1) {
				continue;
			} else if (panel_x == 0 && panel_y == (panels_per_col-1) && i == 2) {
				continue;
			} else if (panel_x == (panels_per_row-1) && panel_y == (panels_per_col-1) && i == 3) {
				continue;
			} else if (panel_y == 0 && i == 4) {
				continue;
			} else if (panel_y == (panels_per_col-1) && i == 5) {
				continue;
			} else if (panel_x == 0 && i == 6) {
				continue;
			} else if (panel_x == (panels_per_row-1) && i == 7) {
				continue;
			} else {
				int cnt;
				MPI_Get_count(&recv_status[i], PARTICLE, &cnt);
				for (int j = 0; j < cnt; j++) {
					particle_t particle = recv_parts_buffer[i][j];
					int block_index_x = floor(particle.x / size * blocks_per_row);
					int block_index_y = floor(particle.y / size * blocks_per_col);
					int block_panel_index = block_index_x % blocks_per_panel_row + block_index_y % blocks_per_panel_col * blocks_per_panel_row;
					blocks[block_panel_index].push_back(particle);				
            	}
			}
        }	

        //MPI_Barrier(WORK_PROC);
	// MPI_Barrier(MPI_COMM_WORLD);
	}else{
		// MPI_Barrier(MPI_COMM_WORLD);
		// MPI_Barrier(MPI_COMM_WORLD);
		// MPI_Barrier(MPI_COMM_WORLD);
		// MPI_Barrier(MPI_COMM_WORLD);
		// MPI_Barrier(MPI_COMM_WORLD);
		// MPI_Barrier(MPI_COMM_WORLD);
		// MPI_Barrier(MPI_COMM_WORLD);
		// MPI_Barrier(MPI_COMM_WORLD);
		// MPI_Barrier(MPI_COMM_WORLD);
		// MPI_Barrier(MPI_COMM_WORLD);
		// MPI_Barrier(MPI_COMM_WORLD);
	}

}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {

	if (rank < rank_size) {
		if (rank == 0) {
			int *global_cnt = new int[num_procs];
			particle_t *global_parts = new particle_t[num_parts];

			int size = 0;
			for (int i = 0; i < num_blocks_per_panel; i++) {
				for (int j = 0; j < blocks[i].size(); j++) {
					size++;
				}
			}
			//MPI_Gather(&size, 1, MPI_INT, global_cnt, 1, MPI_INT, 0, WORK_PROC);
			MPI_Gather(&size, 1, MPI_INT, global_cnt, 1, MPI_INT, 0, MPI_COMM_WORLD);
			int cnt = 0;
			particle_t *local_parts = new particle_t[size];
			for (int i = 0; i < num_blocks_per_panel; i++) {
				for (int j = 0; j < blocks[i].size(); j++) {
					local_parts[cnt] = blocks[i][j];
					cnt++;
				}
			}		
			int *displacements = new int[num_procs];
			displacements[0] = 0;
			for (int i = 1; i < num_procs; i++) {
				displacements[i] = displacements[i-1] + global_cnt[i-1];
			}
			//MPI_Gatherv(local_parts, size, PARTICLE, global_parts, global_cnt, displacements, PARTICLE, 0, WORK_PROC);
			MPI_Gatherv(local_parts, size, PARTICLE, global_parts, global_cnt, displacements, PARTICLE, 0, MPI_COMM_WORLD);
			for (int i = 0; i < num_parts; i++) {
				int idx = global_parts[i].id;
				parts[idx-1] = global_parts[i];
			}

		} else {
			int size = 0;
			for (int i = 0; i < num_blocks_per_panel; i++) {
				for (int j = 0; j < blocks[i].size(); j++) {
					size++;
				}
			}
			MPI_Gather(&size, 1, MPI_INT, NULL, 0, MPI_INT, 0, MPI_COMM_WORLD);
			int cnt = 0;
			particle_t *local_parts = new particle_t[size];
			for (int i = 0; i < num_blocks_per_panel; i++) {
				for (int j = 0; j < blocks[i].size(); j++) {
					local_parts[cnt] = blocks[i][j];
					cnt++;
				}
			}				
			//MPI_Gatherv(local_parts, size, PARTICLE, NULL, NULL, NULL, PARTICLE, 0, WORK_PROC);
			MPI_Gatherv(local_parts, size, PARTICLE, NULL, NULL, NULL, PARTICLE, 0, MPI_COMM_WORLD);
		}
	}else{
		int size = 0;
		particle_t *local_parts = new particle_t[size];
		MPI_Gather(&size, 1, MPI_INT, NULL, 0, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Gatherv(local_parts, size, PARTICLE, NULL, NULL, NULL, PARTICLE, 0, MPI_COMM_WORLD);
	}
	// MPI_Barrier(MPI_COMM_WORLD);
}
