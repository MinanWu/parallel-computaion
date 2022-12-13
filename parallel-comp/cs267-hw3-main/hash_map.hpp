// RMA version
#pragma once

#include "kmer_t.hpp"
#include <upcxx/upcxx.hpp>



struct HashMap {

	std::vector<upcxx::global_ptr<kmer_pair>> rank_data;
	std::vector<upcxx::global_ptr<int32_t>> rank_used;

	size_t total_size;
	size_t size_per_rank;

	upcxx::atomic_domain<int32_t> atomic_domain_used;

	HashMap(size_t size);
	~HashMap();

	bool insert(const kmer_pair& kmer);
	bool find(const pkmer_t& key_kmer, kmer_pair& val_kmer);

};



HashMap::HashMap(size_t size) : atomic_domain_used({upcxx::atomic_op::compare_exchange}) {
	total_size = size;
	size_per_rank = std::ceil(((double) size) / ((double) upcxx::rank_n()));
	rank_data.resize(upcxx::rank_n());
	rank_used.resize(upcxx::rank_n());
	upcxx::global_ptr<kmer_pair> my_data = upcxx::allocate<kmer_pair>(size_per_rank);
	upcxx::global_ptr<int32_t> my_used = upcxx::allocate<int32_t>(size_per_rank);
	for (int rank = 0; rank < upcxx::rank_n(); rank += 1) {
		rank_data[rank] = broadcast(my_data, rank).wait();
		rank_used[rank] = broadcast(my_used, rank).wait();
	}
	memset(my_used.local(), 0, size_per_rank);
}

HashMap::~HashMap() {
	atomic_domain_used.destroy();
}



bool HashMap::insert(const kmer_pair& kmer) {
	uint64_t hash = kmer.hash();
	uint64_t probe = 0;
	uint64_t slot_index;
	uint64_t destination_rank;
	uint64_t local_index;
	int32_t read_value;
	do {
		slot_index = (hash + probe) % total_size;
		destination_rank = (slot_index / size_per_rank);
		local_index = (slot_index % size_per_rank);
		atomic_domain_used.compare_exchange(
			rank_used[destination_rank] + local_index,
			0, 1,
			&read_value,
			std::memory_order_relaxed
		).wait();
		if (read_value == 0) {
			upcxx::rput(
				kmer,
				(rank_data[destination_rank] + local_index)
			).wait();
			return true;
		}
		probe += 1;
	}
	while (probe < total_size);
	return false;
}



bool HashMap::find(const pkmer_t& key_kmer, kmer_pair& val_kmer) {
	uint64_t hash = key_kmer.hash();
	uint64_t probe = 0;
	uint64_t slot_index;
	uint64_t destination_rank;
	uint64_t local_index;
	int32_t slot_status;
	do {
		slot_index = (hash + probe) % total_size;
		destination_rank = (slot_index / size_per_rank);
		local_index = (slot_index % size_per_rank);
		slot_status = upcxx::rget(
			(rank_used[destination_rank] + local_index)
		).wait();
		if (slot_status != 0) {
			val_kmer = upcxx::rget(
				(rank_data[destination_rank] + local_index)
			).wait();
			if (val_kmer.kmer == key_kmer) {
				return true;
			}
		}
		probe += 1;
	}
	while (probe < total_size);
	return false;
}
