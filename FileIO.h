#ifndef FILE_IO
#define FILE_IO

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <utility>
#include <map>

#include "vector_types.h"
//#include "global.h"

struct PPQty_t {
  std::string name;
  void *ptr;
  int maxdim;
  int type;

  PPQty_t(){};

  PPQty_t(std::string name_,  int type_, void* ptr_, int maxdim_) {
    name = name_;
    ptr = ptr_;
    maxdim = maxdim_;
    type = type_;
  }
};

class FileIO {
private:
  std::ofstream file_lattice;
  std::vector< std::pair<std::string, int> > pp_qty_vec_to_write;
  std::map<std::string, PPQty_t> pp_qty_map;

  long N;

public:
  void tokenize(std::string str, std::vector<std::string> &token_v);
  void set_file_prefix(std::string);
  void write_lattice(std::string const&);
  void set_N(long N) {this->N = N; std::cout << "FileIO: N -> " << N << std::endl;};

  void register_pp_qty(std::string name_, int type_, void* ptr_, int maxdim_);

  void add_pp_out_qty(std::string name, int axis);


  void write_per_atom_qty(std::string header_str);



};

#endif
