#include "FileIO.h"

void FileIO::tokenize(std::string str, std::vector<std::string> &token_v) {
		const char DELIMITER = ' ';
		size_t start = str.find_first_not_of(DELIMITER), end=start;

    while (start != std::string::npos){
        // Find next occurence of delimiter
        end = str.find(DELIMITER, start);
        // Push back the token found into vector
        token_v.push_back(str.substr(start, end-start));
        // Skip all occurences of the delimiter to find new start
        start = str.find_first_not_of(DELIMITER, end);
    }
}

void FileIO::set_file_prefix(std::string file_prefix) {
  std::string file_lattice_name = file_prefix+".pplattice";
  file_lattice.open(file_lattice_name.c_str());
  std::cout << "FileIO: " << "Opened set of files with prefix " << file_prefix << std::endl;
}

void FileIO::add_pp_out_qty(std::string name, int axis) {

  std::map<std::string,PPQty_t>::iterator i = pp_qty_map.find(name);
  if (i != pp_qty_map.end()) {
    pp_qty_vec_to_write.push_back( std::make_pair<std::string, int>(name, axis) );
  } else {
    std::cout << "no per-particle quantity " << name << " exists!" << std::endl;
  }

}

void FileIO::register_pp_qty(std::string name_, int type_, void* ptr_, int maxdim_) {
  PPQty_t qty = PPQty_t(name_, type_, ptr_, maxdim_);
  pp_qty_map.insert( std::pair<std::string, PPQty_t>(name_,qty));
}


void FileIO::write_per_atom_qty(std::string header_str) {
  std::vector<std::pair<std::string, int> >::iterator it = this->pp_qty_vec_to_write.begin();

  //header
  file_lattice << header_str;

  //variable names
  file_lattice << "# ";
  for(it=this->pp_qty_vec_to_write.begin(); it!=this->pp_qty_vec_to_write.end(); ++it) {
    file_lattice << std::setw(10) << it->first;
    if(it->second == -1) file_lattice << " ";
    else file_lattice <<"[" << it->second << "] ";
  }

  file_lattice << std::endl;

  //write N lines
  for(long i=0; i<N; i++) {

    for(it=this->pp_qty_vec_to_write.begin(); it!=this->pp_qty_vec_to_write.end(); ++it) {

      PPQty_t qt = pp_qty_map.at(it->first);
      int axis = it->second;

      //double4
      if(qt.type == 0) {
        if(axis == 0) file_lattice << std::setw(10) << static_cast<double4*>(qt.ptr)[i].x << "  ";
        if(axis == 1) file_lattice << std::setw(10) << static_cast<double4*>(qt.ptr)[i].y << "  ";
        if(axis == 2) file_lattice << std::setw(10) << static_cast<double4*>(qt.ptr)[i].z << "  ";
        if(axis == 3) file_lattice << std::setw(10) << static_cast<double4*>(qt.ptr)[i].w << "  ";
        
      	if(axis == 4) file_lattice << std::setw(10) << static_cast<double4*>(qt.ptr)[i].x << "  " 
      												<< static_cast<double4*>(qt.ptr)[i].y << "  " 
      												<< static_cast<double4*>(qt.ptr)[i].z << "  " 
      												<< static_cast<double4*>(qt.ptr)[i].w << "  ";
      }

      //double
      if(qt.type == 1) {
        file_lattice << std::setw(16) << static_cast<double*>(qt.ptr)[i] << "  ";
      }

      //long
      if(qt.type == 2) {
        file_lattice << std::setw(16) << static_cast<long*>(qt.ptr)[i] << "  ";
      }


      // file_lattice << "\t";
    }
    file_lattice << std::endl;
  }
  file_lattice << std::endl << std::endl;


}
