// #ifndef DEVICE_THERMO_ARRAY
// #define DEVICE_THERMO_ARRAY
//
// #include "precision.cuh"
//
// // array of thermo results, partially reduced over threads
// real *dev_thermo_v2; // for temperature
// real *dev_thermo_pe;
// real *dev_thermo_virial; // for pressure
// real4 *dev_thermo_v_ij; // vxx vyy vyx vyy for c.m velocity
//
// //virial tensor
// real *dev_thermo_virial_xx; // V_xx, V_yx, V_xy V_yy for stress
// real *dev_thermo_virial_xy;
// real *dev_thermo_virial_yy;
//
// //v_xx,v_yy,v_xy
// real *dev_thermo_v_xx;
// real *dev_thermo_v_xy;
// real *dev_thermo_v_yy;
//
// //born
// real *dev_thermo_born_xx;
// real *dev_thermo_born_xy;
// real *dev_thermo_born_yy;
//
// #endif
