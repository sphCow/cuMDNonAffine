#ifndef __SUM4_H__
#define __SUM4_H__

struct Sum4 {
  template <typename T>
  __host__ __device__ __forceinline__
  T operator()(const T &a, const T &b) const {
    T o;
    o.x = a.x + b.x;
    o.y = a.y + b.y;
    o.z = a.z + b.z;
    o.w = a.w + b.w;

    return o;
  }
};

template <typename T>
__host__ __device__ __forceinline__
T operator+=(T &a, const T &b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
    
    return a;
}

#endif
