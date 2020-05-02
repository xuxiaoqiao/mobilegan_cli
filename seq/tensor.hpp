#ifndef SEQ_TENSOR_HPP
#define SEQ_TENSOR_HPP

#include <vector>
#include <array>
#include <cassert>

class Tensor3D {
private:
  int dim1;
  int dim2;
  int dim3;
  int _dim2_dim3_prod;
  std::vector<float> data;
public:
  Tensor3D() = delete;
  Tensor3D(int dim1, int dim2, int dim3)
      : dim1(dim1), dim2(dim2), dim3(dim3), data(dim1 * dim2 * dim3, 0.0),
        _dim2_dim3_prod(dim2 * dim3) {
      assert(dim1 > 0 && dim2 > 0 && dim3 > 0);
  }
  float operator()(int idx1, int idx2, int idx3) const {
      int idx = idx1 * _dim2_dim3_prod + idx2 * dim3 + idx3;
      return data[idx];
  }
  float &operator()(int idx1, int idx2, int idx3) {
      int idx = idx1 * _dim2_dim3_prod + idx2 * dim3 + idx3;
      return data[idx];
  }
  float *buffer() {
      return data.data();
  }
  std::array<int, 3> shape() const {
      return {dim1, dim2, dim3};
  }
};

class Tensor4D {
private:
  int dim1;
  int dim2;
  int dim3;
  int dim4;
  int _dim2_dim3_dim4_prod;
  int _dim3_dim4_prod;
  std::vector<float> data;
public:
  Tensor4D(int dim1, int dim2, int dim3, int dim4)
      : dim1(dim1), dim2(dim2), dim3(dim3), dim4(dim4),
        data(dim1 * dim2 * dim3 * dim4, 0.0),
        _dim2_dim3_dim4_prod(dim2 * dim3 * dim4),
        _dim3_dim4_prod(dim3 * dim4) {
      assert(dim1 > 0 && dim2 > 0 && dim3 > 0 && dim4 > 0);
  }
  float operator()(int idx1, int idx2, int idx3, int idx4) const {
      int idx = idx1 * _dim2_dim3_dim4_prod + idx2 * _dim3_dim4_prod + idx3 * dim4 + idx4;
      return data[idx];
  }
  float &operator()(int idx1, int idx2, int idx3, int idx4) {
      int idx = idx1 * _dim2_dim3_dim4_prod + idx2 * _dim3_dim4_prod + idx3 * dim4 + idx4;
      return data[idx];
  }
  float *buffer() {
      return data.data();
  }
  std::array<int, 4> shape() {
      return {dim1, dim2, dim3, dim4};
  }
};

#endif // SEQ_TENSOR_HPP