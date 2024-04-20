#ifndef _TENSOR_H_
#define _TENSOR_H_

typedef enum DataType { FLOAT_32, FLOAT_64, FLOAT_128 } DataType;
typedef struct Tensor {
    unsigned int* shape;
    unsigned int dim;
    void* data;
    DataType data_type;
} Tensor;

unsigned int calc_real_tensor_size(unsigned int* shape, unsigned int dim) {
    unsigned int size = 0;
    for (unsigned int i = 0; i < dim; ++i) size *= shape[i];
    return size;
}

Tensor create_tensor(unsigned int* shape, unsigned int dim, DataType data_type) {
    Tensor tensor = { .shape = shape, .dim = dim, .data_type = data_type };
    unsigned int real_tensor_size = calc_real_tensor_size(shape, dim);
    if (data_type == FLOAT_32) tensor.data = calloc(real_tensor_size, sizeof(float)); 
    else if (data_type == FLOAT_64) tensor.data = calloc(real_tensor_size, sizeof(double));
    else if (data_type == FLOAT_128) tensor.data = calloc(real_tensor_size, sizeof(long double));
    else return (Tensor) {0};
    return tensor;
}

#endif //_TENSOR_H_