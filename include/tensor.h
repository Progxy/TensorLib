#ifndef _TENSOR_H_
#define _TENSOR_H_

#include <assert.h>
#include <time.h>

#define ARR_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#define CAST_PTR(ptr, type) ((type*) (ptr))
#define NOT_USED(var) (void) var
#define MSG_MAX_LEN 512

typedef unsigned char bool;
typedef enum DataType { FLOAT_32, FLOAT_64, FLOAT_128 } DataType;
typedef struct Tensor {
    unsigned int* shape;
    unsigned int dim;
    void* data;
    DataType data_type;
} Tensor;

unsigned int calc_real_tensor_size(unsigned int* shape, unsigned int dim) {
    unsigned int size = 1;
    for (unsigned int i = 0; i < dim; ++i) size *= shape[i];
    return size;
}

Tensor create_tensor(unsigned int* shape, unsigned int dim, DataType data_type, char* err_msg) {
    Tensor tensor = { .shape = shape, .dim = dim, .data_type = data_type, .data = NULL };
    unsigned int real_tensor_size = calc_real_tensor_size(shape, dim);
    if (data_type == FLOAT_32) tensor.data = calloc(real_tensor_size, sizeof(float)); 
    else if (data_type == FLOAT_64) tensor.data = calloc(real_tensor_size, sizeof(double));
    else if (data_type == FLOAT_128) tensor.data = calloc(real_tensor_size, sizeof(long double));
    else {
        snprintf(err_msg, MSG_MAX_LEN, "INVALID_DATA_TYPE");
        return tensor;
    }
    if (tensor.data == NULL) snprintf(err_msg, MSG_MAX_LEN, "BAD_MEMORY");
    return tensor;
}

void calc_space(unsigned int index, Tensor tensor) {
    unsigned int temp = 1;
    if ((index + 1) % tensor.shape[tensor.dim - 1]) printf(", ");
    for (int i = tensor.dim - 1; i >= 0; --i) {
        temp *= tensor.shape[i];
        if (!((index + 1) % temp)) printf("\n");
    }
    return;
}

void print_tensor(Tensor tensor) {
    const unsigned int real_size = calc_real_tensor_size(tensor.shape, tensor.dim);
    printf("DEBUG_INFO: Tensor with shape: [ ");
    for (unsigned int i = 0; i < tensor.dim; ++i) {
        printf("%u%c ", tensor.shape[i], i == (tensor.dim - 1) ? '\0' : ',');
    }
    printf("]\n\n");
    for (unsigned int i = 0; i < real_size; ++i) {
        if (tensor.data_type == FLOAT_32) printf("%f", CAST_PTR(tensor.data, float)[i]);
        if (tensor.data_type == FLOAT_64) printf("%lf", CAST_PTR(tensor.data, double)[i]);
        if (tensor.data_type == FLOAT_128) printf("%Lf", CAST_PTR(tensor.data, long double)[i]);
        calc_space(i, tensor);
    }
    return;
}

void deallocate_tensor(Tensor tensor, bool delete_shape) {
    if (delete_shape) free(tensor.shape);
    free(tensor.data);
    return;
}

void init_seed() {
    srand(time(NULL));
    return;
}

void fill_tensor(void* val, Tensor tensor) {
    unsigned int real_size = calc_real_tensor_size(tensor.shape, tensor.dim);
    for (unsigned int i = 0; i < real_size; ++i) {
        if (tensor.data_type == FLOAT_32) CAST_PTR(tensor.data, float)[i] = *CAST_PTR(val, float);
        if (tensor.data_type == FLOAT_64) CAST_PTR(tensor.data, double)[i] = *CAST_PTR(val, double);
        if (tensor.data_type == FLOAT_128) CAST_PTR(tensor.data, long double)[i] = *CAST_PTR(val, long double);
    }
    return;
}

void randomize_tensor(Tensor tensor) {
    unsigned int real_size = calc_real_tensor_size(tensor.shape, tensor.dim);
    for (unsigned int i = 0; i < real_size; ++i) {
        long double value = (long double) rand() / RAND_MAX;
        if (tensor.data_type == FLOAT_32) CAST_PTR(tensor.data, float)[i] = (float) value;
        if (tensor.data_type == FLOAT_64) CAST_PTR(tensor.data, double)[i] = (double) value;
        if (tensor.data_type == FLOAT_128) CAST_PTR(tensor.data, long double)[i] = value;
    }
    return;
}

void reshape_tensor(Tensor* dest, Tensor base) {
    return;
}

Tensor sum_tensor(Tensor* c, Tensor a, Tensor b) {
    assert(a.dim == b.dim);
    assert(a.data_type == b.data_type);
    for (unsigned int i = 0; i < a.dim; ++i) {
        assert(a.shape[i] == b.shape[i]);
    }
    
    reshape_tensor(c, a);
    
    unsigned int real_size = calc_real_tensor_size(a.shape, a.dim);
    for (unsigned int i = 0; i < real_size; ++i) {
        long double value = (long double) rand() / RAND_MAX;
        if (a.data_type == FLOAT_32) CAST_PTR(a.data, float)[i] = (float) value;
        if (a.data_type == FLOAT_64) CAST_PTR(a.data, double)[i] = (double) value;
        if (a.data_type == FLOAT_128) CAST_PTR(a.data, long double)[i] = value;
    }

    return *c;
}

#endif //_TENSOR_H_