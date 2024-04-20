#ifndef _TENSOR_H_
#define _TENSOR_H_

#include <time.h>
#include <stdarg.h>

#define ARR_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#define CAST_PTR(ptr, type) ((type*) (ptr))
#define ASSERT(condition, err_msg) assert(condition, __LINE__, __FILE__, err_msg);
#define DEALLOCATE_TENSORS(...) deallocate_tensors(sizeof((Tensor[]){__VA_ARGS__}) / sizeof(Tensor), __VA_ARGS__)
#define NOT_USED(var) (void) var
#define MSG_MAX_LEN 512
#define TRUE 1
#define FALSE 0

typedef unsigned char bool;
typedef enum DataType { FLOAT_32 = sizeof(float), FLOAT_64 = sizeof(double), FLOAT_128 = sizeof(long double) } DataType;
typedef struct Tensor {
    unsigned int* shape;
    unsigned int dim;
    void* data;
    DataType data_type;
} Tensor;

const unsigned char data_types[] = { FLOAT_32, FLOAT_64, FLOAT_128 };

void assert(bool condition, unsigned int line, char* file, char* err_msg) {
    if (condition) {
        printf("ERROR: Assert failed in file: %s:%u, with error: %s.\n", file, line, err_msg);
        exit(-1);
    }
    return;
}

void mem_copy(void* dest, void* src, unsigned char size, unsigned int n) {
    ASSERT(src == NULL, "NULL_POINTER");
    for (unsigned int i = 0; i < size * n; ++i) {
        CAST_PTR(dest, unsigned char)[i] = CAST_PTR(src, unsigned char)[i];
    }
    return;
}

unsigned int calc_tensor_size(unsigned int* shape, unsigned int dim) {
    unsigned int size = 1;
    for (unsigned int i = 0; i < dim; ++i) size *= shape[i];
    return size;
}

bool is_valid_data_type(DataType data_type) {
    for (unsigned int i = 0; i < ARR_SIZE(data_types); ++i) {
        if (data_type == data_types[i]) return TRUE;
    }
    return FALSE;
}

Tensor create_tensor(unsigned int* shape, unsigned int dim, DataType data_type) {
    ASSERT(!is_valid_data_type(data_type), "INVALID_DATA_TYPE");
    ASSERT(!dim, "INVALID_DIM");
    Tensor tensor = { .shape = NULL, .dim = dim, .data_type = data_type, .data = NULL };
    tensor.shape = (unsigned int*) calloc(tensor.dim, sizeof(unsigned int));
    ASSERT(tensor.shape == NULL, "BAD_MEMORY");
    mem_copy(tensor.shape, shape, sizeof(unsigned int), tensor.dim);
    tensor.data = calloc(calc_tensor_size(shape, dim), tensor.data_type); 
    ASSERT(tensor.data == NULL, "BAD_MEMORY");
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
    const unsigned int size = calc_tensor_size(tensor.shape, tensor.dim);
    printf("DEBUG_INFO: Tensor with shape: [ ");
    for (unsigned int i = 0; i < tensor.dim; ++i) {
        printf("%u%c ", tensor.shape[i], i == (tensor.dim - 1) ? '\0' : ',');
    }
    printf("]\n\n");
    for (unsigned int i = 0; i < size; ++i) {
        if (tensor.data_type == FLOAT_32) printf("%f", CAST_PTR(tensor.data, float)[i]);
        if (tensor.data_type == FLOAT_64) printf("%lf", CAST_PTR(tensor.data, double)[i]);
        if (tensor.data_type == FLOAT_128) printf("%Lf", CAST_PTR(tensor.data, long double)[i]);
        calc_space(i, tensor);
    }
    return;
}

void deallocate_tensors(int len, ...) {
    va_list args;
    va_start(args, len);

    for (int i = 0; i < len; ++i) {
        Tensor tensor = va_arg(args, Tensor);
        free(tensor.data);
        free(tensor.shape);
    }

    va_end(args);

    return;
}

void init_seed() {
    srand(time(NULL));
    return;
}

void fill_tensor(void* val, Tensor tensor) {
    unsigned int size = calc_tensor_size(tensor.shape, tensor.dim);
    for (unsigned int i = 0; i < size; ++i) {
        if (tensor.data_type == FLOAT_32) CAST_PTR(tensor.data, float)[i] = *CAST_PTR(val, float);
        if (tensor.data_type == FLOAT_64) CAST_PTR(tensor.data, double)[i] = *CAST_PTR(val, double);
        if (tensor.data_type == FLOAT_128) CAST_PTR(tensor.data, long double)[i] = *CAST_PTR(val, long double);
    }
    return;
}

void randomize_tensor(Tensor tensor) {
    unsigned int size = calc_tensor_size(tensor.shape, tensor.dim);
    for (unsigned int i = 0; i < size; ++i) {
        long double value = (long double) rand() / RAND_MAX;
        if (tensor.data_type == FLOAT_32) CAST_PTR(tensor.data, float)[i] = (float) value;
        if (tensor.data_type == FLOAT_64) CAST_PTR(tensor.data, double)[i] = (double) value;
        if (tensor.data_type == FLOAT_128) CAST_PTR(tensor.data, long double)[i] = value;
    }
    return;
}

void reshape_tensor(Tensor* dest, Tensor base) {
    dest -> shape = (unsigned int*) realloc(dest -> shape, sizeof(unsigned int) * base.dim);
    mem_copy(dest -> shape, base.shape, sizeof(unsigned int), base.dim);
    dest -> dim = base.dim;
    dest -> data_type = base.data_type;
    free(dest -> data);
    dest -> data = calloc(calc_tensor_size(dest -> shape, dest -> dim), dest -> data_type);
    ASSERT(dest -> data == NULL, "BAD_MEMORY");
    return;
}

Tensor sum_tensor(Tensor* c, Tensor a, Tensor b) {
    ASSERT(a.dim != b.dim, "DIM_MISMATCH");
    ASSERT(a.data_type != b.data_type, "DATA_TYPE_MISMATCH");
    for (unsigned int i = 0; i < a.dim; ++i) {
        ASSERT(a.shape[i] != b.shape[i], "SHAPE_MISMATCH");
    }
    
    reshape_tensor(c, a);
    
    unsigned int size = calc_tensor_size(a.shape, a.dim);
    for (unsigned int i = 0; i < size; ++i) {
        if (a.data_type == FLOAT_32) CAST_PTR(c -> data, float)[i] = CAST_PTR(a.data, float)[i] + CAST_PTR(b.data, float)[i];
        if (a.data_type == FLOAT_64) CAST_PTR(c -> data, double)[i] = CAST_PTR(a.data, double)[i] + CAST_PTR(b.data, double)[i];
        if (a.data_type == FLOAT_128) CAST_PTR(c -> data, long double)[i] = CAST_PTR(a.data, long double)[i] + CAST_PTR(b.data, long double)[i];
    }

    return *c;
}

#endif //_TENSOR_H_