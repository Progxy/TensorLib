#ifndef _TENSOR_H_
#define _TENSOR_H_

#include <stdarg.h>
#include "./utils.h"

#define DEALLOCATE_TENSORS(...) deallocate_tensors(sizeof((Tensor[]){__VA_ARGS__}) / sizeof(Tensor), __VA_ARGS__)
#define DEALLOCATE_TEMP_TENSORS() alloc_temp_tensor(NULL, 0, FLOAT_32, TRUE)
#define PRINT_TENSOR(tensor) print_tensor(tensor, #tensor)
#define SUBTRACT_TENSOR(c, a, b) op_tensor(c, a, b, SUBTRACTION)
#define SUM_TENSOR(c, a, b) op_tensor(c, a, b, SUM)
#define MULTIPLY_TENSOR(c, a, b) op_tensor(c, a, b, MULTIPLICATION)
#define DIVIDE_TENSOR(c, a, b) op_tensor(c, a, b, DIVISION)
#define SCALAR_SUB_TENSOR(c, a, val) scalar_op_tensor(c, a, val, SUBTRACTION)
#define SCALAR_SUM_TENSOR(a, val) scalar_op_tensor(a, val, SUM)
#define SCALAR_MUL_TENSOR(a, val) scalar_op_tensor(a, val, MULTIPLICATION)
#define SCALAR_DIV_TENSOR(a, val) scalar_op_tensor(a, val, DIVISION)

typedef enum DataType { FLOAT_32 = sizeof(float), FLOAT_64 = sizeof(double), FLOAT_128 = sizeof(long double) } DataType;
typedef enum OperatorFlag { SUM, SUBTRACTION, MULTIPLICATION, DIVISION } OperatorFlag;

typedef struct Tensor {
    unsigned int* shape;
    unsigned int dim;
    void* data;
    DataType data_type;
} Tensor;

const unsigned char data_types[] = { FLOAT_32, FLOAT_64, FLOAT_128 };
const unsigned char operators_flags[] = { SUM, SUBTRACTION, MULTIPLICATION, DIVISION };

static unsigned int calc_tensor_size(unsigned int* shape, unsigned int dim) {
    unsigned int size = 1;
    for (unsigned int i = 0; i < dim; ++i) size *= shape[i];
    return size;
}

Tensor alloc_tensor(unsigned int* shape, unsigned int dim, DataType data_type) {
    ASSERT(!is_valid_enum(data_type, (unsigned char*) data_types, ARR_SIZE(data_types)), "INVALID_DATA_TYPE");
    ASSERT(!dim, "INVALID_DIM");
    Tensor tensor = { .shape = NULL, .dim = dim, .data_type = data_type, .data = NULL };
    tensor.shape = (unsigned int*) calloc(tensor.dim, sizeof(unsigned int));
    ASSERT(tensor.shape == NULL, "BAD_MEMORY");
    mem_copy(tensor.shape, shape, sizeof(unsigned int), tensor.dim);
    tensor.data = calloc(calc_tensor_size(shape, dim), tensor.data_type); 
    ASSERT(tensor.data == NULL, "BAD_MEMORY");
    return tensor;
}

Tensor alloc_temp_tensor(unsigned int* shape, unsigned int dim, DataType data_type, bool clean_cache_flag) {
    static Tensor* cache_tensor = NULL;
    static unsigned int cache_size = 0;

    if (clean_cache_flag) {
        for (unsigned int i = 0; i < cache_size; ++i) DEALLOCATE_TENSORS(cache_tensor[i]);
        free(cache_tensor);
        cache_tensor = NULL;
        cache_size = 0;
        return (Tensor) {};
    } else if (cache_tensor == NULL) cache_tensor = (Tensor*) calloc(1, sizeof(Tensor));
    else cache_tensor = (Tensor*) realloc(cache_tensor, sizeof(Tensor) * (cache_size + 1));

    Tensor temp = alloc_tensor(shape, dim, data_type);
    cache_tensor[cache_size++] = temp;
    return temp;
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

void print_tensor(Tensor tensor, char* tensor_name) {
    const unsigned int size = calc_tensor_size(tensor.shape, tensor.dim);
    printf("DEBUG_INFO: Tensor '%s' has shape: [ ", tensor_name);
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

void reshape_tensor(Tensor* dest, unsigned int* shape, unsigned int dim, DataType data_type) {
    dest -> shape = (unsigned int*) realloc(dest -> shape, sizeof(unsigned int) * dim);
    ASSERT(dest -> shape == NULL, "BAD_MEMORY");
    mem_copy(dest -> shape, shape, sizeof(unsigned int), dim);
    dest -> dim = dim;
    dest -> data_type = data_type;
    free(dest -> data);
    dest -> data = calloc(calc_tensor_size(dest -> shape, dest -> dim), dest -> data_type);
    ASSERT(dest -> data == NULL, "BAD_MEMORY");
    return;
}

void copy_tensor(Tensor* dest, Tensor src) {
    reshape_tensor(dest, src.shape, src.dim, src.data_type);
    unsigned int size = calc_tensor_size(src.shape, src.dim);
    mem_copy(dest -> data, src.data, size, src.data_type);
    return;
}

Tensor op_tensor(Tensor* c, Tensor a, Tensor b, OperatorFlag op_flag) {
    ASSERT(!is_valid_enum(op_flag, (unsigned char*) operators_flags, ARR_SIZE(operators_flags)), "INVALID_OPERATOR");
    ASSERT(a.dim != b.dim, "DIM_MISMATCH");
    ASSERT(a.data_type != b.data_type, "DATA_TYPE_MISMATCH");
    for (unsigned int i = 0; i < a.dim; ++i) {
        ASSERT(a.shape[i] != b.shape[i], "SHAPE_MISMATCH");
    }
    
    Tensor temp = alloc_tensor(a.shape, a.dim, a.data_type);
    
    unsigned int size = calc_tensor_size(a.shape, a.dim);
    if (op_flag == SUM) {
        for (unsigned int i = 0; i < size; ++i) {
            if (a.data_type == FLOAT_32) CAST_AND_OP(a, b, temp, i, float, +);
            if (a.data_type == FLOAT_64) CAST_AND_OP(a, b, temp, i, double, +);
            if (a.data_type == FLOAT_128) CAST_AND_OP(a, b, temp, i, long double, +);
        }
    } else if (op_flag == SUBTRACTION) {
        for (unsigned int i = 0; i < size; ++i) {
            if (a.data_type == FLOAT_32) CAST_AND_OP(a, b, temp, i, float, -);
            if (a.data_type == FLOAT_64) CAST_AND_OP(a, b, temp, i, double, -);
            if (a.data_type == FLOAT_128) CAST_AND_OP(a, b, temp, i, long double, -);
        }
    } else if (op_flag == MULTIPLICATION) {
        for (unsigned int i = 0; i < size; ++i) {
            if (a.data_type == FLOAT_32) CAST_AND_OP(a, b, temp, i, float, *);
            if (a.data_type == FLOAT_64) CAST_AND_OP(a, b, temp, i, double, *);
            if (a.data_type == FLOAT_128) CAST_AND_OP(a, b, temp, i, long double, *);
        }
    } else {
        for (unsigned int i = 0; i < size; ++i) {
            if (a.data_type == FLOAT_32) CAST_AND_OP(a, b, temp, i, float, /);
            if (a.data_type == FLOAT_64) CAST_AND_OP(a, b, temp, i, double, /);
            if (a.data_type == FLOAT_128) CAST_AND_OP(a, b, temp, i, long double, /);
        }
    }

    copy_tensor(c, temp);
    DEALLOCATE_TENSORS(temp);

    return *c;
}

Tensor cross_product_tensor(Tensor* c, Tensor a, Tensor b) {
    ASSERT(a.data_type != b.data_type, "DATA_TYPE_MISMATCH");

    unsigned int* new_shape = (unsigned int*) calloc(a.dim + b.dim, sizeof(unsigned int));
    mem_copy(new_shape, a.shape, a.dim, sizeof(unsigned int));
    mem_copy(new_shape + a.dim, b.shape, b.dim, sizeof(unsigned int));
    Tensor temp = alloc_tensor(new_shape, a.dim + b.dim, a.data_type);
    free(new_shape);

    unsigned int a_size = calc_tensor_size(a.shape, a.dim);
    unsigned int b_size = calc_tensor_size(b.shape, b.dim);
    for (unsigned int i = 0; i < a_size; ++i) {
        for (unsigned int j = 0; j < b_size; ++j) {
            if (a.data_type == FLOAT_32) CAST_PTR(temp.data, float)[i * b_size + j] = CAST_PTR(a.data, float)[i] * CAST_PTR(b.data, float)[j];
            else if (a.data_type == FLOAT_64) CAST_PTR(temp.data, double)[i * b_size + j] = CAST_PTR(a.data, double)[i] * CAST_PTR(b.data, double)[j];
            else if (a.data_type == FLOAT_128) CAST_PTR(temp.data, long double)[i * b_size + j] = CAST_PTR(a.data, long double)[i] * CAST_PTR(b.data, long double)[j];
        }
    }

    copy_tensor(c, temp);
    DEALLOCATE_TENSORS(temp);

    return *c;
}

Tensor scalar_op_tensor(Tensor* tensor, void* scalar, OperatorFlag op_flag) {
    ASSERT(!is_valid_enum(op_flag, (unsigned char*) operators_flags, ARR_SIZE(operators_flags)), "INVALID_OPERATOR");
    Tensor scalar_tensor = alloc_tensor(tensor -> shape, tensor -> dim, tensor -> data_type);
    fill_tensor(scalar, scalar_tensor);
    op_tensor(tensor, *tensor, scalar_tensor, op_flag);
    DEALLOCATE_TENSORS(scalar_tensor);
    return *tensor;
}

#endif //_TENSOR_H_