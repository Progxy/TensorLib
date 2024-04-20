#ifndef _TENSOR_H_
#define _TENSOR_H_

#include <time.h>
#include <stdarg.h>

#define CAST_AND_OP(a, b, c, index, type, op) CAST_PTR(c -> data, type)[index] = CAST_PTR(a.data, type)[index] op CAST_PTR(b.data, type)[index]; 
#define DEALLOCATE_TENSORS(...) deallocate_tensors(sizeof((Tensor[]){__VA_ARGS__}) / sizeof(Tensor), __VA_ARGS__)
#define ASSERT(condition, err_msg) assert(condition, __LINE__, __FILE__, err_msg);
#define PRINT_TENSOR(tensor) print_tensor(tensor, #tensor)
#define SUBTRACT_TENSOR(c, a, b) op_tensor(c, a, b, SUBTRACTION)
#define SUM_TENSOR(c, a, b) op_tensor(c, a, b, SUM)
#define HADAMARD_PRODUCT(c, a, b) op_tensor(c, a, b, MULTIPLICATION)
#define DIVIDE_TENSOR(c, a, b) op_tensor(c, a, b, DIVISION)
#define ARR_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#define CAST_PTR(ptr, type) ((type*) (ptr))
#define NOT_USED(var) (void) var
#define MSG_MAX_LEN 512
#define TRUE 1
#define FALSE 0

typedef unsigned char bool;
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

bool is_valid_enum(unsigned char enum_value, unsigned char* enum_values, unsigned int enum_values_count) {
    for (unsigned int i = 0; i < enum_values_count; ++i) {
        if (enum_value == enum_values[i]) return TRUE;
    }
    return FALSE;
}

Tensor create_tensor(unsigned int* shape, unsigned int dim, DataType data_type) {
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

Tensor op_tensor(Tensor* c, Tensor a, Tensor b, OperatorFlag op_flag) {
    ASSERT(!is_valid_enum(op_flag, (unsigned char*) operators_flags, ARR_SIZE(operators_flags)), "INVALID_OPERATOR");
    ASSERT(a.dim != b.dim, "DIM_MISMATCH");
    ASSERT(a.data_type != b.data_type, "DATA_TYPE_MISMATCH");
    for (unsigned int i = 0; i < a.dim; ++i) {
        ASSERT(a.shape[i] != b.shape[i], "SHAPE_MISMATCH");
    }
    
    reshape_tensor(c, a);
    
    unsigned int size = calc_tensor_size(a.shape, a.dim);
    if (op_flag == SUM) {
        for (unsigned int i = 0; i < size; ++i) {
            if (a.data_type == FLOAT_32) CAST_AND_OP(a, b, c, i, float, +);
            if (a.data_type == FLOAT_64) CAST_AND_OP(a, b, c, i, double, +);
            if (a.data_type == FLOAT_128) CAST_AND_OP(a, b, c, i, long double, +);
        }
    } else if (op_flag == SUBTRACTION) {
        for (unsigned int i = 0; i < size; ++i) {
            if (a.data_type == FLOAT_32) CAST_AND_OP(a, b, c, i, float, -);
            if (a.data_type == FLOAT_64) CAST_AND_OP(a, b, c, i, double, -);
            if (a.data_type == FLOAT_128) CAST_AND_OP(a, b, c, i, long double, -);
        }
    } else if (op_flag == MULTIPLICATION) {
        for (unsigned int i = 0; i < size; ++i) {
            if (a.data_type == FLOAT_32) CAST_AND_OP(a, b, c, i, float, *);
            if (a.data_type == FLOAT_64) CAST_AND_OP(a, b, c, i, double, *);
            if (a.data_type == FLOAT_128) CAST_AND_OP(a, b, c, i, long double, *);
        }
    } else {
        for (unsigned int i = 0; i < size; ++i) {
            if (a.data_type == FLOAT_32) CAST_AND_OP(a, b, c, i, float, /);
            if (a.data_type == FLOAT_64) CAST_AND_OP(a, b, c, i, double, /);
            if (a.data_type == FLOAT_128) CAST_AND_OP(a, b, c, i, long double, /);
        }
    }

    return *c;
}

#endif //_TENSOR_H_