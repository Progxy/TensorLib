#ifndef _UTILS_H_
#define _UTILS_H_

#include <time.h>
#include <stdarg.h>
#define __USE_MISC
#include <math.h>
#include "./types.h"

#define CAST_AND_OP_INDEX(a, b, c, index, type, op) CAST_PTR(c.data, type)[index] = CAST_AND_OP(CAST_PTR_AT_INDEX(a.data, type, index), CAST_PTR_AT_INDEX(b.data, type, index), type, op) 
#define CAST_AND_OP(a, b, type, op) *CAST_PTR(a, type) op *CAST_PTR(b, type)
#define CAST_PTR_AT_INDEX(a, type, index) &(CAST_PTR(a, type)[index])
#define ASSIGN(val, new_val, data_type) assign_data_type(val, (long double) new_val, data_type)
#define SCALAR_MUL(res, a, b, data_type) scalar_op(res, a, b, data_type, MULTIPLICATION)
#define SCALAR_SUB(res, a, b, data_type) scalar_op(res, a, b, data_type, SUBTRACTION)
#define SCALAR_DIV(res, a, b, data_type) scalar_op(res, a, b, data_type, DIVISION)
#define SCALAR_SUM(res, a, b, data_type) scalar_op(res, a, b, data_type, SUM)
#define SCALAR_POW(res, a, b, data_type) scalar_op(res, a, b, data_type, POW)
#define IS_GREATER_OR_EQUAL(a, b, data_type) comparison_op(a, b, data_type, GREATER_OR_EQUAL)
#define IS_LESS_OR_EQUAL(a, b, data_type) comparison_op(a, b, data_type, LESS_OR_EQUAL)
#define IS_GREATER(a, b, data_type) comparison_op(a, b, data_type, GREATER)
#define IS_EQUAL(a, b, data_type) comparison_op(a, b, data_type, EQUAL)
#define IS_LESS(a, b, data_type) comparison_op(a, b, data_type, LESS)
#define ASSERT(condition, err_msg) assert(condition, __LINE__, __FILE__, err_msg);
#define ARR_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#define CAST_PTR(ptr, type) ((type*) (ptr))
#define NOT_USED(var) (void) var
#define MAX(a, b) (a >= b ? a : b)
#define MIN(a, b) (a <= b ? a : b)

void assert(bool condition, unsigned int line, char* file, char* err_msg) {
    if (condition) {
        printf("ERROR: Assert failed in file: %s:%u, with error: %s.\n", file, line, err_msg);
        abort();
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

bool is_valid_enum(unsigned char enum_value, unsigned char* enum_values, unsigned int enum_values_count) {
    for (unsigned int i = 0; i < enum_values_count; ++i) {
        if (enum_value == enum_values[i]) return TRUE;
    }
    return FALSE;
}

void init_seed() {
    srand(time(NULL));
    return;
}

void* assign_data_type(void* val, long double new_val, DataType data_type) {
    if (data_type == FLOAT_32) *CAST_PTR(val, float) = (float) new_val;
    else if (data_type == FLOAT_64) *CAST_PTR(val, double) = (double) new_val;
    else if (data_type == FLOAT_128) *CAST_PTR(val, long double) = new_val;
    return val;
}

bool comparison_op(void* a, void* b, DataType data_type, ComparisonFlag comparison) {
    ASSERT(!is_valid_enum(comparison, (unsigned char*) comparison_flags, ARR_SIZE(comparison_flags)), "INVALID_COMPARISON_FLAG");
    switch (comparison) {
        case EQUAL: {
            if (data_type == FLOAT_32) return CAST_AND_OP(a, b, float, ==);
            else if (data_type == FLOAT_64) return CAST_AND_OP(a, b, double, ==);
            else if (data_type == FLOAT_128) return CAST_AND_OP(a, b, long double, ==);
            return FALSE;
        }
        
        case LESS: {
            if (data_type == FLOAT_32) return CAST_AND_OP(a, b, float, <);
            else if (data_type == FLOAT_64) return CAST_AND_OP(a, b, double, <);
            else if (data_type == FLOAT_128) return CAST_AND_OP(a, b, long double, <);
            return FALSE;
        }

        case LESS_OR_EQUAL: {
            if (data_type == FLOAT_32) return CAST_AND_OP(a, b, float, <=);
            else if (data_type == FLOAT_64) return CAST_AND_OP(a, b, double, <=);
            else if (data_type == FLOAT_128) return CAST_AND_OP(a, b, long double, <=);
            return FALSE;
        }

        case GREATER: {
            if (data_type == FLOAT_32) return CAST_AND_OP(a, b, float, >);
            else if (data_type == FLOAT_64) return CAST_AND_OP(a, b, double, >);
            else if (data_type == FLOAT_128) return CAST_AND_OP(a, b, long double, >);
            return FALSE;
        }

        case GREATER_OR_EQUAL: {
            if (data_type == FLOAT_32) return CAST_AND_OP(a, b, float, >=);
            else if (data_type == FLOAT_64) return CAST_AND_OP(a, b, double, >=);
            else if (data_type == FLOAT_128) return CAST_AND_OP(a, b, long double, >=);
            return FALSE;
        }
    }
    return FALSE;
}

void* scalar_op(void* res, void* a, void* b, DataType data_type, OperatorFlag operation) {
    ASSERT(!is_valid_enum(operation, (unsigned char*) operators_flags, ARR_SIZE(operators_flags)), "INVALID_OPERATOR_FLAG");
    switch(operation) {
        case SUM: {
            if (data_type == FLOAT_32) *CAST_PTR(res, float) = CAST_AND_OP(a, b, float, +);
            else if (data_type == FLOAT_64) *CAST_PTR(res, double) = CAST_AND_OP(a, b, double, +);
            else if (data_type == FLOAT_128) *CAST_PTR(res, long double) = CAST_AND_OP(a, b, long double, +);
            break;
        }

        case SUBTRACTION: {
            if (data_type == FLOAT_32) *CAST_PTR(res, float) = CAST_AND_OP(a, b, float, -);
            else if (data_type == FLOAT_64) *CAST_PTR(res, double) = CAST_AND_OP(a, b, double, -);
            else if (data_type == FLOAT_128) *CAST_PTR(res, long double) = CAST_AND_OP(a, b, long double, -);
            break;
        }

        case MULTIPLICATION: {
            if (data_type == FLOAT_32) *CAST_PTR(res, float) = CAST_AND_OP(a, b, float, *);
            else if (data_type == FLOAT_64) *CAST_PTR(res, double) = CAST_AND_OP(a, b, double, *);
            else if (data_type == FLOAT_128) *CAST_PTR(res, long double) = CAST_AND_OP(a, b, long double, *);
            break;
        }

        case DIVISION: {
            if (data_type == FLOAT_32) *CAST_PTR(res, float) = CAST_AND_OP(a, b, float, /);
            else if (data_type == FLOAT_64) *CAST_PTR(res, double) = CAST_AND_OP(a, b, double, /);
            else if (data_type == FLOAT_128) *CAST_PTR(res, long double) = CAST_AND_OP(a, b, long double, /);
            break;
        }
        
        case POW: {
            if (data_type == FLOAT_32) *CAST_PTR(res, float) = powf(*CAST_PTR(a, float), *CAST_PTR(b, float));
            else if (data_type == FLOAT_64) *CAST_PTR(res, double) = pow(*CAST_PTR(a, double), *CAST_PTR(b, double));
            else if (data_type == FLOAT_128) *CAST_PTR(res, long double) = powl(*CAST_PTR(a, long double), *CAST_PTR(b, long double));
            break;
        }

        case EXP: {
            if (data_type == FLOAT_32) *CAST_PTR(res, float) = expf(*CAST_PTR(a, float));
            else if (data_type == FLOAT_64) *CAST_PTR(res, double) = exp(*CAST_PTR(a, double));
            else if (data_type == FLOAT_128) *CAST_PTR(res, long double) = expl(*CAST_PTR(a, long double));
            break;
        }        
        
        case TANH: {
            if (data_type == FLOAT_32) *CAST_PTR(res, float) = tanhf(*CAST_PTR(a, float));
            else if (data_type == FLOAT_64) *CAST_PTR(res, double) = tanh(*CAST_PTR(a, double));
            else if (data_type == FLOAT_128) *CAST_PTR(res, long double) = tanhl(*CAST_PTR(a, long double));
            break;
        }
    }

    return res;
}

void* sigmoid_func(void* value, void* result, DataType data_type) {
    // Math: \frac{1}{1 + e^{-value}}
    if (data_type == FLOAT_32) *CAST_PTR(result, float) = (1.0f / (1.0f + expf(*CAST_PTR(value, float) * -1)));
    else if (data_type == FLOAT_64) *CAST_PTR(result, double) = (1.0f / (1.0f + exp(*CAST_PTR(value, double) * -1)));
    else if (data_type == FLOAT_128) *CAST_PTR(result, long double) = (1.0f / (1.0f + expl(*CAST_PTR(value, long double) * -1)));
    return result;
}

#endif //_UTILS_H_