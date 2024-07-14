#ifndef _UTILS_H_
#define _UTILS_H_

#include <time.h>
#include <stdarg.h>
#define __USE_MISC
#include <math.h>
#include "./types.h"

#define CAST_AND_OP_INDEX(a, b, c, index, data_type, op) scalar_op(CAST_PTR_AT_INDEX(c, index, data_type), CAST_PTR_AT_INDEX(a, index, data_type), CAST_PTR_AT_INDEX(b, index, data_type), data_type, op)
#define CAST_AND_SINGLE_OP_INDEX(a, c, index, data_type, op) scalar_op(CAST_PTR_AT_INDEX(c, index, data_type), CAST_PTR_AT_INDEX(a, index, data_type), NULL, data_type, op)
#define DEALLOCATE_PTRS(...) deallocate_ptrs(sizeof((void*[]){__VA_ARGS__}) / sizeof(void*), __VA_ARGS__)
#define CAST_AND_OP(a, b, type, op) *CAST_PTR(a, type) op *CAST_PTR(b, type)
#define CAST_PTR_AT_INDEX(a, index, type) (CAST_PTR(a, unsigned char) + (type * index))
#define ASSIGN(val, new_val, data_type) assign_data_type(val, (long double) new_val, data_type)
#define SCALAR_CONJUGATE(res, a, data_type) scalar_op(res, a, NULL, data_type, CONJUGATE)
#define SCALAR_MUL(res, a, b, data_type) scalar_op(res, a, b, data_type, MULTIPLICATION)
#define SCALAR_SUB(res, a, b, data_type) scalar_op(res, a, b, data_type, SUBTRACTION)
#define SCALAR_DIV(res, a, b, data_type) scalar_op(res, a, b, data_type, DIVISION)
#define SCALAR_SQRT(res, a, data_type) scalar_op(res, a, NULL, data_type, SQRT)
#define SCALAR_TANH(res, a, data_type) scalar_op(res, a, NULL, data_type, TANH)
#define SCALAR_SUM(res, a, b, data_type) scalar_op(res, a, b, data_type, SUM)
#define SCALAR_POW(res, a, b, data_type) scalar_op(res, a, b, data_type, POW)
#define SCALAR_EXP(res, a, data_type) scalar_op(res, a, NULL, data_type, EXP)
#define SCALAR_LOG(res, a, data_type) scalar_op(res, a, NULL, data_type, LOG)
#define SCALAR_MAX(res, a, b, data_type) scalar_op(res, a, b, data_type, MAX)
#define SCALAR_MIN(res, a, b, data_type) scalar_op(res, a, b, data_type, MIN)
#define IS_GREATER_OR_EQUAL(a, b, data_type) comparison_op(a, b, data_type, GREATER_OR_EQUAL)
#define IS_LESS_OR_EQUAL(a, b, data_type) comparison_op(a, b, data_type, LESS_OR_EQUAL)
#define IS_GREATER(a, b, data_type) comparison_op(a, b, data_type, GREATER)
#define IS_EQUAL(a, b, data_type) comparison_op(a, b, data_type, EQUAL)
#define IS_LESS(a, b, data_type) comparison_op(a, b, data_type, LESS)
#define ASSERT(condition, err_msg) assert(condition, #condition, __LINE__, __FILE__, err_msg);
#define ARR_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#define CAST_PTR(ptr, type) ((type*) (ptr))
#define NOT_USED(var) (void) var
#define MAX(a, b) (a >= b ? a : b)
#define MIN(a, b) (a <= b ? a : b)

void assert(bool condition, char* condition_str, unsigned int line, char* file, char* err_msg) {
    if (condition) {
        printf("ERROR: Assert condition: '%s' failed in file: %s:%u, with error: %s.\n", condition_str, file, line, err_msg);
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

        case DOT:
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

        case SQRT: {
            if (data_type == FLOAT_32) *CAST_PTR(res, float) = sqrtf(*CAST_PTR(a, float));
            else if (data_type == FLOAT_64) *CAST_PTR(res, double) = sqrt(*CAST_PTR(a, double));
            else if (data_type == FLOAT_128) *CAST_PTR(res, long double) = sqrtl(*CAST_PTR(a, long double));
            break;
        }

        case LOG: {
            if (data_type == FLOAT_32) *CAST_PTR(res, float) = logf(*CAST_PTR(a, float));
            else if (data_type == FLOAT_64) *CAST_PTR(res, double) = log(*CAST_PTR(a, double));
            else if (data_type == FLOAT_128) *CAST_PTR(res, long double) = logl(*CAST_PTR(a, long double));
            break;
        }

        case MAX: {
            if (data_type == FLOAT_32) *CAST_PTR(res, float) = MAX(*CAST_PTR(a, float), *CAST_PTR(b, float));
            else if (data_type == FLOAT_64) *CAST_PTR(res, double) = MAX(*CAST_PTR(a, double), *CAST_PTR(b, double));
            else if (data_type == FLOAT_128) *CAST_PTR(res, long double) = MAX(*CAST_PTR(a, long double), *CAST_PTR(b, long double));
            break;
        }        
        
        case MIN: {
            if (data_type == FLOAT_32) *CAST_PTR(res, float) = MIN(*CAST_PTR(a, float), *CAST_PTR(b, float));
            else if (data_type == FLOAT_64) *CAST_PTR(res, double) = MIN(*CAST_PTR(a, double), *CAST_PTR(b, double));
            else if (data_type == FLOAT_128) *CAST_PTR(res, long double) = MIN(*CAST_PTR(a, long double), *CAST_PTR(b, long double));
            break;
        }

        case CONJUGATE: {
            if (data_type == FLOAT_32) *CAST_PTR(res, float) = -(*CAST_PTR(a, float));
            else if (data_type == FLOAT_64) *CAST_PTR(res, double) = -(*CAST_PTR(a, double));
            else if (data_type == FLOAT_128) *CAST_PTR(res, long double) = -(*CAST_PTR(a, long double));
            break;
        }
    }

    return res;
}

void deallocate_ptrs(int len, ...) {
    va_list args;
    va_start(args, len);
    for (int i = 0; i < len; ++i) {
        void* ptr = va_arg(args, void*);
        free(ptr);
    }
    va_end(args);
    return;
}

void* sigmoid_func(void* value, void* result, DataType data_type) {
    // Math: \frac{1}{1 + e^{-value}}
    if (data_type == FLOAT_32) *CAST_PTR(result, float) = (1.0f / (1.0f + expf(*CAST_PTR(value, float) * -1)));
    else if (data_type == FLOAT_64) *CAST_PTR(result, double) = (1.0f / (1.0f + exp(*CAST_PTR(value, double) * -1)));
    else if (data_type == FLOAT_128) *CAST_PTR(result, long double) = (1.0f / (1.0f + expl(*CAST_PTR(value, long double) * -1)));
    return result;
}

void* normal_func(void* res, void* value, void* variance, void* mean, DataType data_type) {
    // Math: (2\pi\sigma^2)^{-{1/2}}\exp(-\frac{(x-\mu)^2}{2\sigma^2})
    if (data_type == FLOAT_32) *CAST_PTR(res, float) = powf(2.0f * (float) M_PI * (*CAST_PTR(variance, float)), -0.5f) * expf(-(powf(*CAST_PTR(value, float) - *CAST_PTR(mean, float), 2.0f) * (2.0f * (*CAST_PTR(variance, float)))));
    else if (data_type == FLOAT_64) *CAST_PTR(res, double) = pow(2.0 * (double) M_PI * (*CAST_PTR(variance, double)), -0.5) * exp(-(pow(*CAST_PTR(value, double) - *CAST_PTR(mean, double), 2.0) * (2.0 * (*CAST_PTR(variance, double)))));
    else if (data_type == FLOAT_128) *CAST_PTR(res, long double) = powl(2.0L * (long double) M_PI * (*CAST_PTR(variance, long double)), -0.5L) * expl(-(powl(*CAST_PTR(value, long double) - *CAST_PTR(mean, long double), 2.0L) * (2.0L * (*CAST_PTR(variance, long double)))));
    return res;
}

#endif //_UTILS_H_