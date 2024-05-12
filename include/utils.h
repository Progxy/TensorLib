#ifndef _UTILS_H_
#define _UTILS_H_

#include <time.h>
#include "./types.h"

#define CAST_AND_OP(a, b, c, index, type, op) CAST_PTR(c.data, type)[index] = CAST_PTR(a.data, type)[index] op CAST_PTR(b.data, type)[index]; 
#define ASSIGN(val, new_val, data_type) assign_data_type(val, (long double) new_val, data_type)
#define ASSERT(condition, err_msg) assert(condition, __LINE__, __FILE__, err_msg);
#define ARR_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#define CAST_PTR(ptr, type) ((type*) (ptr))
#define NOT_USED(var) (void) var
#define MAX(a, b) (a >= b ? a : b)
#define MIN(a, b) (a <= b ? a : b)
#define FALSE 0
#define TRUE 1

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

#endif //_UTILS_H_