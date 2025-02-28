#ifndef _TYPES_H_
#define _TYPES_H_

#define FALSE 0
#define TRUE 1

typedef unsigned char bool;

typedef enum DataType { FLOAT_32 = sizeof(float), FLOAT_64 = sizeof(double), FLOAT_128 = sizeof(long double) } DataType;
typedef enum OperatorFlag { NO_OP = -1, SUM, SUBTRACTION, MULTIPLICATION, DIVISION, POW, EXP, TANH, DOT, SQRT, LOG, MAX, MIN, ABS, CONJUGATE, NORM, SOFTMAX } OperatorFlag;
typedef enum ComparisonFlag { EQUAL, LESS, LESS_OR_EQUAL, GREATER, GREATER_OR_EQUAL, NEGATIVE, POSITIVE } ComparisonFlag;

const unsigned char data_types[] = { FLOAT_32, FLOAT_64, FLOAT_128 };
const unsigned char operators_flags[] = { SUM, SUBTRACTION, MULTIPLICATION, DIVISION, POW, EXP, TANH, DOT, SQRT, LOG, MAX, MIN, ABS, CONJUGATE, NORM, SOFTMAX };
const unsigned char comparison_flags[] = { EQUAL, LESS, LESS_OR_EQUAL, GREATER, GREATER_OR_EQUAL, NEGATIVE, POSITIVE };

typedef struct Tensor {
    unsigned int* shape;
    unsigned int rank;
    void* data;
    DataType data_type;
    void* grad_node;
} Tensor;

typedef struct GradNode {
    Tensor derived_value;
    Tensor* value;
    bool is_value_updated;
    OperatorFlag operation;
    struct GradNode** children;
    unsigned int children_count;
    struct GradNode** parents;
    unsigned int parents_count;
    void* exp;
} GradNode;

typedef struct Value {
    void* data;
    DataType data_type;
} Value;

#endif //_TYPES_H_
