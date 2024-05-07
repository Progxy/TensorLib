#ifndef _TYPES_H_
#define _TYPES_H_

typedef unsigned char bool;

typedef enum DataType { FLOAT_32 = sizeof(float), FLOAT_64 = sizeof(double), FLOAT_128 = sizeof(long double) } DataType;
typedef enum OperatorFlag { SUM, SUBTRACTION, MULTIPLICATION, DIVISION } OperatorFlag;

const unsigned char data_types[] = { FLOAT_32, FLOAT_64, FLOAT_128 };
const unsigned char operators_flags[] = { SUM, SUBTRACTION, MULTIPLICATION, DIVISION };

typedef struct Tensor {
    unsigned int* shape;
    unsigned int rank;
    void* data;
    DataType data_type;
    GradNode* grad_node;
} Tensor;

typedef struct GradNode {
    Tensor derived_value;
    OperatorFlag operation;
    struct GradNode** children;
    unsigned int children_count;
    struct GradNode** parents;
    unsigned int parents_count;
} GradNode;

#endif //_TYPES_H_