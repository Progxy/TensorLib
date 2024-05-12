#include <stdio.h>
#include <stdlib.h>
#include "./include/autograd.h"

int main() {
    unsigned int shape[] = { 1 };    
    Tensor a = alloc_graph_tensor(shape, ARR_SIZE(shape), FLOAT_32);
    Tensor b = alloc_graph_tensor(shape, ARR_SIZE(shape), FLOAT_32);
    float val_a = 3.0f;
    float val_b = 1.0f;
    fill_tensor(&val_a, a);
    fill_tensor(&val_b, b);
    Tensor c = empty_tensor(a.data_type);
    TENSOR_GRAPH_MULTIPLY(&c, a, b);
    derive_node(a.grad_node);
    printf("da/dc: %f\n", CAST_PTR(CAST_PTR(a.grad_node, GradNode) -> derived_value.data, float)[0]);
    printf("db/dc: %f\n", CAST_PTR(CAST_PTR(b.grad_node, GradNode) -> derived_value.data, float)[0]);
    return 0;
}