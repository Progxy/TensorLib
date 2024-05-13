#include <stdio.h>
#include <stdlib.h>
#include "./include/autograd.h"

// Be able to derive those:
// Math: \frac{1}{1 + e^{-value}}
// Math: 0.5x(1 + {\tanh}[{\sqrt{2/\pi}}({x} + 0.044715{x}^{3})]

int main() {
    unsigned int shape[] = { 1 };    
    Tensor a = alloc_tensor(shape, ARR_SIZE(shape), FLOAT_32);
    Tensor b = alloc_tensor(shape, ARR_SIZE(shape), FLOAT_32);
    alloc_grad_graph_node(a.data_type, &a);
    float val_a = 2.0f;
    float val_b = -2.0f;
    fill_tensor(&val_a, a);
    fill_tensor(&val_b, b);
    Tensor c = empty_tensor(a.data_type);
    TENSOR_GRAPH_POW(&c, a, b);
    printf("c: %f\n", CAST_PTR(c.data, float)[0]);
    DEALLOCATE_TENSORS(b);
    derive_node(a.grad_node);
    printf("dc/da: %f\n", CAST_PTR(CAST_PTR(a.grad_node, GradNode) -> derived_value.data, float)[0]);
    deallocate_grad_graph(a.grad_node);
    DEALLOCATE_TENSORS(a);
    return 0;
}
