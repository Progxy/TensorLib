#include <stdio.h>
#include <stdlib.h>
#include "./include/autograd.h"

// Be able to derive those:
// Math: (1 + e^{-value})^{-1}
// Math: 0.5x(1 + {\tanh}[{\sqrt{2/\pi}}({x} + 0.044715{x}^{3})]

int main() {
    unsigned int shape[] = { 1 };    
    Tensor a = alloc_tensor(shape, ARR_SIZE(shape), FLOAT_32);
    Tensor b = alloc_tensor(shape, ARR_SIZE(shape), FLOAT_32);
    Tensor e = alloc_tensor(shape, ARR_SIZE(shape), FLOAT_32);
    alloc_grad_graph_node(a.data_type, &a);
    alloc_grad_graph_node(b.data_type, &b);
    float val = 1.0f;
    float val_e = -1.0f;
    fill_tensor(&val_e, a);
    fill_tensor(&val, b);
    fill_tensor(&val_e, e);
    Tensor c = empty_tensor(a.data_type);
    TENSOR_GRAPH_EXP(&c, a);
    Tensor pc = empty_tensor(a.data_type);
    TENSOR_GRAPH_POW(&pc, c, e);
    Tensor d = empty_tensor(a.data_type);
    TENSOR_GRAPH_SUM(&d, b, pc);
    Tensor f = empty_tensor(a.data_type);
    TENSOR_GRAPH_POW(&f, d, e);
    derive_node(a.grad_node);
    printf("f: %f, df/da: %f\n", CAST_PTR(f.data, float)[0], CAST_PTR(CAST_PTR(a.grad_node, GradNode) -> derived_value.data, float)[0]);
    deallocate_grad_graph(a.grad_node);
    DEALLOCATE_TENSORS(a, b, c, d, e, f);
    float res = 0.0f;
    sigmoid_func(&val_e, &res, FLOAT_32);
    printf("f: %f, df/da: %f\n", res, res * (1.0f - res)); 
    return 0;
}
