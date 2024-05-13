#include <stdio.h>
#include <stdlib.h>
#include "./include/autograd.h"

// Be able to derive those:
// Math: (1 + e^{-value})^{-1}

void test();

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
    TENSOR_GRAPH_EXP(&c, a, a.data_type);
    Tensor pc = empty_tensor(a.data_type);
    TENSOR_GRAPH_POW(&pc, c, &val_e, c.data_type);
    Tensor d = empty_tensor(a.data_type);
    TENSOR_GRAPH_SUM(&d, b, pc);
    Tensor f = empty_tensor(a.data_type);
    TENSOR_GRAPH_POW(&f, d, &val_e, d.data_type);
    derive_node(a.grad_node);
    printf("f: %f, df/da: %f\n", CAST_PTR(f.data, float)[0], CAST_PTR(CAST_PTR(a.grad_node, GradNode) -> derived_value.data, float)[0]);
    deallocate_grad_graph(a.grad_node);
    DEALLOCATE_TENSORS(a, b, c, d, e, f);
    float res = 0.0f;
    sigmoid_func(&val_e, &res, FLOAT_32);
    printf("f: %f, df/da: %f\n", res, res * (1.0f - res)); 
    test();
    return 0;
}

// Math: 0.5x(1 + {\tanh}[{\sqrt{2/\pi}}({x} + 0.044715{x}^{3})]
// Math: 2x + 2x^2
void test() {
    unsigned int shape[] = {1};
    float val = 1.0f;
    Tensor x, temp;
    alloc_tensor_grad_graph(x, shape, ARR_SIZE(shape), FLOAT_32);
    alloc_tensor_grad_graph(temp, shape, ARR_SIZE(shape), FLOAT_32);
    fill_tensor(&val, x);
    val = 0.044715f;
    fill_tensor(&val, temp);
    val = 3.0f;
    Tensor a = empty_tensor(x.data_type);
    TENSOR_GRAPH_POW(&a, x, &val, x.data_type); // Math: x^3
    Tensor b = empty_tensor(x.data_type);
    TENSOR_GRAPH_MUL(&b, a, temp); // Math: 0.044715x^3
    Tensor c = empty_tensor(x.data_type);
    TENSOR_GRAPH_SUM(&c, x, b); // Math: x + 0.044715x^3
    derive_node(x.grad_node);
    printf("expr_val: %f, expr_derivative_val: %f\n", CAST_PTR(c.data, float)[0], CAST_PTR(CAST_PTR(x.grad_node, GradNode) -> derived_value.data, float)[0]);
    DEALLOCATE_TENSORS(x, temp, a, b, c);
    return;
}