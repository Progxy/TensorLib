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
    test_gelu();
    return 0;
}

// Math: 0.5x(1 + {\tanh}[{\sqrt{2/\pi}}({x} + 0.044715{x}^{3})])
void test_gelu() {
    unsigned int shape[] = {1};
    Tensor x, x1, x2, x3, x4;
    alloc_tensor_grad_graph(x, shape, ARR_SIZE(shape), FLOAT_32);
    alloc_tensor_grad_graph(x1, shape, ARR_SIZE(shape), FLOAT_32);
    alloc_tensor_grad_graph(x2, shape, ARR_SIZE(shape), FLOAT_32);
    alloc_tensor_grad_graph(x3, shape, ARR_SIZE(shape), FLOAT_32);
    alloc_tensor_grad_graph(x4, shape, ARR_SIZE(shape), FLOAT_32);
    float val = 1.0f;
    fill_tensor(&val, x);
    val = 0.044715f;
    fill_tensor(&val, x1);
    val = sqrtf(2.0f / M_PI);
    fill_tensor(&val, x2);    
    val = 1.0f;
    fill_tensor(&val, x3);    
    val = 0.5f;
    fill_tensor(&val, x4);
    val = 3.0f;
    Tensor a = empty_tensor(x.data_type);
    TENSOR_GRAPH_POW(&a, x, &val, x.data_type); // Math: x^3
    Tensor b = empty_tensor(x.data_type);
    TENSOR_GRAPH_MUL(&b, a, x1); // Math: 0.044715x^3
    Tensor c = empty_tensor(x.data_type);
    TENSOR_GRAPH_SUM(&c, x, b); // Math: x + 0.044715x^3
    Tensor d = empty_tensor(x.data_type);
    TENSOR_GRAPH_MUL(&d, c, x2); // Math: {\sqrt{2/\pi}}({x} + 0.044715{x}^{3})
    Tensor e = empty_tensor(x.data_type);
    TENSOR_GRAPH_TANH(&e, d, x.data_type); // Math: {\tanh}[{\sqrt{2/\pi}}({x} + 0.044715{x}^{3})]
    Tensor f = empty_tensor(x.data_type);
    TENSOR_GRAPH_SUM(&f, e, x3); // Math: (1 + {\tanh}[{\sqrt{2/\pi}}({x} + 0.044715{x}^{3})])
    Tensor g = empty_tensor(x.data_type);
    TENSOR_GRAPH_MUL(&g, x, x4); // Math: 0.5x
    Tensor h = empty_tensor(x.data_type);
    TENSOR_GRAPH_MUL(&h, g, f); // Math: 0.5x(1 + {\tanh}[{\sqrt{2/\pi}}({x} + 0.044715{x}^{3})])
    derive_node(x.grad_node);
    printf("expr_val: %f, expr_derivative_val: %f\n", CAST_PTR(h.data, float)[0], CAST_PTR(CAST_PTR(x.grad_node, GradNode) -> derived_value.data, float)[0]);
    DEALLOCATE_GRAD_GRAPHS(x.grad_node, x1.grad_node, x2.grad_node, x3.grad_node, x4.grad_node);
    DEALLOCATE_TENSORS(x, x1, x2, x3, x4, a, b, c, d, e, f, g, h);
    return;
}