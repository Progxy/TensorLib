#include <stdio.h>
#include <stdlib.h>
#include "./include/autograd.h"

void test();
void div_test();

int main() {
    unsigned int shape[] = {1};
    float val = 1.0f;

    Tensor x, x1, x2, x3, x4;
    alloc_tensor_grad_graph_filled(x, shape, ARR_SIZE(shape), FLOAT_32, &val);
    alloc_tensor_grad_graph_filled(x1, shape, ARR_SIZE(shape), FLOAT_32, (val = 0.044715f, &val));
    alloc_tensor_grad_graph_filled(x2, shape, ARR_SIZE(shape), FLOAT_32, (val = sqrtf(2.0f / M_PI), &val));
    alloc_tensor_grad_graph_filled(x3, shape, ARR_SIZE(shape), FLOAT_32, (val = 1.0f, &val));
    alloc_tensor_grad_graph_filled(x4, shape, ARR_SIZE(shape), FLOAT_32, (val = 0.5f, &val));
    
    Tensor a, b, c, d, e, f, g, h;
    EMPTY_TENSORS(x.data_type, &a, &b, &c, &d, &e, &f, &g, &h);

    // Math: 0.5x(1 + {\tanh}[{\sqrt{2/\pi}}({x} + 0.044715{x}^{3})])
    TENSOR_GRAPH_MUL(&d, x2, *TENSOR_GRAPH_SUM(&c, x, *TENSOR_GRAPH_MUL(&b, x1, *TENSOR_GRAPH_POW(&a, x, (val = 3.0f, &val), x.data_type))));
    TENSOR_GRAPH_MUL(&h, *TENSOR_GRAPH_MUL(&g, x, x4), *TENSOR_GRAPH_SUM(&f, x3, *TENSOR_GRAPH_TANH(&e, d, x.data_type)));
    
    derive_node(x.grad_node);
    printf("expr_val: %f, expr_derivative_val: %f\n", CAST_PTR(h.data, float)[0], CAST_PTR(CAST_PTR(x.grad_node, GradNode) -> derived_value.data, float)[0]);
    DEALLOCATE_GRAD_GRAPHS(x.grad_node, x1.grad_node, x2.grad_node, x3.grad_node, x4.grad_node);
    test();
    div_test();
    return 0;
}

void test() {
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
    float res = 0.0f;
    sigmoid_func(&val_e, &res, FLOAT_32);
    printf("f: %f, df/da: %f\n", res, res * (1.0f - res)); 
    return;
}

void div_test() {
    unsigned int shape[] = {1};
    float val = 1.0f;
    Tensor x, x1;
    alloc_tensor_grad_graph_filled(x, shape, ARR_SIZE(shape), FLOAT_32, &val);
    alloc_tensor_grad_graph_filled(x1, shape, ARR_SIZE(shape), FLOAT_32, (val = 3.0f, &val));

    Tensor y = empty_tensor(x.data_type);
    TENSOR_GRAPH_DIV(&y, x1, x);
    printf("value: %f\n", CAST_PTR(y.data, float)[0]);
    derive_node(x.grad_node);
    printf("dy/dx: %f\n", CAST_PTR(CAST_PTR(x.grad_node, GradNode) -> derived_value.data, float)[0]);
    DEALLOCATE_GRAD_GRAPHS(x.grad_node, x1.grad_node);
    return;
}