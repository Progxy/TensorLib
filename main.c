#include <stdio.h>
#include <stdlib.h>
#include "./include/autograd.h"

int main() {
    unsigned int shape[] = {1};

    Tensor x, x1, x2, x3, x4;
    alloc_tensor_grad_graph(x, shape, ARR_SIZE(shape), FLOAT_32);
    alloc_tensor_grad_graph(x1, shape, ARR_SIZE(shape), FLOAT_32);
    alloc_tensor_grad_graph(x2, shape, ARR_SIZE(shape), FLOAT_32);
    alloc_tensor_grad_graph(x3, shape, ARR_SIZE(shape), FLOAT_32);
    alloc_tensor_grad_graph(x4, shape, ARR_SIZE(shape), FLOAT_32);
    
    float val = 1.0f;
    fill_tensor(&val, x);
    fill_tensor((val = 0.044715f, &val), x1);
    fill_tensor((val = sqrtf(2.0f / M_PI), &val), x2);    
    fill_tensor((val = 1.0f, &val), x3);    
    fill_tensor((val = 0.5f, &val), x4);
    
    Tensor a, b, c, d, e, f, g, h;
    EMPTY_TENSORS(x.data_type, &a, &b, &c, &d, &e, &f, &g, &h);

    val = 3.0f;

    // Math: 0.5x(1 + {\tanh}[{\sqrt{2/\pi}}({x} + 0.044715{x}^{3})])
    TENSOR_GRAPH_MUL(&d, x2, *TENSOR_GRAPH_SUM(&c, x, *TENSOR_GRAPH_MUL(&b, x1, *TENSOR_GRAPH_POW(&a, x, &val, x.data_type))));
    TENSOR_GRAPH_MUL(&h, *TENSOR_GRAPH_MUL(&g, x, x4), *TENSOR_GRAPH_SUM(&f, x3, *TENSOR_GRAPH_TANH(&e, d, x.data_type)));
    
    derive_node(x.grad_node);
    printf("expr_val: %f, expr_derivative_val: %f\n", CAST_PTR(h.data, float)[0], CAST_PTR(CAST_PTR(x.grad_node, GradNode) -> derived_value.data, float)[0]);
    DEALLOCATE_GRAD_GRAPHS(x.grad_node, x1.grad_node, x2.grad_node, x3.grad_node, x4.grad_node);
    DEALLOCATE_TENSORS(x, x1, x2, x3, x4, a, b, c, d, e, f, g, h);
    
    return 0;
}