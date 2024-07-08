#include <stdio.h>
#include <stdlib.h>
#include "./include/autograd.h"

void test_sigmoid();
void test_gelu();

#define DERIVED_TENSOR(node) CAST_PTR(node, GradNode) -> derived_value
#define DERIVED_VALUE(node, type) CAST_PTR(DERIVED_TENSOR(node).data, type)

int main() {
    // test_sigmoid();
    // test_gelu();
    Tensor a = identity_tensor(2, 2, FLOAT_32);
    transpose_tensor(&a);
    PRINT_TENSOR(a, "\t");
    return 0;
}

void test_sigmoid() {
    unsigned int shape[] = { 2, 2 };    
    float val = 1.0f;
    
    Tensor x, x1;
    alloc_tensor_grad_graph_filled(x, shape, ARR_SIZE(shape), FLOAT_32, &val);
    alloc_tensor_grad_graph_filled(x1, shape, ARR_SIZE(shape), FLOAT_32, &val);

    Tensor a, b, c, d;
    EMPTY_TENSORS(x.data_type, &a, &b, &c, &d);

    // Math: \frac{1}{1 + e^{-x}}
    TENSOR_GRAPH_POW(&b, *TENSOR_GRAPH_EXP(&a, x, x.data_type), (val = -1.0f, &val), x.data_type);
    TENSOR_GRAPH_POW(&d, *TENSOR_GRAPH_SUM(&c, x1, b), &val, x.data_type);
    
    printf("Result: \n");
    PRINT_TENSOR(d, "\t");
    derive_r_node(d.grad_node, TRUE);
    printf("Diff result: \n");
    PRINT_TENSOR(DERIVED_TENSOR(x.grad_node), "\t");
    DEALLOCATE_GRAD_GRAPHS(x.grad_node, x1.grad_node);
    
    Tensor sigmoid_tensor = alloc_tensor(shape, ARR_SIZE(shape), FLOAT_32);
    fill_tensor((val = 1.0f, &val), sigmoid_tensor);

    unsigned int size = TENSOR_SIZE(sigmoid_tensor);
    for (unsigned int i = 0; i < size; ++i) {
        sigmoid_func(CAST_PTR(sigmoid_tensor.data, float) + i, CAST_PTR(sigmoid_tensor.data, float) + i, sigmoid_tensor.data_type);
    }
    
    printf("Result: \n");
    PRINT_TENSOR(sigmoid_tensor, "\t");
    Tensor temp = empty_tensor(sigmoid_tensor.data_type);
    RESHAPE_TENSOR(&temp, sigmoid_tensor);
    fill_tensor(&val, temp);
    MULTIPLY_TENSOR(&sigmoid_tensor, sigmoid_tensor, *SUBTRACT_TENSOR(&temp, temp, sigmoid_tensor));
    printf("Diff Result: \n");
    PRINT_TENSOR(sigmoid_tensor, "\t");
    DEALLOCATE_TENSORS(sigmoid_tensor, temp);

    return;
}

void test_gelu() {
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
    
    derive_r_node(h.grad_node, TRUE);
    printf("expr_val: %f, expr_derivative_val: %f\n", CAST_PTR(h.data, float)[0], DERIVED_VALUE(x.grad_node, float)[0]);
    DEALLOCATE_GRAD_SINGLE_GRAPHS(x1.grad_node, x2.grad_node, x3.grad_node, x4.grad_node);
    DEALLOCATE_GRAD_GRAPHS(x.grad_node);
    return;
}