#include <math.h>
#include <stdio.h>
#include "./include/autograd.h"
#include "include/tensor.h"
#include "include/types.h"
#include "include/utils.h"

void test_sigmoid(void);
Tensor test_gelu(Tensor x);
Tensor tensor_sigmoid(Tensor x);

#define DERIVED_TENSOR(node) CAST_PTR(node, GradNode) -> derived_value
#define DERIVED_VALUE(node, type) CAST_PTR(DERIVED_TENSOR(node).data, type)

static Tensor sigmoid_t(Tensor* tensor) {
    Tensor x1;
    void* temp = calloc(1, tensor -> data_type);
    ALLOC_TENSOR_GRAD_GRAPH_FILLED(x1, tensor -> shape, tensor -> rank, tensor -> data_type, ASSIGN(temp, 1.0L, tensor -> data_type));

    Tensor a, b, c, d;
    EMPTY_TENSORS(tensor -> data_type, &a, &b, &c, &d);

    // Math: \frac{1}{1 + e^{-x}}
    TENSOR_GRAPH_POW(&b, *TENSOR_GRAPH_EXP(&a, *tensor), ASSIGN(temp, -1.0L, tensor -> data_type));
    TENSOR_GRAPH_POW(&d, *TENSOR_GRAPH_SUM(&c, x1, b), temp);

    DEALLOCATE_TENSORS(x1, a, b, c);
    DEALLOCATE_PTRS(temp);

    return d;
}

int main(void) {
    // test_sigmoid();

    float val = 1.0f;
    unsigned int shape[] = {2, 1};
    unsigned int shape_w[] = {1, 2};
    unsigned int shape_b[] = {2, 2};

    Tensor activation, weights, biases;
    ALLOC_TENSOR_GRAD_GRAPH_FILLED(activation, shape, ARR_SIZE(shape), FLOAT_32, &val);
    ALLOC_TENSOR_GRAD_GRAPH_FILLED(weights, shape_w, ARR_SIZE(shape_w), FLOAT_32, &val);
    ALLOC_TENSOR_GRAD_GRAPH_FILLED(biases, shape_b, ARR_SIZE(shape_b), FLOAT_32, &val);

    Tensor res = test_gelu(activation);
    Tensor next_activation;
    EMPTY_TENSORS(activation.data_type, &next_activation);
    TENSOR_GRAPH_DOT(&next_activation, res, weights);
    TENSOR_GRAPH_SUM(&next_activation, next_activation, biases);

    res = sigmoid_t(&next_activation);

    PRINT_TENSOR(res, "\t");

    val = 120.0f;
    fill_tensor(&val, *NODE_TENSOR(activation.grad_node));
    set_update_flag(FALSE, activation.grad_node);
    backward_pass(res.grad_node);
    PRINT_TENSOR(*NODE_TENSOR(res.grad_node), "\t");

    val = 1.0f;
    fill_tensor(&val, *NODE_TENSOR(activation.grad_node));
    set_update_flag(FALSE, activation.grad_node);
    backward_pass(res.grad_node);
    PRINT_TENSOR(*NODE_TENSOR(res.grad_node), "\t");

    print_grad_node(activation.grad_node, 0);

    return 0;
}

Tensor tensor_sigmoid(Tensor x) {
    unsigned int shape[] = { 2, 2 };
    float temp_val = 0.0f;

    Tensor x1;
    ALLOC_TENSOR_GRAD_GRAPH_FILLED(x1, shape, ARR_SIZE(shape), FLOAT_32, (temp_val = 1.0f, &temp_val));

    Tensor a, b, c, d;
    EMPTY_TENSORS(x.data_type, &a, &b, &c, &d);

    // Math: \frac{1}{1 + e^{-x}}
    TENSOR_GRAPH_POW(&b, *TENSOR_GRAPH_EXP(&a, x), (temp_val = -1.0f, &temp_val));
    TENSOR_GRAPH_POW(&d, *TENSOR_GRAPH_SUM(&c, x1, b), (temp_val = -1.0f, &temp_val));

    return d;
}

void test_sigmoid(void) {
    unsigned int shape[] = { 2, 2 };
    float val = 1.0f;
    float temp_val = 0.0f;

    Tensor x, x1;
    ALLOC_TENSOR_GRAD_GRAPH_FILLED(x, shape, ARR_SIZE(shape), FLOAT_32, &val);
    ALLOC_TENSOR_GRAD_GRAPH_FILLED(x1, shape, ARR_SIZE(shape), FLOAT_32, (temp_val = 1.0f, &temp_val));

    Tensor a, b, c, d;
    EMPTY_TENSORS(x.data_type, &a, &b, &c, &d);

    // Math: \frac{1}{1 + e^{-x}}
    TENSOR_GRAPH_POW(&b, *TENSOR_GRAPH_EXP(&a, x), (temp_val = -1.0f, &temp_val));
    TENSOR_GRAPH_POW(&d, *TENSOR_GRAPH_SUM(&c, x1, b), (temp_val = -1.0f, &temp_val));

    printf("Result: \n");
    PRINT_TENSOR(d, "\t");
    derive_r_node(d.grad_node, TRUE);
    printf("Diff result: \n");
    PRINT_TENSOR(DERIVED_TENSOR(x.grad_node), "\t");
    DEALLOCATE_GRAD_GRAPHS(x.grad_node, x1.grad_node);

    Tensor sigmoid_tensor = alloc_tensor(shape, ARR_SIZE(shape), FLOAT_32);
    fill_tensor(&val, sigmoid_tensor);

    unsigned int size = TENSOR_SIZE(sigmoid_tensor);
    for (unsigned int i = 0; i < size; ++i) {
        sigmoid_func(CAST_PTR(sigmoid_tensor.data, float) + i, CAST_PTR(sigmoid_tensor.data, float) + i, sigmoid_tensor.data_type);
    }

    printf("Result: \n");
    PRINT_TENSOR(sigmoid_tensor, "\t");
    Tensor temp = alloc_tensor(sigmoid_tensor.shape, sigmoid_tensor.rank, sigmoid_tensor.data_type);
    fill_tensor((temp_val = 1.0f, &temp_val), temp);
    MULTIPLY_TENSOR(&sigmoid_tensor, sigmoid_tensor, *SUBTRACT_TENSOR(&temp, temp, sigmoid_tensor));
    printf("Diff Result: \n");
    PRINT_TENSOR(sigmoid_tensor, "\t");
    DEALLOCATE_TENSORS(sigmoid_tensor, temp);

    return;
}

Tensor test_gelu(Tensor x) {
    unsigned int shape[] = {2, 1};
    float val = 1.0f;

    Tensor x1, x2, x3, x4;
    ALLOC_TENSOR_GRAD_GRAPH_FILLED(x1, shape, ARR_SIZE(shape), FLOAT_32, (val = 0.044715f, &val));
    ALLOC_TENSOR_GRAD_GRAPH_FILLED(x2, shape, ARR_SIZE(shape), FLOAT_32, (val = sqrtf(2.0f / M_PI), &val));
    ALLOC_TENSOR_GRAD_GRAPH_FILLED(x3, shape, ARR_SIZE(shape), FLOAT_32, (val = 1.0f, &val));
    ALLOC_TENSOR_GRAD_GRAPH_FILLED(x4, shape, ARR_SIZE(shape), FLOAT_32, (val = 0.5f, &val));

    Tensor a, b, c, d, e, f, g, h;
    EMPTY_TENSORS(x.data_type, &a, &b, &c, &d, &e, &f, &g, &h);

    // Math: 0.5x(1 + {\tanh}[{\sqrt{2/\pi}}({x} + 0.044715{x}^{3})])
    TENSOR_GRAPH_MUL(&d, x2, *TENSOR_GRAPH_SUM(&c, x, *TENSOR_GRAPH_MUL(&b, x1, *TENSOR_GRAPH_POW(&a, x, (val = 3.0f, &val)))));
    TENSOR_GRAPH_MUL(&h, *TENSOR_GRAPH_MUL(&g, x, x4), *TENSOR_GRAPH_SUM(&f, x3, *TENSOR_GRAPH_TANH(&e, d)));

    return h;
}
