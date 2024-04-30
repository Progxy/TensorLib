#include <stdio.h>
#include <stdlib.h>
#include "./include/tensor.h"

int main() {
    unsigned int shape[] = {2, 2};
    const float data[] = { 
        1.0f, 0.0f,
        0.0f, 1.0f
    };
    Tensor a = alloc_tensor(shape, ARR_SIZE(shape), FLOAT_32);
    Tensor b = alloc_tensor(shape, ARR_SIZE(shape), FLOAT_32);
    set_tensor((void*) data, a);
    set_tensor((void*) data, b);
    PRINT_TENSOR(a);
    PRINT_TENSOR(b);
    Tensor c = alloc_tensor(shape, ARR_SIZE(shape), FLOAT_32);
    unsigned int middle = (a.rank + b.rank) / 2;
    contract_tensor(cross_product_tensor(&c, a, b), middle, middle - 1);
    PRINT_TENSOR(c);
    DEALLOCATE_TENSORS(a, b);
    return 0;
}