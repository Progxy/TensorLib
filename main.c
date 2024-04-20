#include <stdio.h>
#include <stdlib.h>
#include "./include/tensor.h"

int main() {
    unsigned int shape[] = {2, 2, 2, 2};
    Tensor a = create_tensor(shape, ARR_SIZE(shape), FLOAT_32);
    Tensor b = create_tensor(shape, ARR_SIZE(shape), FLOAT_32);
    randomize_tensor(b);
    Tensor c = create_tensor(shape, ARR_SIZE(shape), FLOAT_32);
    SUM_TENSOR(&c, a, b);
    PRINT_TENSOR(c);
    SUBTRACT_TENSOR(&a, c, b);
    PRINT_TENSOR(a);
    DEALLOCATE_TENSORS(a, b, c);
    return 0;
}