#include <stdio.h>
#include <stdlib.h>
#include "./include/tensor.h"

int main() {
    unsigned int shape[] = {2, 2};
    Tensor a = create_tensor(shape, ARR_SIZE(shape), FLOAT_32);
    float val = 1.0f;
    SCALAR_MUL_TENSOR(&a, &val);
    PRINT_TENSOR(a);
    DEALLOCATE_TENSORS(a);
    return 0;
}