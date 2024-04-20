#include <stdio.h>
#include <stdlib.h>
#include "./include/tensor.h"

int main() {
    unsigned int shape[] = {2, 2, 2, 2};
    char err_msg[MSG_MAX_LEN];
    Tensor tensor = create_tensor(shape, ARR_SIZE(shape), FLOAT_32, err_msg);
    if (tensor.data == NULL) {
        printf("ERROR: failed to create the tensor: '%s'!\n", err_msg);
        return -1;
    }
    print_tensor(tensor);
    return 0;
}