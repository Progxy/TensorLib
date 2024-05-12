#ifndef _AUTOGRAD_H_
#define _AUTOGRAD_H_

#include "./tensor.h"

#define IS_DENOMINATOR(parent, child) parent == child -> parents[1]
#define TENSOR_GRAPH_MULTIPLY(c, a, b) graph_op(c, a, b, MULTIPLICATION)
#define TENSOR_GRAPH_DIVIDE(c, a, b) graph_op(c, a, b, DIVISION)
#define TENSOR_GRAPH_SUBTRACT(c, a, b) graph_op(c, a, b, SUBTRACTION)
#define TENSOR_GRAPH_SUM(c, a, b) graph_op(c, a, b, SUM)

void alloc_grad_graph_node(DataType data_type, Tensor* value) {
    GradNode* node = (GradNode*) calloc(1, sizeof(GradNode));
    node -> children = NULL;
    node -> value = value;
    node -> children_count = 0;
    node -> derived_value = empty_tensor(data_type);
    value -> grad_node = node;
    return;  
}

void deallocate_grad_graph(GradNode* node) {
    for (unsigned int i = 0; i < node -> children_count; ++i) {
        deallocate_grad_graph(node -> children[i]);
    }
    DEALLOCATE_TENSORS(node -> derived_value);
    free(node -> children);
    free(node -> parents);
    free(node);
    node = NULL;
    return;
}

void add_child(GradNode* child, GradNode* parent) {
    parent -> children = (GradNode**) realloc(parent -> children, sizeof(GradNode*) * (parent -> children_count + 1));
    parent -> children[(parent -> children_count)++] = child;     
    child -> parents = (GradNode**) realloc(child -> parents, sizeof(GradNode*) * (child -> parents_count + 1));
    child -> parents[(child -> parents_count)++] = parent; 
    return;
}

Tensor alloc_graph_tensor(unsigned int* shape, unsigned int rank, DataType data_type) {
    Tensor tensor = alloc_tensor(shape, rank, data_type);
    alloc_grad_graph_node(data_type, &tensor); 
    return tensor;
}

Tensor* graph_op(Tensor* c, Tensor a, Tensor b, OperatorFlag operation) {
    op_tensor(c, a, b, operation);
    alloc_grad_graph_node(a.data_type, c);
    CAST_PTR(c -> grad_node, GradNode) -> operation = operation; 
    add_child(c -> grad_node, a.grad_node);
    add_child(c -> grad_node, b.grad_node);
    return c;
}

GradNode* get_other_parent(GradNode* child, GradNode* parent) {
    for (unsigned int i = 0; i < child -> parents_count; ++i) if (child -> parents[i] != parent) return child -> parents[i];
    return NULL;
}

void derive_op(GradNode* node, GradNode* child) {
    switch (child -> operation) {
        case SUM: {
            void* temp = calloc(1, node -> derived_value.data_type);
            ASSIGN(temp, 1.0L, node -> derived_value.data_type);
            set_tensor(temp, node -> derived_value);
            free(temp);
            break;       
        }

        case SUBTRACTION: {
            void* temp = calloc(1, node -> derived_value.data_type);
            ASSIGN(temp, -1.0L, node -> derived_value.data_type);
            set_tensor(temp, node -> derived_value);
            free(temp);
            break;        
        }

        case MULTIPLICATION:
            copy_tensor(&(node -> derived_value), *(get_other_parent(child, node) -> value));
            break;       
        
        case DIVISION: {
            if (IS_DENOMINATOR(node, child)) {
                Tensor temp = empty_tensor(node -> derived_value.data_type);
                void* exp = calloc(1, temp.data_type);
                pow_tensor(&temp, *(node -> value), ASSIGN(exp, -2.0L, temp.data_type));
                free(exp);
                copy_tensor(&(node -> derived_value), *(get_other_parent(child, node) -> value));
                MULTIPLY_TENSOR(&(node -> derived_value), node -> derived_value, temp);
                DEALLOCATE_TENSORS(temp);
            } else {
                void* exp = calloc(1, node -> derived_value.data_type);
                pow_tensor(&(node -> derived_value), *(get_other_parent(child, node) -> value), ASSIGN(exp, -1.0L, node -> derived_value.data_type));
                free(exp);
            }
            break;
        }
    }
    return;
}

void derive_node(GradNode* node) {
    Tensor diff = empty_tensor(node -> derived_value.data_type);
    for (unsigned int i = 0; i < node -> children_count; ++i) {
        derive_node(node -> children[i]); 
        derive_op(node, node -> children[i]);
        Tensor temp = empty_tensor(node -> derived_value.data_type);
        SUM_TENSOR(&diff, diff, MULTIPLY_TENSOR(&temp, node -> derived_value, node -> children[i] -> derived_value));
        DEALLOCATE_TENSORS(temp);
    }
    copy_tensor(&(node -> derived_value), diff);
    DEALLOCATE_TENSORS(diff);
    return;
}

// NOTE: you can use the toposort to linearize the graph and run smoothly the forward and backward pass
// void toposort(GradNode* graph) {
//     return;
// }

#endif //_AUTOGRAD_H_