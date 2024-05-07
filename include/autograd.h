#ifndef _AUTOGRAD_H_
#define _AUTOGRAD_H_

#include "./tensor.h"

static inline float powf(float base, float exp) {
    float val = base;
    for (float i = 1.0f; i < exp; ++i) val *= base;
    return val;
}

GradNode* alloc_grad_graph_node(DataType data_type) {
    GradNode* node = (GradNode*) calloc(1, sizeof(GradNode));
    node -> children = NULL;
    node -> children_count = 0;
    node -> derived_value = empty_tensor(data_type);
    return node;  
}

void deallocate_grad_graph(GradNode* node) {
    for (unsigned int i = 0; i < node -> children_count; ++i) {
        deallocate_grad_graph(node -> children + i);
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

Tensor compute_graph(Tensor a, Tensor b, OperatorFlag operation) {
    Tensor c = empty_tensor(a.data_type);
    op_tensor(&c, a, b, operation);
    GradNode* new_node = alloc_grad_graph_node(a.data_type);
    new_node -> operation = operation;
    add_child(new_node, a.grad_node);
    add_child(new_node, b.grad_node);
    return c;
}

GradNode* get_other_parent(GradNode* child, GradNode* parent) {
    for (unsigned int i = 0; i < child -> parents_count; ++i) if (child -> parents[i] != parent) return child -> parents[i];
    return NULL;
}

float derive_op(GradNode* node, GradNode* child) {
    switch (child -> operation) {
        case SUM:
            void* temp = calloc(1, node -> derived_value.data_type);
            ASSIGN(temp, 1.0L, node -> derived_value.data_type);
            set_tensor(temp, node -> derived_value);
            free(temp);
            break;       
            
        case SUBTRACTION:
            void* temp = calloc(1, node -> derived_value.data_type);
            ASSIGN(temp, -1.0L, node -> derived_value.data_type);
            set_tensor(temp, node -> derived_value);
            free(temp);
            break;        
        
        case MULTIPLICATION:
            node -> derived_value = get_other_parent(child, node) -> value;
            break;       
        
        case DIVISION:
            node -> derived_value = get_other_parent(child, node) -> value * (1.0f / powf(node -> value, 2.0f));
            break;
    }

    return node -> derived_value;
}

float derive_node(GradNode* node) {
    float temp = 0.0f;
    for (unsigned int i = 0; i < node -> children_count; ++i) {
        node -> derived_value = derive_node(node -> children + i) * derive_op(node, node -> children[i]);
        temp += node -> derived_value;
    }
    node -> derived_value = temp;
    return node -> derived_value;
}

// NOTE: you can use the toposort to linearize the graph and run smoothly the forward and backward pass
void toposort(GradNode* graph) {
    return;
}

#endif //_AUTOGRAD_H_