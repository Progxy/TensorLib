#ifndef _AUTOGRAD_H_
#define _AUTOGRAD_H_

#include "./tensor.h"

typedef struct GradNode {
    void* value;
    void* derived_value;
    DataType data_type;
    OperatorFlag operation;
    struct GradNode** children;
    struct GradNode* parents[2];
    unsigned int children_count;
} GradNode;

GradNode* alloc_grad_graph_node(DataType data_type) {
    GradNode* node = (GradNode*) calloc(1, sizeof(GradNode));
    node -> data_type = data_type;
    node -> children = NULL;
    node -> children_count = 0;
    return node;  
}

void deallocate_grad_graph(GradNode* node) {
    for (unsigned int i = 0; i < node -> children_count; ++i) {
        deallocate_grad_graph(node -> children + i);
    }
    free(node);
    node = NULL;
    return;
}

void add_child(GradNode* child, GradNode* parent) {
    parent -> children = (GradNode**) realloc(parent -> children, sizeof(GradNode*) * (parent -> children_count + 1));
    parent -> children[(parent -> children_count)++] = child; 
    return;
}

void exec_operation(GradNode* node, void* value_a, void* value_b) {
    switch (node -> operation) {
        case SUMMATION:
            SUM(node -> value, value_a, value_b, node -> data_type);
            break;

        case SUBTRACTION: 
            SUBTRACT(node -> value, value_a, value_b, node -> data_type);
            break;        
        
        case MULTIPLICATION: 
            MULTIPLY(node -> value, value_a, value_b, node -> data_type);
            break;        
        
        case DIVISION: 
            DIVIDE(node -> value, value_a, value_b, node -> data_type);
            break;
        
    }
    return;
}

GradNode* compute_graph(GradNode* node_a, GradNode* node_b, OperatorFlag operation) {
    ASSERT(node_a -> data_type != node_b -> data_type, "DATA_TYPE_MISMATCH");
    GradNode* new_node = alloc_grad_graph_node(node_a -> data_type);
    new_node -> operation = operation;
    add_child(new_node, node_a);
    add_child(new_node, node_b);
    exec_operation(new_node, node_a -> value, node_b -> value);
    return new_node;
}

GradNode* get_other_parent(GradNode* parents[2], GradNode* parent) {
    for (unsigned int i = 0; i < 2; ++i) if (parents[i] != parent) return parents[i];
    return NULL;
}

void* derive_op(GradNode* node, GradNode* child) {
    switch (child -> operation) {
        case SUMMATION:
            ASSIGN(node -> derived_value, 1.0L, node -> data_type);
            break;       
            
        case SUBTRACTION:
            ASSIGN(node -> derived_value, -1.0L, node -> data_type);
            break;        
        
        case MULTIPLICATION:
            if (node -> data_type == FLOAT_32) ASSIGN(node -> derived_value, *CAST_PTR(get_other_parent(child -> parents, node) -> value, float), node -> data_type);
            else if (node -> data_type == FLOAT_64) ASSIGN(node -> derived_value, *CAST_PTR(get_other_parent(child -> parents, node) -> value, double), node -> data_type);
            else if (node -> data_type == FLOAT_128) ASSIGN(node -> derived_value, *CAST_PTR(get_other_parent(child -> parents, node) -> value, long double), node -> data_type);
            break;        
        
        case DIVISION:
            void* temp = calloc(1, node -> data_type);
            void* exp = calloc(1, node -> data_type);
            ASSIGN(temp, 1.0L, node -> data_type);
            ASSIGN(exp, 2.0L , node -> data_type);
            DIVIDE(node -> derived_value, temp, POW(node -> derived_value, node -> value, exp, node -> data_type), node -> data_type);
            MULTIPLY(node -> derived_value, node -> derived_value, get_other_parent(child -> parents, node) -> value, node -> data_type);
            DEALLOCATE_PTRS(temp, exp);
            break;
    }

    return node -> derived_value;
}

void* derive_node(GradNode* node) {
    void* temp = calloc(1, node -> data_type);
    ASSIGN(temp, 0.0L, node -> data_type);
    for (unsigned int i = 0; i < node -> children_count; ++i) {
        MULTIPLY(node -> derived_value, derive_node(node -> children + i), derive_op(node, node -> children[i]), node -> data_type);
        SUM(temp, temp, node -> derived_value, node -> data_type);
    }
    if (node -> data_type == FLOAT_32) ASSIGN(node -> derived_value, *CAST_PTR(temp, float), node -> data_type);
    else if (node -> data_type == FLOAT_64) ASSIGN(node -> derived_value, *CAST_PTR(temp, double), node -> data_type);
    else if (node -> data_type == FLOAT_128) ASSIGN(node -> derived_value, *CAST_PTR(temp, long double), node -> data_type);
    DEALLOCATE_PTRS(temp);
    return node -> derived_value;
}

void forward_graph(GradNode* head) {
    if (head == NULL) return; 
    for (unsigned int i = 0; i < head -> children_count; ++i) {
        exec_operation(head -> children[i], head -> value, get_other_parent(head -> children[i] -> parents, head) -> value);
        forward_graph(head -> children[i]);
    }
    return;
}

// NOTE: you can use the toposort to linearize the graph and run smoothly the forward and backward pass
void toposort(GradNode* graph) {
    return;
}

#endif //_AUTOGRAD_H_