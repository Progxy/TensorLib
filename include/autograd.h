#ifndef _AUTOGRAD_H_
#define _AUTOGRAD_H_

typedef enum OperatorFlag { SUMMATION, SUBTRACTION, MULTIPLICATION, DIVISION } OperatorFlag;
typedef struct GradNode {
    float value;
    float derived_value;
    OperatorFlag operation;
    struct GradNode** children;
    struct GradNode* parents[2];
    unsigned int children_count;
} GradNode;

static inline float powf(float base, float exp) {
    float val = base;
    for (float i = 1.0f; i < exp; ++i) val *= base;
    return val;
}

GradNode* alloc_grad_graph_node() {
    GradNode* node = (GradNode*) calloc(1, sizeof(GradNode));
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

void exec_operation(GradNode* node, float value_a, float value_b) {
    switch (node -> operation) {
        case SUMMATION:
            node -> value = value_a + value_b;
            break;

        case SUBTRACTION: 
            node -> value = value_a - value_b;
            break;        
        
        case MULTIPLICATION: 
            node -> value = value_a * value_b;
            break;      
        
        case DIVISION: 
            node -> value = value_a / value_b;
            break;
        
    }
    return;
}

GradNode* compute_graph(GradNode* node_a, GradNode* node_b, OperatorFlag operation) {
    GradNode* new_node = alloc_grad_graph_node();
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

float derive_op(GradNode* node, GradNode* child) {
    switch (child -> operation) {
        case SUMMATION:
            node -> derived_value = 1.0f;
            break;       
            
        case SUBTRACTION:
            node -> derived_value = -1.0f;
            break;        
        
        case MULTIPLICATION:
            node -> derived_value = get_other_parent(child -> parents, node) -> value;
            break;       
        
        case DIVISION:
            node -> derived_value = get_other_parent(child -> parents, node) -> value * (1.0f / powf(node -> value, 2.0f));
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

void forward_graph(GradNode* head) {
    if (head == NULL) return; 
    for (unsigned int i = 0; i < head -> children_count; ++i) {
        exec_operation(head -> children[i], head -> value, get_other_parent(head -> children[i] -> parents, head) -> value);
        forward_graph(head -> children[i]);
    }
    return;
}

float operation(float a, float b, OperatorFlag operator) {
    float c = 0.0f;
    switch (operator) {
        case SUMMATION: 
            c = a + b;
            break;
    }
    return c;
}

// NOTE: you can use the toposort to linearize the graph and run smoothly the forward and backward pass
void toposort(GradNode* graph) {
    return;
}

#endif //_AUTOGRAD_H_