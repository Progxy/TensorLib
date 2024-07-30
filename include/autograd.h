#ifndef _AUTOGRAD_H_
#define _AUTOGRAD_H_

#include "./tensor.h"

#define alloc_tensor_grad_graph_filled(tensor, shape, rank, data_type, val) alloc_grad_graph_node(data_type, (tensor = alloc_tensor(shape, rank, data_type), fill_tensor(val, tensor), &tensor))
#define alloc_tensor_grad_graph(tensor, shape, rank, data_type) alloc_grad_graph_node(data_type, (tensor = alloc_tensor(shape, rank, data_type), &tensor))
#define DEALLOCATE_GRAD_SINGLE_GRAPHS(...) deallocate_grad_graphs(sizeof((GradNode*[]){__VA_ARGS__}) / sizeof(GradNode*), (int) TRUE, __VA_ARGS__)
#define DEALLOCATE_GRAD_GRAPHS(...) deallocate_grad_graphs(sizeof((GradNode*[]){__VA_ARGS__}) / sizeof(GradNode*), (int) FALSE, __VA_ARGS__)
#define NODE_DERIVED_TENSOR(node) CAST_PTR(node, GradNode) -> derived_value
#define IS_DENOMINATOR(parent, child) parent == child -> parents[1]
#define NODE_TENSOR(node) CAST_PTR(node, GradNode) -> value
#define DERIVE_NODE_REVERSE(node) derive_r_node(node, TRUE)

// TENSOR FUNCTIONS OPERATIONS
#define TENSOR_GRAPH_NORM(c, a, val) graph_op(c, a, (Tensor) {.data = val, .data_type = (a).data_type}, NORM)
#define TENSOR_GRAPH_POW(c, a, val) graph_op(c, a, (Tensor) {.data = val, .data_type = (a).data_type}, POW)
#define TENSOR_GRAPH_SOFTMAX(c, a) graph_op(c, a, (Tensor) {.data_type = (a).data_type}, SOFTMAX)
#define TENSOR_GRAPH_TANH(c, a) graph_op(c, a, (Tensor) {.data_type = (a).data_type}, TANH)
#define TENSOR_GRAPH_SQRT(c, a) graph_op(c, a, (Tensor) {.data_type = (a).data_type}, SQRT)
#define TENSOR_GRAPH_EXP(c, a) graph_op(c, a, (Tensor) {.data_type = (a).data_type}, EXP)
#define TENSOR_GRAPH_LOG(c, a) graph_op(c, a, (Tensor) {.data_type = (a).data_type}, LOG)
#define TENSOR_GRAPH_ABS(c, a) graph_op(c, a, (Tensor) {.data_type = (a).data_type}, ABS)

// TENSORS OPERATIONS
#define TENSOR_GRAPH_MUL(c, a, b) graph_op(c, a, b, MULTIPLICATION)
#define TENSOR_GRAPH_SUB(c, a, b) graph_op(c, a, b, SUBTRACTION)
#define TENSOR_GRAPH_DIV(c, a, b) graph_op(c, a, b, DIVISION)
#define TENSOR_GRAPH_DOT(c, a, b) graph_op(c, a, b, DOT)
#define TENSOR_GRAPH_SUM(c, a, b) graph_op(c, a, b, SUM)
#define TENSOR_GRAPH_MAX(c, a, b) graph_op(c, a, b, MAX)
#define TENSOR_GRAPH_MIN(c, a, b) graph_op(c, a, b, MIN)

void alloc_grad_graph_node(DataType data_type, Tensor* value) {
    GradNode* node = (GradNode*) calloc(1, sizeof(GradNode));
    node -> children = NULL;
    node -> value = (Tensor*) calloc(1, sizeof(Tensor));
    copy_tensor(node -> value, *value);
    node -> exp = NULL;
    node -> children_count = 0;
    node -> derived_value = empty_tensor(data_type);
    reshape_tensor(&(node -> derived_value), node -> value -> shape, node -> value -> rank, node -> value -> data_type);
    value -> grad_node = node;
    return;  
}

void* deallocate_grad_graph(bool single_removal_flag, GradNode* node, void*** deallocated_ptrs, unsigned int* deallocated_ptrs_count) {
    for (unsigned int i = 0; i < *deallocated_ptrs_count; ++i) {
        if ((*deallocated_ptrs)[i] == node) return NULL;
    }
    for (unsigned int i = 0; (i < node -> children_count) && !single_removal_flag; ++i) {
        if (node -> children[i] != NULL) node -> children[i] = deallocate_grad_graph(single_removal_flag, node -> children[i], deallocated_ptrs, deallocated_ptrs_count);
    }
    DEALLOCATE_TENSORS(node -> derived_value, *(node -> value));
    free(node -> value);
    free(node -> children);
    node -> children = NULL;
    free(node -> parents);
    node -> parents = NULL;
    if (node -> exp != NULL) free(node -> exp);
    node -> exp = NULL;
    *deallocated_ptrs = (void**) realloc(*deallocated_ptrs, sizeof(void*) * ((*deallocated_ptrs_count) + 1));
    (*deallocated_ptrs)[(*deallocated_ptrs_count)++] = node;
    free(node);
    node = NULL;
    return node;
}

void deallocate_grad_graphs(int len, ...) {
    va_list args;
    va_start(args, len);
    bool single_removal_flag = va_arg(args, int);
    void** deallocated_ptrs = (void**) calloc(1, sizeof(void*));
    unsigned int deallocated_ptrs_count = 0;
    for (int i = 0; i < len; ++i) {
        GradNode* node = va_arg(args, GradNode*);
        deallocate_grad_graph(single_removal_flag, node, &deallocated_ptrs, &deallocated_ptrs_count);
    }
    free(deallocated_ptrs);
    va_end(args);
    return;
}

void add_child(GradNode* child, GradNode* parent) {
    parent -> children = (GradNode**) realloc(parent -> children, sizeof(GradNode*) * (parent -> children_count + 1));
    parent -> children[(parent -> children_count)++] = child;
    child -> parents = (GradNode**) realloc(child -> parents, sizeof(GradNode*) * (child -> parents_count + 1));
    child -> parents[(child -> parents_count)++] = parent; 
    return;
}

Tensor* graph_op(Tensor* c, Tensor a, Tensor b, OperatorFlag operation) {
    op_tensor(c, a, b, operation);
    alloc_grad_graph_node(a.data_type, c);
    CAST_PTR(c -> grad_node, GradNode) -> operation = operation; 
    add_child(c -> grad_node, a.grad_node);
    if (operation == EXP || operation == TANH || operation == LOG || operation == ABS || operation == SOFTMAX) return c;
    else if (operation == POW || operation == NORM) {
        CAST_PTR(c -> grad_node, GradNode) -> exp = (void*) calloc(1, a.data_type);
        mem_copy(CAST_PTR(c -> grad_node, GradNode) -> exp, b.data, b.data_type, 1);
        return c;
    } else if (operation == SQRT) {
        CAST_PTR(c -> grad_node, GradNode) -> exp = calloc(1, a.data_type);
        void* val = (void*) calloc(1, a.data_type);
        ASSIGN(val, 0.5L, a.data_type);
        mem_copy(CAST_PTR(c -> grad_node, GradNode) -> exp, val, a.data_type, 1);
        free(val);
        return c;
    }
    add_child(c -> grad_node, b.grad_node);
    return c;
}

void derive_op(GradNode* node, GradNode* child) {
    switch (child -> operation) {
        case SUM: {
            copy_tensor(&(node -> derived_value), child -> derived_value);
            break;       
        }

        case SUBTRACTION: {
            if (node == child -> parents[0]) copy_tensor(&(node -> derived_value), child -> derived_value);
            else CONJUGATE_TENSOR(&(node -> derived_value), child -> derived_value);
            break;        
        }

        case MULTIPLICATION: {
            Tensor* other_parent = (node == child -> parents[0]) ? child -> parents[1] -> value : child -> parents[0] -> value;
            MULTIPLY_TENSOR(&(node -> derived_value), child -> derived_value, *other_parent);
            break;       
        }
        
        case DIVISION: {
            Tensor* other_parent = (node == child -> parents[0]) ? child -> parents[1] -> value : child -> parents[0] -> value;
            if (IS_DENOMINATOR(node, child)) {
                void* val = (void*) calloc(1, node -> derived_value.data_type);
                ASSIGN(val, 2.0L, node -> derived_value.data_type);
                MULTIPLY_TENSOR(&(node -> derived_value), child -> derived_value, *DIVIDE_TENSOR(&(node -> derived_value), *other_parent, *POW_TENSOR(&(node -> derived_value), *(node -> value), val)));
                CONJUGATE_TENSOR(&(node -> derived_value), node -> derived_value);
                free(val);
            } else {    
                DIVIDE_TENSOR(&(node -> derived_value), child -> derived_value, *other_parent);
            }
            break;
        }

        case SQRT:
        case POW: {
            void* temp = calloc(1, node -> derived_value.data_type);
            void* tmp = calloc(1, node -> derived_value.data_type);
            copy_tensor(&(node -> derived_value), *(node -> value));
            POW_TENSOR(&(node -> derived_value), node -> derived_value, SCALAR_SUB(temp, child -> exp, ASSIGN(tmp, 1.0L, node -> derived_value.data_type), node -> derived_value.data_type));
            SCALAR_MUL_TENSOR(&(node -> derived_value), child -> exp);
            free(tmp);
            free(temp);
            MULTIPLY_TENSOR(&(node -> derived_value), child -> derived_value, node -> derived_value);
            break;
        }

        case EXP: {
            copy_tensor(&(node -> derived_value), *(child -> value));
            MULTIPLY_TENSOR(&(node -> derived_value), child -> derived_value, node -> derived_value);
            break;
        }

        case TANH: {
            void* val = calloc(1, node -> derived_value.data_type);
            TANH_TENSOR(&(node -> derived_value), *(node -> value));
            POW_TENSOR(&(node -> derived_value), node -> derived_value, ASSIGN(val, 2.0L, node -> derived_value.data_type));
            SCALAR_SUM_TENSOR(CONJUGATE_TENSOR(&(node -> derived_value), node -> derived_value), ASSIGN(val, 1.0L, node -> derived_value.data_type));
            free(val);
            MULTIPLY_TENSOR(&(node -> derived_value), child -> derived_value, node -> derived_value);
            break;
        }

        case DOT: {
            if (node == child -> parents[0]) {
                Tensor b_t = empty_tensor(node -> derived_value.data_type);
                transpose_tensor(copy_tensor(&b_t, *(child -> parents[1] -> value)));
                DOT_TENSOR(&(node -> derived_value), child -> derived_value, b_t);
                DEALLOCATE_TENSORS(b_t);
            } else {
                Tensor a_t = empty_tensor(node -> derived_value.data_type);
                transpose_tensor(copy_tensor(&a_t, *(child -> parents[0] -> value)));
                DOT_TENSOR(&(node -> derived_value), a_t, child -> derived_value);
                DEALLOCATE_TENSORS(a_t);
            }
            break;
        }

        case LOG: {
            Tensor temp = alloc_tensor(node -> value -> shape, node -> value -> rank, node -> value -> data_type);
            void* val = (void*) calloc(1, node -> derived_value.data_type);
            fill_tensor(ASSIGN(val, 1.0L, node -> derived_value.data_type), temp);
            free(val);
            DIVIDE_TENSOR(&(node -> derived_value), temp, *(node -> value));
            MULTIPLY_TENSOR(&(node -> derived_value), child -> derived_value, node -> derived_value);
            break;
        }

        case MAX:
        case MIN: {
            void* val = (void*) calloc(1, node -> derived_value.data_type);
            fill_tensor(val, node -> derived_value);
            ASSIGN(val, 1.0L, node -> derived_value.data_type);
            unsigned int size = TENSOR_SIZE(node -> derived_value);
            for (unsigned int i = 0; i < size; ++i) {
                if (!IS_EQUAL(CAST_PTR_AT_INDEX(child -> value -> data, i, node -> derived_value.data_type), CAST_PTR_AT_INDEX(node -> value -> data, i, node -> value -> data_type), node -> derived_value.data_type)) continue;
                ASSIGN(CAST_PTR_AT_INDEX(node -> derived_value.data, i, node -> derived_value.data_type), 1.0L, node -> derived_value.data_type);
            }
            free(val);
            MULTIPLY_TENSOR(&(node -> derived_value), child -> derived_value, node -> derived_value);
            break;
        }

        case ABS: {
            unsigned int size = TENSOR_SIZE(node -> derived_value);
            for (unsigned int i = 0; i < size; ++i) {
                if (IS_NEGATIVE(CAST_PTR_AT_INDEX(node -> value -> data, i, node -> value -> data_type), node -> value -> data_type)) ASSIGN(CAST_PTR_AT_INDEX(node -> derived_value.data, i, node -> derived_value.data_type), -1.0L, node -> derived_value.data_type); 
                else if (IS_POSITIVE(CAST_PTR_AT_INDEX(node -> value -> data, i, node -> value -> data_type), node -> value -> data_type)) ASSIGN(CAST_PTR_AT_INDEX(node -> derived_value.data, i, node -> derived_value.data_type), 1.0L, node -> derived_value.data_type); 
            }
            MULTIPLY_TENSOR(&(node -> derived_value), child -> derived_value, node -> derived_value);
            break;
        }
                
        case CONJUGATE: {
            ASSERT(TRUE, "Impossible to differentiate the CONJUGATE operation!");
            break;
        }

        case NORM: {
            void* temp = (void*) calloc(1, node -> derived_value.data_type);
            SCALAR_SUB(temp, child -> exp, ASSIGN(temp, 1.0L, node -> derived_value.data_type), node -> derived_value.data_type);
            void* zero = (void*) calloc(1, node -> derived_value.data_type);
            unsigned int size = TENSOR_SIZE(node -> derived_value);
            for (unsigned int i = 0; i < size; ++i) {
                if (IS_EQUAL(CAST_PTR_AT_INDEX(node -> value -> data, i, node -> value -> data_type), zero, node -> value -> data_type)) continue;
                if (IS_EQUAL(temp, zero, node -> derived_value.data_type)) ASSIGN(CAST_PTR_AT_INDEX(node -> derived_value.data, i, node -> derived_value.data_type), 1.0L, node -> derived_value.data_type);
                else {
                    SCALAR_MUL(CAST_PTR_AT_INDEX(node -> derived_value.data, i, node -> derived_value.data_type), SCALAR_POW(CAST_PTR_AT_INDEX(node -> derived_value.data, i, node -> derived_value.data_type), CAST_PTR_AT_INDEX(node -> value -> data, i, node -> value -> data_type), temp, node -> derived_value.data_type), temp, node -> derived_value.data_type);
                    CAST_AND_OP_INDEX(node -> derived_value.data, child -> value -> data, node -> derived_value.data, i, node -> derived_value.data_type, DIVISION);
                }
                if (IS_NEGATIVE(CAST_PTR_AT_INDEX(node -> value -> data, i, node -> value -> data_type), node -> value -> data_type) && !IS_NEGATIVE(CAST_PTR_AT_INDEX(node -> derived_value.data, i, node -> derived_value.data_type), node -> derived_value.data_type)) CAST_AND_SINGLE_OP_INDEX(node -> derived_value.data, node -> derived_value.data, i, node -> derived_value.data_type, CONJUGATE);
            }
            DEALLOCATE_PTRS(temp, zero);
            MULTIPLY_TENSOR(&(node -> derived_value), child -> derived_value, node -> derived_value);
            break;
        }
        
        case SOFTMAX: {
            unsigned int size = TENSOR_SIZE(node -> derived_value);
            unsigned int shape[] = {size, size};
            Tensor temp_tensor = alloc_tensor(shape, ARR_SIZE(shape), node -> derived_value.data_type);
            void* temp = (void*) calloc(1, node -> derived_value.data_type);
            void* one = (void*) calloc(1, node -> derived_value.data_type);
            ASSIGN(one, 1.0L, node -> derived_value.data_type);

            for (unsigned int i = 0; i < size; ++i) {
                for (unsigned int j = 0; j < size; ++j) {
                    if (i == j) SCALAR_MUL(CAST_PTR_AT_INDEX(temp_tensor.data, i * size + j, temp_tensor.data_type), CAST_PTR_AT_INDEX(child -> value -> data, i, child -> value -> data_type), SCALAR_SUB(temp, one, CAST_PTR_AT_INDEX(child -> value -> data, i, child -> value -> data_type), child -> value -> data_type), temp_tensor.data_type); 
                    else SCALAR_CONJUGATE(CAST_PTR_AT_INDEX(temp_tensor.data, i * size + j, temp_tensor.data_type), SCALAR_MUL(CAST_PTR_AT_INDEX(temp_tensor.data, i * size + j, temp_tensor.data_type), CAST_PTR_AT_INDEX(child -> value -> data, j, child -> value -> data_type), CAST_PTR_AT_INDEX(child -> value -> data, i, child -> value -> data_type), temp_tensor.data_type), temp_tensor.data_type); 
                }
            } 
        
            DOT_TENSOR(&(node -> derived_value), temp_tensor, *(child -> value));

            DEALLOCATE_TENSORS(temp_tensor);
            DEALLOCATE_PTRS(temp, one);
            
            break;
        }
    }
    return;
}

// Derive using forward-mode
void derive_node(GradNode* node) {
    // Seed the sink with 1.0
    if (node -> children_count == 0) {
        copy_tensor(&(node -> derived_value), *(node -> value));
        void* value = calloc(1, node -> derived_value.data_type);
        fill_tensor(ASSIGN(value, 1.0L, node -> derived_value.data_type), node -> derived_value);
        free(value);
        return;
    }

    Tensor diff = alloc_tensor(node -> value -> shape, node -> value -> rank, node -> value -> data_type);
    for (unsigned int i = 0; i < node -> children_count; ++i) {
        derive_node(node -> children[i]); 
        derive_op(node, node -> children[i]);
        SUM_TENSOR(&diff, diff, node -> derived_value);
    }

    copy_tensor(&(node -> derived_value), diff);
    DEALLOCATE_TENSORS(diff);
    
    return;
}

// Derive using reverse-mode
void derive_r_node(GradNode* node, bool is_sink) {
    if (node -> parents_count == 0) return;
    else if (is_sink) {
        void* val = calloc(1, node -> derived_value.data_type);
        ASSIGN(val, 1.0L, node -> derived_value.data_type);
        fill_tensor(val, node -> derived_value);
        free(val);
    }

    for (unsigned int i = 0; i < node -> parents_count; ++i) {
        Tensor temp = empty_tensor(node -> derived_value.data_type);
        copy_tensor(&temp, node -> parents[i] -> derived_value);
        derive_op(node -> parents[i], node);
        SUM_TENSOR(&(node -> parents[i] -> derived_value), node -> parents[i] -> derived_value, temp);
        DEALLOCATE_TENSORS(temp);
        derive_r_node(node -> parents[i], FALSE);
    }
    
    return;
}

GradNode* get_sink(GradNode* node) {
    if (node -> children_count) return get_sink(node -> children[0]);
    else return node;
}

void forward_pass(GradNode* node) {
    GradNode* child = node -> children[0];
    
    OperatorFlag op_flag = child -> operation;
    if (op_flag == TANH || op_flag == EXP || op_flag == LOG) op_tensor(child -> value, *(node -> value), (Tensor) {.data_type = node -> derived_value.data_type}, op_flag);
    else if (op_flag == POW || op_flag == SQRT) op_tensor(child -> value, *(node -> value), (Tensor) {.data = child -> exp, .data_type = node -> derived_value.data_type}, op_flag);
    else op_tensor(child -> value, *(child -> parents[0] -> value), *(child -> parents[1] -> value), op_flag);
    
    if (child -> children_count) forward_pass(child);
    
    return;
}

#endif //_AUTOGRAD_H_