class LogicLayer {
    constructor(prev_size, size, input_A_probs, input_B_probs, table_probs) {
        this.size = size;
        this.prev_size = prev_size;

        this.input_A_probs = input_A_probs;
        this.input_B_probs = input_B_probs;
        this.table_probs = table_probs;
    }

    forward(prev_layer_output) {
        let a_inputs = matmul(this.input_A_probs, prev_layer_output)
        let b_inputs = matmul(this.input_B_probs, prev_layer_output)

        let table_output = tableOpperator(a_inputs, b_inputs);
    
        const result = weightedLogicOutput(table_output, this.table_probs);
        return result;

    }
}
