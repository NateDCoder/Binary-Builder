class NueralNetwork {
    constructor(layers) {
        this.layers = layers
    }
    forward(input) {
        let prev_output = input
        for (let i = 0; i < this.layers.length; i++) {
            let layer = this.layers[i];
            prev_output = layer.forward(prev_output)
        }
        return prev_output;
    }
}