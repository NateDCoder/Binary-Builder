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
    show() {
        for (let i = 0; i < this.layers.length; i++) {
            this.layers[i].update();
            this.layers[i].show();
            this.layers[i].showGUI();
        }
    }
    mousePressed(x, y) {
        for (let i = 0; i < this.layers.length; i++) {
            this.layers[i].mousePressed(x, y);
        }
    }
    mouseReleased() {
        for (let i = 0; i < this.layers.length; i++) {
            this.layers[i].mouseReleased();
        }
    }
}