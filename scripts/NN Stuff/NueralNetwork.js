class NueralNetwork {
    constructor(layers) {
        this.layers = layers;
    }
    async init() {
        this.inputA = await this.getJSONData("inputA.json");
        this.inputB = await this.getJSONData("inputB.json");
        this.operators = await this.getJSONData("tableOperators.json");
        console.log(this.inputA);
    }

    async getJSONData(filepath) {
        try {
            const response = await fetch(filepath);
            const data = await response.json();
            return data;
        } catch (error) {
            console.error("Error fetching JSON:", error);
        }
    }
    updateTimeStep(timeStep) {
        for (let i = 0; i < this.layers.length; i++) {
            for (let j = 0; j < this.inputA[i].length; j++) {
                for (let k = 0; k < this.inputA[i][j].length; k++) {
                    this.layers[i].input_A_probs[k][j] = this.inputA[i][j][k][timeStep]
                    this.layers[i].input_B_probs[k][j] = this.inputB[i][j][k][timeStep]
                }
            }
            for (let j = 0; j < this.operators[i].length; j++) {
                for (let k = 0; k < this.operators[i][j].length; k++) {
                    this.layers[i].table_probs[k][j] = this.operators[i][j][k][timeStep]
                }
            }
        }
        console.log(binaryInputs)
        transpose(this.forward(transpose(binaryInputs)))
    }
    forward(input) {
        let prev_output = input;
        for (let i = 0; i < this.layers.length; i++) {
            let layer = this.layers[i];
            prev_output = layer.forward(prev_output);
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
