class Neuron {
    constructor(aIndex, bIndex) {
        this.aIndex = aIndex;
        this.bIndex = bIndex;
        this.weights = Array.from({ length: 16 }, () => Math.random());
    }

    softmax(weights) {
        let expValues = weights.map(w => Math.exp(w));
        let sumExp = expValues.reduce((a, b) => a + b, 0);
        return expValues.map(v => v / sumExp);
    }

    output(A, B) {
        this.probabilities = this.softmax(this.weights);
        this.weightGradient = Array.from({ length: 16 }, (_, i) => tableOperator(A, B, i));
        this.aDelta = Array.from({ length: 16 }, (_, i) => derivativeA(A, B, i));
        this.bDelta = Array.from({ length: 16 }, (_, i) => derivativeB(A, B, i));

        return this.weightGradient.reduce((sum, val, i) => sum + val * this.probabilities[i], 0);
    }

    updateWeights(lr) {
        this.weights = this.weights.map((w, i) => w - this.weightGradient[i] * lr);
    }
}

class Layer {
    constructor(size, prevSize) {
        this.size = size;
        this.layer = [];
        for (let i = 0; i < size; i++) {
            let indexA = Math.floor(Math.random() * prevSize);
            let indexB = Math.floor(Math.random() * (prevSize - 1));

            if (indexA === indexB) indexB = prevSize - 1;

            this.layer.push(new Neuron(indexA, indexB));
        }
    }

    layerOutput(prevLayerOutput) {
        return this.layer.map(neuron => neuron.output(prevLayerOutput[neuron.aIndex], prevLayerOutput[neuron.bIndex]));
    }

    updateWeightDelta(error) {
        console.log(error);
        this.layer.forEach((neuron, i) => {
            neuron.weightGradient = neuron.weightGradient.map(g => g * error[i]);
            neuron.updateWeights(0.1);
        });
    }
}

class NeuralNetwork {
    constructor(modelSize) {
        this.layers = [];
        for (let i = 0; i < modelSize.length - 2; i++) {
            this.layers.push(new Layer(modelSize[i + 1], modelSize[i]));
        }
    }

    forward(input) {
        let hidden = [this.layers[0].layerOutput(input)];
        for (let i = 1; i < this.layers.length; i++) {
            hidden.push(this.layers[i].layerOutput(hidden[hidden.length - 1]));
        }
        return hidden[hidden.length - 1];
    }

    backwardProp(error) {
        this.layers[this.layers.length - 1].updateWeightDelta(error);
    }
}