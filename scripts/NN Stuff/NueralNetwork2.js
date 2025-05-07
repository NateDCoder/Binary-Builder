class NueralNetwork {
  constructor(networkSize, input) {
    this.networkSize = networkSize;
    this.layers = [];
    this.layer = [];
    for (let i = 0; i < input.length; i++) {
      this.layer.push({
        forward: () => {
          return int(number1.binary[i]);
        },
        position: createVector(
          number1.position.x + 30 / 2,
          number1.position.y + i * 30
        ),
      });
    }
    this.layers.push(this.layer);
    console.log(this.layers);
    for (let i = 1; i < networkSize.length - 1; i++) {
      let layer = [];
      for (let j = 0; j < networkSize[i]; j++) {
        let inputA = int(rng() * networkSize[i - 1]);
        let inputB = int(rng() * (networkSize[i - 1] - 1));

        if (inputA == inputB) {
          inputB = networkSize[i - 1] - 1;
        }

        layer.push(
          new Neuron(
            () => this.layers[i - 1][inputA].forward(),
            () => this.layers[i - 1][inputB].forward(),
            createVector(250, (j + 2) * 50),
            this.layers[i - 1][inputA].position,
            this.layers[i - 1][inputB].position
          )
        );
      }
      this.layers.push(layer);
    }
  }
  output() {
    let output = [];
    for (let neuron of this.layers[this.layers.length - 1]) {
      output.push(neuron.forward());
    }
    return output;
  }
  show() {
    this.layers[this.layers.length - 1].map((element, index) => {
      stroke(lerpColor(color(100), color(255, 255, 0), element.forward()));
      line(
        element.position.x,
        element.position.y,
        output.position.x,
        output.position.y + index * 30
      );
    });
    for (let i = 1; i < this.layers.length; i++) {
      this.layers[i].map((element) => element.show());
      this.layers[i].map((element) => element.showGUI());
    }
  }
  clear() {
    for (let i = 1; i < this.layers.length; i++) {
        this.layers[i].map((element) => element.clear()); 
    }
  }
  update(error) {
    console.log(error);
    for (let i = 1; i < this.layers.length; i++) {
        this.layers[i].map((element) => element.backward(error, 0.01)); 
    }
  }

  applyGradients() {
    for (let i = 1; i < this.layers.length; i++) {
        this.layers[i].map((element) => element.applyGradients()); 
    }
  }
}
