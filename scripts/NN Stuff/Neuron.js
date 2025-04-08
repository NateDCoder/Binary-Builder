class Neuron {
  constructor(supplierA, supplierB, position, aPosition, bPosition) {
    this.probabilities = [...Array(16)].map(() => rng());
    this.supplierA = supplierA;
    this.supplierB = supplierB;
    this.gradients = [];
    this.gradientsA = [];
    this.gradientsB = [];
    this.position = position;
    this.aPosition = aPosition;
    this.bPosition = bPosition;
    this.averageGradient = [];
  }

  forward() {
    var percentages = softmax(this.probabilities);
    var sum = 0;
    this.gradients = [];
    this.gradientsA = [];
    this.gradientsB = [];

    for (let i = 0; i < percentages.length; i++) {
      this.gradientsA.push(derivativeA(this.supplierA(), this.supplierB(), i));
      this.gradientsB.push(derivativeB(this.supplierA(), this.supplierB(), i));

      let gradient = tableOpperator(this.supplierA(), this.supplierB(), i);
      sum += percentages[i] * gradient;
      this.gradients.push(gradient);
    }
    return sum;
  }
  backward(outputGradient, LR) {
    this.probabilitiesGradients = this.probabilities.map((element, index) => {
      // console.log(LR, outputGradient[index], this.gradients[index]);
      return LR * outputGradient[index] * this.gradients[index];
    });

    this.averageGradient.push(this.probabilitiesGradients);
  }

  clear () {
    this.probabilitiesGradients = [];
  }
  show() {
    stroke(lerpColor(color(100), color(255, 255, 0), this.supplierA()));
    line(this.aPosition.x, this.aPosition.y, this.position.x, this.position.y);

    stroke(lerpColor(color(100), color(255, 255, 0), this.supplierB()));
    line(this.bPosition.x, this.bPosition.y, this.position.x, this.position.y);

    fill(lerpColor(color(100), color(255, 255, 0), this.forward()));
    noStroke();
    circle(this.position.x, this.position.y, 15);
  }
  showGUI() {
    noStroke();
    fill(180);
    if (dist(mouseX, mouseY, this.position.x, this.position.y) < 15) {
      rect(this.position.x, this.position.y, 50, 50);
      fill(0);
      text(
        int(100 * this.forward()) / 100,
        this.position.x + 25,
        this.position.y + 15
      );
      const maxIndex = this.probabilities.reduce(
        (maxIdx, curr, idx, arr) => (curr > arr[maxIdx] ? idx : maxIdx),
        0
      );
      
      text(
        maxIndex,
        this.position.x + 25,
        this.position.y + 35
      );
    }
  }
  applyGradients() {
    let average = new Array(16).fill(0);
    for (let i = 0; i < this.averageGradient.length; i++) {
      for (let j = 0; j < this.averageGradient[i].length; j++) {
        average[j] += this.averageGradient[i][j];
        console.log(this.averageGradient[i][j]);
      }
    }
    average = average.map((element) => {return element / this.averageGradient.length});
  }
}
