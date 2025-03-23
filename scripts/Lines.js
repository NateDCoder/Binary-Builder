class SampleLine {
  constructor(startX, startY, inputSupllier) {
    this.start = createVector(startX, startY);
    this.inputSupllier = inputSupllier;
  }

  show() {
    if (this.inputSupllier) {
      this.inputSupllier() ? stroke(255, 255, 0) : stroke(100);
    } else {
      stroke(100);
    }
    strokeWeight(3);
    line(this.start.x, this.start.y, mouseX, mouseY);
  }
}

class Line {
  constructor(startX, startY, endXSupplier, endYSupplier, inputSupllier) {
    this.start = createVector(startX, startY);
    this.endXSupplier = endXSupplier;
    this.endYSupplier = endYSupplier;
    this.inputSupllier = inputSupllier;
  }

  show() {
    if (this.inputSupllier) {
      this.inputSupllier() ? stroke(255, 255, 0) : stroke(100);
    } else {
      stroke(100);
    }

    strokeWeight(3);
    line(this.start.x, this.start.y, this.endYSupplier(), this.endYSupplier());
  }
}
