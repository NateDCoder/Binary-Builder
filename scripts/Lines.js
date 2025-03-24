class SampleLine {
  constructor(startXSupplier, startYSupplier, inputSupllier) {
    this.startXSupplier = startXSupplier;
    this.startYSupplier = startYSupplier;
    this.inputSupllier = inputSupllier;
  }

  show() {
    if (this.inputSupllier) {
      this.inputSupllier() ? stroke(255, 255, 0) : stroke(100);
    } else {
      stroke(100);
    }
    strokeWeight(3);
    line(this.startXSupplier(), this.startYSupplier(), mouseX, mouseY);
  }
}

class Line {
  constructor(
    startXSupplier,
    startYSupplier,
    endXSupplier,
    endYSupplier,
    inputSupllier
  ) {
    this.startXSupplier = startXSupplier;
    this.startYSupplier = startYSupplier;
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
    line(
      this.startXSupplier(),
      this.startYSupplier(),
      this.endXSupplier(),
      this.endYSupplier()
    );
  }
}
