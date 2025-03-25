class SampleLine {
  constructor(
    startXSupplier,
    startYSupplier,
    inputSupplier,
    logicGateInput,
    logicGateIndex
  ) {
    this.startXSupplier = startXSupplier;
    this.startYSupplier = startYSupplier;
    this.inputSupplier = inputSupplier;
    if (logicGateInput) {
      this.logicGateInput = true;
      this.index = logicGateIndex;
    } else {
      this.logicGateInput = false;
      this.index = logicGateIndex;
    }
  }

  show() {
    if (this.inputSupplier) {
      this.inputSupplier() ? stroke(255, 255, 0) : stroke(100);
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
    inputSupplier,
    logicGateInput,
    logicGateIndex
  ) {
    this.startXSupplier = startXSupplier;
    this.startYSupplier = startYSupplier;
    this.endXSupplier = endXSupplier;
    this.endYSupplier = endYSupplier;
    this.inputSupplier = inputSupplier;
    if (logicGateInput) {
      this.logicGateInput = true;
      this.index = logicGateIndex;
    } else {
      this.logicGateInput = false;
      this.index = logicGateIndex;
    }
  }

  show() {
    if (this.inputSupplier) {
      this.inputSupplier() ? stroke(255, 255, 0) : stroke(100);
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
  toJSON() {
    return {
      startX: this.startXSupplier(),
      startY: this.startYSupplier(),
      endX: this.endXSupplier(),
      endY: this.endYSupplier(),
      inputSupplier: this.inputSupplier, // You may need to adjust this if it’s more complex
      logicGateInput: this.logicGateInput,
      index: this.index
    };
  }
  static fromJSON(data, restoreSupplierFunc) {
    // Recreate the suppliers based on the restored data
    const line = new Line(
      () => restoreSupplierFunc(data.startX),
      () => restoreSupplierFunc(data.startY),
      () => restoreSupplierFunc(data.endX),
      () => restoreSupplierFunc(data.endY),
      data.inputSupplier,
      data.logicGateInput,
      data.index
    );
    return line;
  }
}
