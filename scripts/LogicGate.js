class LogicGate {
  constructor(x, y, type) {
    this.position = createVector(x, y);
    this.type = type;
    this.inputASupplier = () => false;
    this.inputBSupplier = () => false;
    this.draggable = false;

    this.inputALine = null;
    this.inputBLine = null;

    this.inputAIndex = null;
    this.inputBIndex = null;
  }
  show() {
    if (this.draggable) {
      this.position = createVector(mouseX, mouseY);
    }
    switch (this.type) {
      case "and":
        drawAndGate(this.position.x, this.position.y, 50, 30, this.output());
        break;
      case "not":
        drawNotGate(this.position.x, this.position.y, 30, 30, this.output());
        break;
      case "or":
        drawOrGate(
          this.position.x - 20,
          this.position.y - 35 / 2,
          40,
          35,
          this.output()
        );
        break;
      case "xor":
        drawXORGate(
          this.position.x - 20,
          this.position.y - 35 / 2,
          40,
          35,
          this.output()
        );
        break;
    }
  }
  output() {
    switch (this.type) {
      case "and":
        return this.inputASupplier() && this.inputBSupplier();
      case "not":
        return !this.inputASupplier();
      case "or":
        return this.inputASupplier() || this.inputBSupplier();
      case "xor":
        return (
          (this.inputASupplier() || this.inputBSupplier()) &&
          !(this.inputASupplier() && this.inputBSupplier())
        );
    }
  }

  assignInput(inputSupplier, y, line, index) {
    if (this.type !== "not") {
      if (y - this.position.y < 0) {
        // Remove the old line if overwriting
        if (this.inputALine) {
          lines = lines.filter((line) => line !== this.inputALine);
        }
        this.inputAIndex = index;
        this.inputASupplier = inputSupplier;
        this.inputALine = line; // Store the reference to the new line
      } else {
        if (this.inputBLine) {
          lines = lines.filter((line) => line !== this.inputBLine);
        }
        this.inputBIndex = index;
        this.inputBSupplier = inputSupplier;
        this.inputBLine = line;
      }
    } else {
      if (this.inputALine) {
        lines = lines.filter((line) => line !== this.inputALine);
      }
      this.inputAIndex = index;
      this.inputASupplier = inputSupplier;
      this.inputALine = line;
    }
  }
  intersect(x, y) {
    return isPointInRect(
      x,
      y,
      this.position.x - 25,
      this.position.y - 15,
      50,
      30
    );
  }
  toJSON() {
    return {
      type: this.type,
      x: this.position.x,
      y: this.position.y,
      inputALine: this.inputALine ? this.inputALine.toJSON() : null,
      inputBLine: this.inputBLine ? this.inputBLine.toJSON() : null,
      inputAIndex: this.inputAIndex !== null ? this.inputAIndex : null,
      inputBIndex: this.inputBIndex !== null ? this.inputBIndex : null,
      // Do not store the function, but relevant data that can rebuild it later
    };
  }

  static fromJSON(data, restoreSupplierFunc) {
    const gate = new LogicGate(data.x, data.y, data.type);
    if (data.inputALine) {
      gate.inputALine = Line.fromJSON(data.inputALine, restoreSupplierFunc);
    }
    if (data.inputBLine) {
      gate.inputBLine = Line.fromJSON(data.inputBLine, restoreSupplierFunc);
    }
    return gate;
  }
}
