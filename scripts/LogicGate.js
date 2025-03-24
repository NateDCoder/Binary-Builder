class LogicGate {
    constructor(x, y, type) {
      this.position = createVector(x, y);
      this.type = type;
      this.inputASupplier = () => false;
      this.inputBSupplier = () => false;
      this.draggable = false;
  
      this.inputALine = null;
      this.inputBLine = null;
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
  
    assignInput(inputSupplier, y, line) {
      if (this.type !== "not") {
        if (y - this.position.y < 0) {
          // Remove the old line if overwriting
          if (this.inputALine) {
            lines = lines.filter((line) => line !== this.inputALine);
          }
          this.inputASupplier = inputSupplier;
          this.inputALine = line; // Store the reference to the new line
        } else {
          if (this.inputBLine) {
            lines = lines.filter((line) => line !== this.inputBLine);
          }
          this.inputBSupplier = inputSupplier;
          this.inputBLine = line;
        }
      } else {
        if (this.inputALine) {
          lines = lines.filter((line) => line !== this.inputALine);
        }
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
  }
  
  
  