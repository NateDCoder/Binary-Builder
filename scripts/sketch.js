const NUM_OF_BITS = 8;
var number1, number2, output;
var logicGates = [];
var lines = [];
var contextMenu; // Create a variable for the ContextMenu
var menuOptions = ["and", "or", "not"];

var sampleLine = null;
function setup() {
  let canvas = createCanvas(600, 600);
  canvas.elt.oncontextmenu = () => false;
  number1 = new BinaryDisplay(0, 30, 40 + 15);
  number2 = new BinaryDisplay(0, 30, height / 2 + 20 + 15);

  output = new outputBinaryDisplay(width - 30, height / 2 - 30 * 4);

  contextMenu = new ContextMenu(menuOptions);
}

function draw() {
  background(0);
  number1.show();
  number1.update();

  number2.show();
  number2.update();

  output.show();
  output.update();

  if (sampleLine) {
    sampleLine.show();
  }
  for (let line of lines) {
    line.show();
  }

  for (let logicGate of logicGates) {
    logicGate.show();
  }
  contextMenu.draw();
}

function mousePressed() {
  let intersect = false;
  for (let i = 0; i < logicGates.length; i++) {
    if (logicGates[i].intersect(mouseX, mouseY)) {
      intersect = true;
      if (mouseButton === RIGHT) {
        sampleLine = new SampleLine(
          logicGates[i].position.x + 15,
          logicGates[i].position.y,
          () => logicGates[i].output()
        );
      } else {
        logicGates[i].draggable = true;
      }
    }
  }
  if (mouseButton === RIGHT && !intersect) {
    if (number1.intersect() || number2.intersect()) return;
  }
  if (mouseButton === RIGHT && !intersect) {
    contextMenu.show(mouseX, mouseY);
  }
}

function mouseReleased() {
  for (let logicGate of logicGates) {
    logicGate.draggable = false;
  }
  let contextData = contextMenu.handleClick(mouseX, mouseY);
  console.log(contextData);
  if (contextData) {
    switch (contextData.i) {
      case 0:
        logicGates.push(new LogicGate(contextData.x, contextData.y, "and"));
        break;
      case 1:
        logicGates.push(new LogicGate(contextData.x, contextData.y, "or"));
        break;
      case 2:
        logicGates.push(new LogicGate(contextData.x, contextData.y, "not"));
        break;
    }
  }
  if (sampleLine) {
    for (let i = 0; i < logicGates.length; i++) {
      if (logicGates[i].intersect(mouseX, mouseY)) {
        let my = mouseY;
        let lgY = logicGates[i].position.y;
        lines.push(
          new Line(
            sampleLine.start.x,
            sampleLine.start.y,
            () => logicGates[i].position.x - 25,
            () =>
              logicGates[i].type == "or" || logicGates[i].type == "and"
                ? my - lgY < 0
                  ? logicGates[i].position.y - 8
                  : logicGates[i].position.y + 8
                : logicGates[i].position.y,
            sampleLine.inputSupllier
          )
        );
        logicGates[i].assignInput(
          sampleLine.inputSupllier,
          mouseY,
          lines[lines.length - 1]
        );
      }
    }
    if (output.intersect()) {
      console.log(sampleLine);
      console.log(sampleLine.inputSupllier());
      let endPoint = createVector(output.endPoint.x, output.endPoint.y)
      lines.push(
        new Line(
          sampleLine.start.x,
          sampleLine.start.y,
          () => endPoint.x,
          () => endPoint.y,
          sampleLine.inputSupllier
        )
      );

      if (output.inputLines[output.intersectedBit]) {
        lines = lines.filter((line) => line !== output.inputLines[output.intersectedBit]);
      }
      output.inputLines[output.intersectedBit] = lines[lines.length - 1];

      output.bitSuppliers[output.intersectedBit] = sampleLine.inputSupllier;
    }
  }
  sampleLine = null;
}
function keyPressed() {
  if (keyCode === BACKSPACE) {
    for (let i = 0; i < logicGates.length; i++) {
      if (logicGates[i].intersect(mouseX, mouseY)) {
        if (logicGates[i].inputALine) {
          lines = lines.filter((line) => line !== logicGates[i].inputALine);
        }
        if (logicGates[i].inputBLine) {
          lines = lines.filter((line) => line !== logicGates[i].inputBLine);
        }
        logicGates.splice(i, 1); // Remove the element at index i
        i--; // Adjust the index after removal to prevent skipping elements
      }
    }
  }
}

function dec2Bin(dec) {
  if (dec >= 256) {
    dec = 255; // Cap at 255 if the number exceeds 8-bit limit
  }
  if (dec >= 0) {
    return dec.toString(2).padStart(8, "0"); // Add leading zeros for 8-bit binary
  } else {
    let binary = ((~-dec + 1) >>> 0).toString(2); // Handle negative numbers with 2's complement
    return binary.slice(-8).padStart(8, "0"); // Ensure it's 8 bits
  }
}

function bin2Dec(bin) {
  if (bin.length !== 8) {
    console.error("Binary string must be 8 bits long.");
    return null;
  }

  // Handle negative 2's complement conversion

  return parseInt(bin, 2);
}

class BinaryDisplay {
  constructor(number, startX, startY) {
    this.binary = dec2Bin(number);
    this.position = createVector(startX, startY);
    this.SCALE = 30;
  }
  show() {
    stroke(80);
    strokeWeight(2);
    for (let i = 0; i < NUM_OF_BITS; i++) {
      fill(46);
      rect(
        this.position.x - this.SCALE / 2,
        this.position.y - this.SCALE / 2 + i * this.SCALE,
        this.SCALE,
        this.SCALE,
        10
      );
    }

    textSize(20);
    textAlign(CENTER, CENTER);

    for (let i = 0; i < this.binary.length; i++) {
      int(this.binary[i]) ? fill(0, 255, 0) : fill(255);
      text(this.binary[i], this.position.x, i * this.SCALE + this.position.y);
    }
  }
  update() {
    if (
      isPointInRect(
        mouseX,
        mouseY,
        this.position.x - this.SCALE / 2,
        this.position.y - this.SCALE / 2,
        this.SCALE,
        this.SCALE * 8
      )
    ) {
      fill(255);
      text(bin2Dec(this.binary), this.position.x, this.position.y - this.SCALE);
    }
  }
  intersect() {
    for (let i = 0; i < NUM_OF_BITS; i++) {
      if (
        isPointInRect(
          mouseX,
          mouseY,
          this.position.x - this.SCALE / 2,
          this.position.y - this.SCALE / 2 + i * this.SCALE,
          this.SCALE,
          this.SCALE
        )
      ) {
        console.log(i);
        sampleLine = new SampleLine(
          this.position.x + this.SCALE / 2,
          this.position.y + i * this.SCALE,
          () => int(this.binary[i]) == 1
        );
        return true;
      }
    }
    return false;
  }
}

function isPointInRect(px, py, rx, ry, rw, rh) {
  return px >= rx && px <= rx + rw && py >= ry && py <= ry + rh;
}

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
    }
  }

  assignInput(inputSupplier, y, line) {
    if (this.type == "or" || this.type == "and") {
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

class outputBinaryDisplay extends BinaryDisplay {
  constructor(startX, startY) {
    super(0, startX, startY);
    this.bitSuppliers = [];
    this.inputLines = []
    for (let i = 0; i < NUM_OF_BITS; i++) {
      this.bitSuppliers.push(() => false);
      this.inputLines.push(null)
    }

    this.inputLine = null;
  }
  update() {
    super.update();
    for (let i = 0; i < NUM_OF_BITS; i++) {
      this.binary = replaceBit(this.binary, i, this.bitSuppliers[i]()?"1":"0");
    }
  }

  intersect() {
    for (let i = 0; i < NUM_OF_BITS; i++) {
      if (
        isPointInRect(
          mouseX,
          mouseY,
          this.position.x - this.SCALE / 2,
          this.position.y - this.SCALE / 2 + i * this.SCALE,
          this.SCALE,
          this.SCALE
        )
      ) {
        this.intersectedBit = i;
        this.endPoint = createVector(
          this.position.x - this.SCALE / 2,
          this.position.y + i * this.SCALE
        );
        return true;
      }
    }
    return false;
  }
}

function replaceBit(bin, position, newBit) {
  if (position < 0 || position >= bin.length) {
    console.error(
      "Invalid position. Must be between 0 and " + (bin.length - 1)
    );
    return bin;
  }
  if (newBit !== "0" && newBit !== "1") {
    console.error("Invalid bit. Must be '0' or '1'.");
    return bin;
  }

  // Replace the bit at the specified position and return the modified string
  return bin.substring(0, position) + newBit + bin.substring(position + 1);
}
