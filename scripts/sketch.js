const NUM_OF_BITS = 8;
var number1, number2;
var logicGates = [];
var lines = [];
var contextMenu; // Create a variable for the ContextMenu
var menuOptions = ["and", "or", "not"];

var sampleLine = null;
function setup() {
  let canvas = createCanvas(600, 600);
  canvas.elt.oncontextmenu = () => false;
  number1 = new BinaryDisplay(134, 30, 40 + 15);
  number2 = new BinaryDisplay(1, 30, height / 2 + 20 + 15);

  contextMenu = new ContextMenu(menuOptions);
}

function draw() {
  background(0);
  number1.show();
  number1.update();

  number2.show();
  number2.update();

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
          () => false
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
        lines.push(
          new Line(
            sampleLine.start.x,
            sampleLine.start.y,
            () => logicGates[i].position.x,
            () => logicGates[i].position.y,
            sampleLine.inputSupllier
          )
        );
      }
    }
  }
  sampleLine = null;
}
function keyPressed() {
  if (keyCode === BACKSPACE) {
    for (let i = 0; i < logicGates.length; i++) {
      if (logicGates[i].intersect(mouseX, mouseY)) {
        logicGates.splice(i, 1); // Remove the element at index i
        i--; // Adjust the index after removal to prevent skipping elements
      }
    }
  }
}

function dec2Bin(dec) {
  if (dec >= 256) {
    return new Number(255).toString(2);
  }
  if (dec >= 0) {
    return dec.toString(2);
  } else {
    return (~dec).toString(2);
  }
}

class BinaryDisplay {
  constructor(number, startX, startY) {
    this.number = number;
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

    // Padding for leading zeros
    let padding = NUM_OF_BITS - this.binary.length;
    for (let i = 0; i < padding; i++) {
      fill(255); // Dim the padding zeros
      text(0, this.position.x, i * this.SCALE + this.position.y);
    }

    // Main binary numbers
    for (let i = 0; i < this.binary.length; i++) {
      int(this.binary[i]) ? fill(0, 255, 0) : fill(255);
      text(
        this.binary[i],
        this.position.x,
        (i + padding) * this.SCALE + this.position.y
      );
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
      text(this.number, this.position.x, this.position.y - this.SCALE);
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
    this.draggable = false;
  }
  show() {
    if (this.draggable) {
      this.position = createVector(mouseX, mouseY);
    }
    switch (this.type) {
      case "and":
        drawAndGate(this.position.x, this.position.y, 50, 30);
        break;
      case "not":
        drawNotGate(this.position.x, this.position.y, 30, 30);
        break;
      case "or":
        drawOrGate(this.position.x - 20, this.position.y - 35 / 2, 40, 35);
        break;
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
