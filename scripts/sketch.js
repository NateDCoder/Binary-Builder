const NUM_OF_BITS = 8;
var number1, number2;
var andGate;
var logicGates = [];

var contextMenu; // Create a variable for the ContextMenu
var menuOptions = ["and", "or", "not"];
function setup() {
  let canvas = createCanvas(600, 600);
  canvas.elt.oncontextmenu = () => false;
  number1 = new BinaryDisplay(134, 30, 40 + 15);
  number2 = new BinaryDisplay(1, 30, height / 2 + 20 + 15);
  andGate = new LogicGate(width / 2, height / 2, "or");

  contextMenu = new ContextMenu(menuOptions);
}

function draw() {
  background(0);
  number1.show();
  number1.update();

  number2.show();
  number2.update();

  andGate.show();
  for (let logicGate of logicGates) {
    logicGate.show();
  }
  contextMenu.draw();
}

function mousePressed() {
  if (mouseButton === RIGHT) {
    contextMenu.show(mouseX, mouseY);
  }
}

function mouseReleased() {
  let contextData = contextMenu.handleClick(mouseX, mouseY);
  console.log(contextData)
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
}

function isPointInRect(px, py, rx, ry, rw, rh) {
  return px >= rx && px <= rx + rw && py >= ry && py <= ry + rh;
}

class LogicGate {
  constructor(x, y, type) {
    this.position = createVector(x, y);
    this.type = type;
  }
  show() {
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
}
