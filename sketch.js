const NUM_OF_BITS = 8;
var number1, number2;
var andGate;
var logicGates = [];
function setup() {
  let canvas = createCanvas(600, 600);
  canvas.elt.oncontextmenu = () => false;
  number1 = new BinaryDisplay(134, 30, 40 + 15);
  number2 = new BinaryDisplay(1, 30, height / 2 + 20 + 15);
  andGate = new LogicGate(width / 2, height / 2, "or");
}

function draw() {
  background(0);
  number1.show();
  number1.update();

  number2.show();
  number2.update();

  andGate.show();
}

function mousePressed() {
  if (mouseButton === RIGHT) {
    console.log("Right click detected!");
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

function drawAndGate(x, y, w, h) {
  // Draw the rectangle part (left side)
  fill(100);
  noStroke();
  strokeWeight(2);
  rect(x - w / 2, y - h / 2, w / 2, h);
  // Draw the rounded arc part (right side)
  arc(x, y, h, h, -HALF_PI, HALF_PI);
}

function drawOrGate(posX, posY, sizeX, sizeY) {
  fill(100);
  noStroke();

  beginShape();

  // Top curve (outer arc)
  vertex(posX, posY);
  bezierVertex(
    posX + sizeX * 0.25,
    posY,
    posX + sizeX * 0.75,
    posY,
    posX + sizeX,
    posY + sizeY / 2
  );

  // Bottom curve (outer arc)
  bezierVertex(
    posX + sizeX * 0.75,
    posY + sizeY,
    posX + sizeX * 0.25,
    posY + sizeY,
    posX,
    posY + sizeY
  );

  // Inner curve (concave part)
  bezierVertex(
    posX + sizeX / 4,
    posY + sizeY / 2,
    posX + sizeX / 4,
    posY + sizeY / 2,
    posX,
    posY
  );

  endShape(CLOSE);
}

function drawNotGate(x, y, w, h) {
  fill(100);
  noStroke();

  // Draw triangle for NOT gate
  beginShape();
  vertex(x - w / 2, y - h / 2);
  vertex(x - w / 2, y + h / 2);
  vertex(x + w / 2, y);
  endShape(CLOSE);

  fill(255, 255, 0);
  // Draw small circle at the tip (inverter)
  circle(x + w / 2 + 7, y, 8);
}
