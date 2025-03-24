class BinaryDisplay {
  constructor(number, startX, startY) {
    this.binary = dec2Bin(number);
    this.position = createVector(startX, startY);
    this.SCALE = 30;
    this.rectXStart = this.position.x - this.SCALE / 2;
    this.rectYStarts = [];
    for (let i = 0; i < NUM_OF_BITS; i++) {
      this.rectYStarts.push(this.position.y - this.SCALE / 2 + i * this.SCALE);
    }
  }
  show() {
    stroke(80);
    strokeWeight(2);
    fill(46);
    for (let i = 0; i < NUM_OF_BITS; i++) {
      rect(this.rectXStart, this.rectYStarts[i], this.SCALE, this.SCALE, 10);
    }

    textSize(20);
    textAlign(CENTER, CENTER);

    for (let i = 0; i < this.binary.length; i++) {
      int(this.binary[i]) ? fill(0, 255, 0) : fill(255);
      text(this.binary[i], this.position.x, i * this.SCALE + this.position.y);
    }
  }
  update(x, y) {
    if (
      !isPointInRect(
        x,
        y,
        this.rectXStart,
        this.rectYStarts[0],
        this.SCALE,
        this.SCALE * 8
      )
    )
      return;
    fill(255);
    text(bin2Dec(this.binary), this.position.x, this.position.y - this.SCALE);
  }
  intersect(mouseButton) {
    for (let i = 0; i < NUM_OF_BITS; i++) {
      if (this.isMouseOverBit(i)) {
        if (mouseButton === RIGHT) {
          this.handleRightClick(i);
        } else {
          this.toggleBit(i);
        }
        return true;
      }
    }
    return false;
  }

  isMouseOverBit(i) {
    return isPointInRect(
      mouseX,
      mouseY,
      this.rectXStart,
      this.rectYStarts[i],
      this.SCALE,
      this.SCALE
    );
  }

  handleRightClick(i) {
    console.log(i);
    sampleLine = new SampleLine(
      () => this.position.x + this.SCALE / 2, // X Position Supplier
      () => this.position.y + i * this.SCALE, // Y Position Supplier
      () => int(this.binary[i]) == 1 // Binary state check
    );
  }

  toggleBit(i) {
    this.binary = replaceBit(
      this.binary,
      i,
      this.binary[i] === "1" ? "0" : "1"
    );
  }
}

class OutputBinaryDisplay extends BinaryDisplay {
  constructor(startX, startY) {
    super(0, startX, startY);
    this.bitSuppliers = [];
    this.inputLines = [];
    for (let i = 0; i < NUM_OF_BITS; i++) {
      this.bitSuppliers.push(() => false);
      this.inputLines.push(null);
    }

    this.inputLine = null;
  }
  update() {
    super.update();
    for (let i = 0; i < NUM_OF_BITS; i++) {
      if (this.bitSuppliers[i]) {
        this.binary = replaceBit(
          this.binary,
          i,
          this.bitSuppliers[i]() ? "1" : "0"
        );
      }
    }
  }

  intersect() {
    for (let i = 0; i < NUM_OF_BITS; i++) {
      if (this.isMouseOverBit(i)) {
        this.intersectedBit = i;
        this.endPoint = createVector(
          this.rectXStart,
          this.position.y + i * this.SCALE
        );
        return true;
      }
    }
    return false;
  }
}
