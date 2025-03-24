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
  intersect(mouseButton) {
    if (mouseButton === RIGHT) {
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
            () => this.position.x + this.SCALE / 2,
            () => this.position.y + i * this.SCALE,
            () => int(this.binary[i]) == 1
          );
          return true;
        }
      }
      return false;
    } else {
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
          this.binary = replaceBit(
            this.binary,
            i,
            this.binary[i] == "1" ? "0" : "1"
          );
          return true;
        }
      }
      return false;
    }
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

