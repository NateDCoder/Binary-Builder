function drawAndGate(x, y, w, h, input) {
    // Draw the rectangle part (left side)
    input ? fill(255, 255, 0) : fill(100);
    noStroke();
    strokeWeight(2);
    rect(x - w / 2, y - h / 2, w / 2, h);
    // Draw the rounded arc part (right side)
    arc(x, y, h, h, -HALF_PI, HALF_PI);
  }
  
  function drawOrGate(posX, posY, sizeX, sizeY, input) {
    input ? fill(255, 255, 0) : fill(100);
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
  
  function drawNotGate(x, y, w, h, input) {
    input ?  fill(100) : fill(255, 255, 0);
    noStroke();
  
    // Draw triangle for NOT gate
    beginShape();
    vertex(x - w / 2, y - h / 2);
    vertex(x - w / 2, y + h / 2);
    vertex(x + w / 2, y);
    endShape(CLOSE);
  
    input ? fill(255, 255, 0) : fill(100);
    // Draw small circle at the tip (inverter)
    circle(x + w / 2 + 3, y, 8);
  }