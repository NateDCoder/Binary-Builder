const NUM_OF_BITS = 8;
var number1, number2, output;
var logicGates = [];
var lines = [];
var contextMenu; // Create a variable for the ContextMenu
var menuOptions = ["and", "or", "not", "xor"];

var sampleLine = null;
function setup() {
  let canvas = createCanvas(600, 600);
  canvas.elt.oncontextmenu = () => false;
  number1 = new BinaryDisplay(1, 30, 40 + 15);
  number2 = new BinaryDisplay(1, 30, height / 2 + 20 + 15);

  output = new OutputBinaryDisplay(width - 30, height / 2 - 30 * 4);

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
          () => logicGates[i].position.x + 15,
          () => logicGates[i].position.y,
          () => logicGates[i].output(),
          logicGates[i]
        );
      } else {
        logicGates[i].draggable = true;
      }
    }
  }
  if (number1.intersect(mouseButton) || number2.intersect(mouseButton)) return;

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
      case 3:
        logicGates.push(new LogicGate(contextData.x, contextData.y, "xor"));
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
            sampleLine.startXSupplier,
            sampleLine.startYSupplier,
            () => logicGates[i].position.x - 25,
            () =>
              logicGates[i].type !== "not"
                ? my - lgY < 0
                  ? logicGates[i].position.y - 8
                  : logicGates[i].position.y + 8
                : logicGates[i].position.y,
            sampleLine.inputSupllier,
            sampleLine.logicGate
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
      let endPoint = createVector(output.endPoint.x, output.endPoint.y);
      lines.push(
        new Line(
          sampleLine.startXSupplier,
          sampleLine.startYSupplier,
          () => endPoint.x,
          () => endPoint.y,
          sampleLine.inputSupllier
        )
      );

      if (output.inputLines[output.intersectedBit]) {
        lines = lines.filter(
          (line) => line !== output.inputLines[output.intersectedBit]
        );
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
        for (let j = 0; j < lines.length; j++) {
          if (lines[j].logicGate === logicGates[i]) {
            lines.splice(j, 1);
            j--;
          }
        }
        logicGates.splice(i, 1); // Remove the element at index i
        i--; // Adjust the index after removal to prevent skipping elements
      }
    }
  }
}
