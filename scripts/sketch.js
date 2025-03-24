const NUM_OF_BITS = 8;
var number1, number2, output;
var logicGates = [];
var lines = [];
var contextMenu; // Create a variable for the ContextMenu
const menuOptions = ["and", "or", "not", "xor"];

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
  number1.update(mouseX, mouseY);

  number2.show();
  number2.update(mouseX, mouseY);

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
  for (let logicGate of logicGates) {
    if (logicGate.intersect(mouseX, mouseY)) {
      intersect = true;
      if (mouseButton === RIGHT) {
        sampleLine = new SampleLine(
          () => logicGate.position.x + 15,
          () => logicGate.position.y,
          () => logicGate.output()
        );
      } else {
        logicGate.draggable = true;
      }
    }
  }
  if (number1.intersect(mouseButton) || number2.intersect(mouseButton)) return;

  if (mouseButton === RIGHT && !intersect) {
    contextMenu.show(mouseX, mouseY);
  }
}
function handleContextWindow() {
  let contextData = contextMenu.handleClick(mouseX, mouseY);
  console.log(contextData);
  if (contextData)
    logicGates.push(
      new LogicGate(contextData.x, contextData.y, menuOptions[contextData.i])
    );
}
function turnOffDraggable() {
  for (let logicGate of logicGates) {
    logicGate.draggable = false;
  }
}
function mouseReleased() {
  turnOffDraggable();
  handleContextWindow();
  if (!sampleLine) return;

  for (let logicGate of logicGates) {
    if (logicGate.intersect(mouseX, mouseY)) {
      let my = mouseY;
      let lgY = logicGate.position.y;
      lines.push(
        new Line(
          sampleLine.startXSupplier,
          sampleLine.startYSupplier,
          () => logicGate.position.x - 25,
          () =>
            logicGate.type !== "not"
              ? my - lgY < 0
                ? logicGate.position.y - 8
                : logicGate.position.y + 8
              : logicGate.position.y,
          sampleLine.inputSupllier
        )
      );
      logicGate.assignInput(
        sampleLine.inputSupllier,
        mouseY,
        lines[lines.length - 1]
      );
    }
  }
  if (output.intersect()) {
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
