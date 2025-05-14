const NUM_OF_BITS = 8;
var number1, number2, output, answer;
var logicGates = [];
var lines = [];
var contextMenu; // Create a variable for the ContextMenu
const menuOptions = ["and", "or", "not", "xor"];

var sampleLine = null;

var nn;
function setup() {
    let canvas = createCanvas(600, 600);
    canvas.elt.oncontextmenu = () => false;
    number1 = new BinaryDisplay(inputs[currentIndex], 30, height / 2 - 30 * 4);
    // number1 = new BinaryDisplay(1, 30, 40 + 15);
    // number2 = new BinaryDisplay(1, 30, height / 2 + 20 + 15);

    output = new OutputBinaryDisplay(width - 100, height / 2 - 30 * 4);
    answer = new BinaryDisplay(outputs[currentIndex], width - 60, height / 2 - 30 * 4);

    contextMenu = new ContextMenu(menuOptions);
    nn = new NueralNetwork([
        new LogicLayer(4, 5, INPUT_A_PROBS_0, INPUT_B_PROBS_0, TABLE_PROBS_0, 0),
        new LogicLayer(5, 5, INPUT_A_PROBS_1, INPUT_B_PROBS_1, TABLE_PROBS_1, 1),
        new LogicLayer(5, 5, INPUT_A_PROBS_2, INPUT_B_PROBS_2, TABLE_PROBS_2, 2),
        new LogicLayer(5, 4, INPUT_A_PROBS_3, INPUT_B_PROBS_3, TABLE_PROBS_3, 3)
    ]);
    // let sliced = transpose(binaryInputs)
    //     .slice(4, 8)
    //     .map((row) => row.slice(0, 15));
    console.log(transpose(nn.forward(transpose(binaryInputs))));
}

function draw() {
    background(0);
    number1.show();
    number1.update(mouseX, mouseY);

    // number2.show();
    // number2.update(mouseX, mouseY);

    answer.show();
    answer.update(mouseX, mouseY);

    // if (sampleLine) {
    //   sampleLine.show();
    // }
    // for (let line of lines) {
    //   line.show();
    // }

    // for (let logicGate of logicGates) {
    //   logicGate.show();
    // }
    contextMenu.draw();
    // nn.clear();
    // let error = []
    // for (let j = 0; j < 128; j++) {
    //   number1.binary = dec2Bin(inputs[j * 2]);
    //   answer.binary = dec2Bin(outputs[j * 2]);

    //   error = []

    //   let nnOutput = nn.output();
    //   for (let i = 0; i < NUM_OF_BITS; i++) {
    //     error.push(nnOutput[i] - int(answer.binary[i]));
    //     output.bitSuppliers[i] = () => round(nnOutput[i]);
    //   }
    //   nn.update(error);
    // }
    // nn.applyGradients();
    nn.show();

    output.show();
    output.update(mouseX, mouseY);
}

function mousePressed() {
    nn.mousePressed(mouseX, mouseY);
    let intersect = false;
    for (let i = 0; i < logicGates.length; i++) {
        if (logicGates[i].intersect(mouseX, mouseY)) {
            intersect = true;
            if (mouseButton === RIGHT) {
                sampleLine = new SampleLine(
                    () => logicGates[i].position.x + 15,
                    () => logicGates[i].position.y,
                    () => logicGates[i].output(),
                    true,
                    i
                );
            } else {
                logicGates[i].draggable = true;
            }
            break;
        }
    }
    if (number1.intersect(mouseButton)) return; // || number2.intersect(mouseButton)) return;
    if (mouseButton === RIGHT && !intersect) {
        contextMenu.show(mouseX, mouseY);
    }
}
function handleContextWindow() {
    let contextData = contextMenu.handleClick(mouseX, mouseY);
    console.log(contextData);
    if (contextData)
        logicGates.push(new LogicGate(contextData.x, contextData.y, menuOptions[contextData.i]));
}
function turnOffDraggable() {
    for (let logicGate of logicGates) {
        logicGate.draggable = false;
    }
}
function mouseReleased() {
    turnOffDraggable();
    handleContextWindow();
    nn.mouseReleased();
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
                    sampleLine.inputSupplier,
                    sampleLine.logicGateInput,
                    sampleLine.index
                )
            );
            logicGate.assignInput(
                sampleLine.inputSupplier,
                mouseY,
                lines[lines.length - 1],
                lines.length - 1
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
                sampleLine.inputSupplier,
                sampleLine.logicGateInput,
                sampleLine.index
            )
        );

        if (output.inputLines[output.intersectedBit]) {
            lines = lines.filter((line) => line !== output.inputLines[output.intersectedBit]);
        }
        console.log("IntersectedBit", output.intersectedBit);
        output.inputLines[output.intersectedBit] = lines[lines.length - 1];
        output.lineIndexs[output.intersectedBit] = lines.length - 1;
        output.bitSuppliers[output.intersectedBit] = sampleLine.inputSupplier;
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
    } else if (keyCode === UP_ARROW) {
        currentIndex = Math.min(255, ++currentIndex);
        number1.binary = dec2Bin(inputs[currentIndex]);
        answer.binary = dec2Bin(outputs[currentIndex]);
    } else if (keyCode === DOWN_ARROW) {
        currentIndex = Math.max(0, --currentIndex);
        number1.binary = dec2Bin(inputs[currentIndex]);
        answer.binary = dec2Bin(outputs[currentIndex]);
    } else if (key == "l" || key == "L") {
        loadProgress();
    } else if (key == "s" || key == "S") {
        saveProgress();
    }
}

function saveProgress() {
    const state = {
        logicGates: logicGates.map((gate) => gate.toJSON()),
        lines: lines.map((line) => line.toJSON()),
        output: output.toJSON(),
        currentIndex: currentIndex,
        number1Binary: number1.binary,
        answerBinary: answer.binary,
    };

    localStorage.setItem("circuitState", JSON.stringify(state));
    console.log("Progress saved.");
}
const restoreSupplierFunc = (value) => {
    return value; // This can be customized depending on how the suppliers are structured
};
async function loadProgress() {
    var savedState = localStorage.getItem("circuitState");
    if (!savedState) {
        try {
            savedState = await (await fetch("adder.json")).json();
        } catch (e) {}
    }
    console.log(savedState);
    if (savedState) {
        try {
            var state = JSON.parse(savedState);
        } catch (e) {
            var state = savedState;
        }
        for (let i = 0; i < state.logicGates.length; i++) {
            console.log(state.logicGates[i]);
        }

        // Restore logic gates
        logicGates = state.logicGates.map((data) => {
            const gate = new LogicGate(data.x, data.y, data.type);
            if (data.inputALine)
                gate.inputALine = Line.fromJSON(data.inputALine, restoreSupplierFunc);
            if (data.inputBLine)
                gate.inputBLine = Line.fromJSON(data.inputBLine, restoreSupplierFunc);
            return gate;
        });

        // Restore lines
        lines = state.lines.map((lineData) => Line.fromJSON(lineData, restoreSupplierFunc));

        for (let i = 0; i < lines.length; i++) {
            if (lines[i].logicGateInput) {
                lines[i].inputSupplier = () => logicGates[lines[i].index].output();
            } else {
                lines[i].inputSupplier = () => int(number1.binary[lines[i].index]) == 1;
            }
        }

        for (let i = 0; i < state.logicGates.length; i++) {
            console.log(state.logicGates[i]);
            if (state.logicGates[i].inputAIndex !== null) {
                logicGates[i].inputAIndex = state.logicGates[i].inputAIndex;
                logicGates[i].inputASupplier = () =>
                    lines[state.logicGates[i].inputAIndex].inputSupplier();
            }
            if (state.logicGates[i].inputBIndex !== null) {
                logicGates[i].inputBIndex = state.logicGates[i].inputBIndex;
                logicGates[i].inputBSupplier = () =>
                    lines[state.logicGates[i].inputBIndex].inputSupplier();
            }
        }
        console.log(state.output.lineIndexs);
        for (let i = 0; i < state.output.lineIndexs.length; i++) {
            if (state.output.lineIndexs[i] !== null) {
                output.inputLines[i] = lines[state.output.lineIndexs[i]];
                output.lineIndexs[i] = state.output.lineIndexs[i];
                output.bitSuppliers[i] = () => lines[state.output.lineIndexs[i]].inputSupplier();
            }
        }

        // Restore binary displays
        currentIndex = state.currentIndex;
        number1.binary = state.number1Binary;
        answer.binary = state.answerBinary;

        console.log("Progress loaded.");
    } else {
        console.log("No saved progress found.");
    }
}
